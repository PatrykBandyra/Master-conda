import random

import torch
import torch.nn as nn
from einops import rearrange
from gymnasium.spaces import flatdim
from torch import optim


class QNetwork(nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            cfg,
            layers,
            parameter_sharing,
            use_rnn,
            use_orthogonal_init,
            device,
    ):
        super().__init__()
        hidden_dims = list(layers)

        self.action_space = action_space

        self.n_agents = len(obs_space)
        obs_shape = [flatdim(o) for o in obs_space]
        action_shape = [flatdim(a) for a in action_space]

        if not parameter_sharing:
            self.critic = MultiAgentIndependentNetwork(
                obs_shape, hidden_dims, action_shape, use_rnn, use_orthogonal_init
            )
            self.target = MultiAgentIndependentNetwork(
                obs_shape, hidden_dims, action_shape, use_rnn, use_orthogonal_init
            )
        else:
            self.critic = MultiAgentSharedNetwork(
                obs_shape,
                hidden_dims,
                action_shape,
                parameter_sharing,
                use_rnn,
                use_orthogonal_init,
            )
            self.target = MultiAgentSharedNetwork(
                obs_shape,
                hidden_dims,
                action_shape,
                parameter_sharing,
                use_rnn,
                use_orthogonal_init,
            )

        self.hard_update()
        self.to(device)

        for param in self.target.parameters():
            param.requires_grad = False

        if type(cfg.optimizer) is str:
            self.optimizer_class = getattr(optim, cfg.optimizer)
        else:
            self.optimizer_class = cfg.optimizer

        self.optimizer = self.optimizer_class(self.critic.parameters(), lr=cfg.lr)

        self.gamma = cfg.gamma
        self.grad_clip = cfg.grad_clip
        self.device = device
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau
        self.double_q = cfg.double_q

        self.updates = 0
        self.last_target_update = 0

        self.standardise_returns = cfg.standardise_returns
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)

        print(self)

    def forward(self, inputs):
        raise NotImplementedError("Forward not implemented. Use act or update instead!")

    def init_hiddens(self, batch_size):
        return self.critic.init_hiddens(batch_size, self.device)

    def act(self, inputs, hiddens, epsilon, action_masks=None):
        with torch.no_grad():
            inputs = [
                torch.tensor(i, device=self.device).view(1, 1, -1) for i in inputs
            ]
            values, hiddens = self.critic(inputs, hiddens)
        if action_masks is not None:
            masked_values = []
            for value, mask in zip(values, action_masks):
                masked_values.append(value * mask + (1 - mask) * -1e8)
            values = masked_values
        if epsilon > random.random():
            if action_masks is not None:
                # random index of action with mask = 1
                actions = [
                    random.choice([i for i, m in enumerate(mask) if m == 1])
                    for mask in action_masks
                ]
            else:
                actions = self.action_space.sample()
        else:
            actions = [value.argmax(-1).squeeze().cpu().item() for value in values]
        return actions, hiddens

    def _compute_loss(self, batch):
        obss = batch.obss
        actions = batch.actions.unsqueeze(-1)
        rewards = batch.rewards
        dones = batch.dones[1:].unsqueeze(0).repeat(self.n_agents, 1, 1)
        filled = batch.filled
        action_masks = batch.action_mask

        # (n_agents, ep_length, batch_size, n_actions)
        q_values, _ = self.critic(obss, hiddens=None)
        q_values = torch.stack(q_values)
        chosen_q_values = q_values[:, :-1].gather(-1, actions).squeeze(-1)

        # compute target
        with torch.no_grad():
            target_q_values, _ = self.target(obss, hiddens=None)
            target_q_values = torch.stack(target_q_values)[:, 1:]
            if action_masks is not None:
                target_q_values[action_masks[:, 1:] == 0] = -1e8

        if self.double_q:
            q_values_clone = q_values.clone().detach()[:, 1:]
            if action_masks is not None:
                q_values_clone[action_masks[:, 1:] == 0] = -1e8
            a_prime = q_values_clone.argmax(-1)
            target_qs = target_q_values.gather(-1, a_prime.unsqueeze(-1)).squeeze(-1)
        else:
            target_qs, _ = target_q_values.max(dim=-1)

        if self.standardise_returns:
            target_qs = rearrange(target_qs, "A E B -> E B A")
            target_qs = target_qs * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean
            target_qs = rearrange(target_qs, "E B A -> A E B")

        returns = rewards + self.gamma * target_qs.detach() * (1 - dones)

        if self.standardise_returns:
            returns = rearrange(returns, "A E B -> E B A")
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)
            returns = rearrange(returns, "E B A -> A E B")

        loss = torch.nn.functional.mse_loss(
            chosen_q_values, returns.detach(), reduction="none"
        ).sum(dim=0)
        return (loss * filled).sum() / filled.sum()

    def update(self, batch):
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer.step()
        self.updates += 1
        self.update_target()
        return {"loss": loss.item()}

    def update_target(self):
        if (
                self.target_update_interval_or_tau > 1.0
                and (self.updates - self.last_target_update)
                >= self.target_update_interval_or_tau
        ):
            self.hard_update()
            self.last_target_update = self.updates
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

    def soft_update(self, tau):
        for target_param, source_param in zip(
                self.target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                (1 - tau) * target_param.data + tau * source_param.data
            )

    def hard_update(self):
        self.target.load_state_dict(self.critic.state_dict())

class MultiAgentIndependentNetwork(nn.Module):
    def __init__(
            self,
            input_sizes,
            hidden_dims,
            output_sizes,
            use_rnn=False,
            use_orthogonal_init=True,
    ):
        super().__init__()
        assert len(input_sizes) == len(
            output_sizes
        ), "Expect same number of input and output sizes"
        self.independent = nn.ModuleList()

        for in_size, out_size in zip(input_sizes, output_sizes):
            dims = [in_size] + hidden_dims + [out_size]
            self.independent.append(
                make_network(
                    dims, use_rnn=use_rnn, use_orthogonal_init=use_orthogonal_init
                )
            )

    def forward(
            self,
            inputs: Union[List[torch.Tensor], torch.Tensor],
            hiddens: Optional[List[torch.Tensor]] = None,
    ):
        if hiddens is None:
            hiddens = [None] * len(inputs)
        futures = [
            torch.jit.fork(model, x, h)
            for model, x, h in zip(self.independent, inputs, hiddens)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        outs = [x for x, _ in results]
        hiddens = [h for _, h in results]
        return outs, hiddens

    def init_hiddens(self, batch_size, device):
        return [model.init_hiddens(batch_size, device) for model in self.independent]


class MultiAgentSharedNetwork(nn.Module):
    def __init__(
            self,
            input_sizes,
            hidden_dims,
            output_sizes,
            sharing_indices,
            use_rnn=False,
            use_orthogonal_init=True,
    ):
        super().__init__()
        assert len(input_sizes) == len(
            output_sizes
        ), "Expect same number of input and output sizes"
        self.num_agents = len(input_sizes)

        if sharing_indices is True:
            self.sharing_indices = len(input_sizes) * [0]
        elif sharing_indices is False:
            self.sharing_indices = list(range(len(input_sizes)))
        else:
            self.sharing_indices = sharing_indices
        assert len(self.sharing_indices) == len(
            input_sizes
        ), "Expect same number of sharing indices as agents"

        self.num_networks = 0
        self.networks = nn.ModuleList()
        self.agents_by_network = []
        self.input_sizes = []
        self.output_sizes = []
        created_networks = set()
        for i in self.sharing_indices:
            if i in created_networks:
                # network already created
                continue

            # agent indices that share this network
            network_agents = [
                j for j, idx in enumerate(self.sharing_indices) if idx == i
            ]
            in_sizes = [input_sizes[j] for j in network_agents]
            in_size = in_sizes[0]
            assert all(
                idim == in_size for idim in in_sizes
            ), f"Expect same input sizes across all agents sharing network {i}"
            out_sizes = [output_sizes[j] for j in network_agents]
            out_size = out_sizes[0]
            assert all(
                odim == out_size for odim in out_sizes
            ), f"Expect same output sizes across all agents sharing network {i}"

            dims = [in_size] + hidden_dims + [out_size]
            self.networks.append(
                make_network(
                    dims, use_rnn=use_rnn, use_orthogonal_init=use_orthogonal_init
                )
            )
            self.agents_by_network.append(network_agents)
            self.input_sizes.append(in_size)
            self.output_sizes.append(out_size)
            self.num_networks += 1
            created_networks.add(i)

    def forward(
            self,
            inputs: Union[List[torch.Tensor], torch.Tensor],
            hiddens: Optional[List[torch.Tensor]] = None,
    ):
        assert all(
            x.dim() == 3 for x in inputs
        ), "Expect each agent input to be 3D tensor (seq_len, batch, input_size)"
        assert hiddens is None or all(
            x is None or x.dim() == 3 for x in hiddens
        ), "Expect hidden state to be 3D tensor (num_layers, batch, hidden_size)"

        batch_size = inputs[0].size(1)
        assert all(
            x.size(1) == batch_size for x in inputs
        ), "Expect all agent inputs to have same batch size"

        # group inputs and hiddens by network
        network_inputs = []
        network_hiddens = []
        for agent_indices in self.agents_by_network:
            net_inputs = [inputs[i] for i in agent_indices]
            if hiddens is None or all(h is None for h in hiddens):
                net_hiddens = None
            else:
                net_hiddens = [hiddens[i] for i in agent_indices]
            network_inputs.append(torch.cat(net_inputs, dim=1))
            network_hiddens.append(
                torch.cat(net_hiddens, dim=1) if net_hiddens is not None else None
            )

        # forward through networks
        futures = [
            torch.jit.fork(network, x, h)
            for network, x, h in zip(self.networks, network_inputs, network_hiddens)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        outs = [x.split(batch_size, dim=1) for x, _ in results]
        hiddens = [
            h.split(batch_size, dim=1) if h is not None else None for _, h in results
        ]

        # group outputs by agents
        agent_outputs = []
        agent_hiddens = []
        self.idx_by_network = [0] * self.num_networks
        for network_idx in self.sharing_indices:
            idx_within_network = self.idx_by_network[network_idx]
            agent_outputs.append(outs[network_idx][idx_within_network])
            if hiddens[network_idx] is not None:
                agent_hiddens.append(hiddens[network_idx][idx_within_network])
            else:
                agent_hiddens.append(None)
            self.idx_by_network[network_idx] += 1
        return agent_outputs, agent_hiddens

    def init_hiddens(self, batch_size, device):
        return [
            self.networks[network_idx].init_hiddens(batch_size, device)
            for network_idx in self.sharing_indices
        ]

class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device: str = "cpu"):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def update(self, arr):
        arr = arr.reshape(-1, arr.size(-1))
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
                m_a
                + m_b
                + torch.square(delta)
                * self.count
                * batch_count
                / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

