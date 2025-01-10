import argparse
import json
import logging
import os
import random
from argparse import Namespace
from datetime import datetime, timedelta
from typing import Dict, Any, List

import flappy_bird_gymnasium  # noqa: F401 - used indirectly by gymnasium
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import yaml
from torch import nn
from torch.types import Tensor

from dqn import DQN
from experience_buffer import ExperienceBuffer, Experience

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(FORMATTER)
LOGGER.addHandler(stream_handler)

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib.use('Agg')  # Do not render plots on screen

# Hyper params names
LR = 'learning_rate'
DF = 'discount_factor'
STSTN = 'steps_to_sync_target_net'
MBS = 'mini_batch_size'
ED = 'epsilon_decay'
EM = 'epsilon_min'
FC1N = 'fc1_nodes'


class Agent:
    def __init__(self, config_set_name: str, device_name: str, is_training: bool = True, do_render: bool = False,
                 trial: optuna.Trial = None):
        # Configuration ================================================================================================
        with open('config.yml', 'r') as file:
            all_config_sets = yaml.safe_load(file)
            cfg: Dict[str, Any] = all_config_sets[config_set_name]

        self.config_set_name: str = config_set_name
        self.env_id: str = cfg['env_id']

        self.trial: optuna.Trial = trial

        if trial is not None:
            o_cfg: Dict[str, Dict[str, Any]] = cfg['optuna']
            self.learning_rate: float = trial.suggest_float(
                LR, o_cfg[LR]['min'], o_cfg[LR]['max'], step=o_cfg[LR]['step']
            ) if o_cfg[LR] is not None else cfg[LR]
            self.discount_factor: float = trial.suggest_float(
                DF, o_cfg[DF]['min'], o_cfg[DF]['max'], step=o_cfg[DF]['step']
            ) if o_cfg[DF] is not None else cfg[DF]
            self.steps_to_sync_target_net: int = trial.suggest_categorical(
                STSTN, o_cfg[STSTN]['values']
            ) if o_cfg[STSTN] is not None else cfg[STSTN]
            self.mini_batch_size: int = trial.suggest_categorical(
                MBS, o_cfg[MBS]['values']
            ) if o_cfg[MBS] is not None else cfg[MBS]
            self.epsilon_decay: int = trial.suggest_float(
                ED, o_cfg[ED]['min'], o_cfg[ED]['max'], step=o_cfg[ED]['step']
            ) if o_cfg[ED] is not None else cfg[ED]
            self.epsilon_min: int = trial.suggest_float(
                EM, o_cfg[EM]['min'], o_cfg[EM]['max'], step=o_cfg[EM]['step']
            ) if o_cfg[EM] is not None else cfg[EM]
            self.fc1_nodes: int = trial.suggest_categorical(
                FC1N, o_cfg[FC1N]['values']
            ) if o_cfg[FC1N] is not None else cfg[FC1N]
        else:
            self.learning_rate: float = cfg[LR]
            self.discount_factor: float = cfg[DF]
            self.steps_to_sync_target_net: int = cfg[STSTN]
            self.mini_batch_size: int = cfg[MBS]
            self.epsilon_decay: float = cfg[ED]
            self.epsilon_min: float = cfg[EM]
            self.fc1_nodes: int = cfg[FC1N]

        self.exp_buffer_size: int = cfg['exp_buffer_size']
        self.epsilon_init: float = cfg['epsilon_init']
        self.episode_max_reward: float = cfg['episode_max_reward']
        self.episode_max_num: int = cfg['episode_max_num']
        self.env_make_params: Dict[Any, Any] = cfg.get('env_make_params', {})
        self.enable_double_dqn: bool = cfg['enable_double_dqn']
        self.enable_dueling_dqn: bool = cfg['enable_dueling_dqn']
        self.seed: bool = cfg['seed']

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.is_training = is_training

        # Environment ==================================================================================================
        self.env = gym.make(self.env_id, render_mode='human' if do_render else None, **self.env_make_params)
        self.exp_buffer = ExperienceBuffer(self.exp_buffer_size) if is_training else None

        # Neural Network ===============================================================================================
        self.device = 'cuda' if device_name == 'cuda' and torch.cuda.is_available() else 'cpu'

        self.policy_dqn = None
        self.target_dqn = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.epsilon = self.epsilon_init

        # Logging ======================================================================================================
        if self.trial is None:
            self.log_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}.log')
            self.model_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}.pt')
            self.graph_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}.png')
        else:
            self.log_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}_trail_{self.trial.number}.log')
            self.model_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}_trail_{self.trial.number}.pt')
            self.graph_file_name: str = os.path.join(OUTPUT_DIR,
                                                     f'{self.config_set_name}_trail_{self.trial.number}.png')

        file_handler = logging.FileHandler(self.log_file_name, mode='w' if self.is_training else 'a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(file_handler)

    def run(self) -> float:
        step_counter: int = 0
        total_step_count: int = 0
        best_reward: float = -9999999.0
        last_graph_update_time = datetime.now()
        epsilon_history: List[float] = list()
        rewards_per_episode: List[float] = list()
        steps_per_episode: List[int] = list()
        time_per_episode: List[float] = list()

        if self.is_training:
            LOGGER.info(f'Training starting for environment {self.env_id}')

        if self.trial is not None:
            LOGGER.info(f'Training starting for trial {self.trial.number}')

        num_actions = self.env.action_space.n
        num_states = self.env.observation_space.shape[0]

        self.policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(self.device)

        if self.is_training:
            self.target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(self.device)
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)
        else:
            self.policy_dqn.load_state_dict(torch.load(self.model_file_name))
            self.policy_dqn.eval()

        for episode in range(self.episode_max_num):
            episode_start_time = datetime.now()

            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            terminated = False
            episode_reward = 0.0
            episode_step_count = 0

            while not terminated and episode_reward < self.episode_max_reward:
                action: Tensor = self.__select_action_epsilon_greedy(state)
                new_state, reward, terminated, truncated, info = self.env.step(action.item())

                episode_reward += reward
                episode_step_count += 1
                total_step_count += 1

                new_state = torch.tensor(new_state, dtype=torch.float, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)

                if self.is_training:
                    self.exp_buffer.append(Experience((state, action, new_state, reward, terminated)))
                    step_counter += 1

                    # If enough experience has been collected
                    if len(self.exp_buffer) > self.mini_batch_size:
                        mini_batch = self.exp_buffer.sample(self.mini_batch_size)
                        self.__optimize(mini_batch)

                        # Decay epsilon
                        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(self.epsilon)

                        # Copy policy network to target network after a certain number of steps
                        if step_counter > self.steps_to_sync_target_net:
                            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                            step_counter = 0

                state = new_state

            rewards_per_episode.append(episode_reward)
            steps_per_episode.append(episode_step_count)

            current_time = datetime.now()
            episode_duration = current_time - episode_start_time
            time_per_episode.append(episode_duration.total_seconds())

            if self.is_training:
                # Save model when new best reward is obtained.
                if episode_reward > best_reward:
                    reward_increase = (episode_reward - best_reward) / abs(best_reward) * 100
                    LOGGER.info(f'New best reward {episode_reward:0.1f} ({reward_increase:+.1f}%) ' +
                                f'at episode {episode}, saving model...')
                    torch.save(self.policy_dqn.state_dict(), self.model_file_name)
                    best_reward = episode_reward
                else:
                    LOGGER.info(f'Reward {episode_reward:0.1f} at episode {episode}')

                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.__save_graph(rewards_per_episode, epsilon_history, steps_per_episode, time_per_episode)
                    last_graph_update_time = current_time

        if self.trial is not None:
            LOGGER.info(
                f'Trial number: {self.trial.number}; Best reward: {best_reward}; Trial params: {self.trial.params}'
            )
        return best_reward  # for optuna objective

    def __select_action_epsilon_greedy(self, state: Tensor) -> Tensor:
        if self.is_training and random.random() < self.epsilon:
            action = self.env.action_space.sample()
            return torch.tensor(action, dtype=torch.int64, device=self.device)
        else:
            with torch.no_grad():
                return self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

    def __save_graph(self, rewards_per_episode, epsilon_history, steps_per_episode, time_per_episode):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        if self.trial is None:
            fig.suptitle(f'Training stats for environment {self.env_id}')
        else:
            fig.suptitle(f'Training stats for environment {self.env_id}; trial {self.trial.number}')

        # Rewards sum per episode - episode number
        axes[0, 0].plot(rewards_per_episode)
        axes[0, 0].set_title('Rewards sum per episode')
        axes[0, 0].set_xlabel('Episode number')
        axes[0, 0].set_ylabel('Rewards sum')

        # Rolling average of rewards sum per episode - episode number
        window_size: int = 100
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - window_size - 1):(x + 1)])

        axes[0, 1].plot(mean_rewards)
        axes[0, 1].set_title(f'Rolling average - rewards sum per episode (window={window_size})')
        axes[0, 1].set_xlabel('Episode number')
        axes[0, 1].set_ylabel('Average rewards sum')

        # Epsilon
        axes[1, 0].plot(epsilon_history)
        axes[1, 0].set_title('Epsilon change over combined steps')
        axes[1, 0].set_xlabel('Step number')
        axes[1, 0].set_ylabel('Epsilon')

        # Steps per episode
        axes[1, 1].plot(steps_per_episode)
        axes[1, 1].set_title('Steps per episode')
        axes[1, 1].set_xlabel('Episode number')
        axes[1, 1].set_ylabel('Steps sum')

        fig.tight_layout()
        fig.savefig(self.graph_file_name)
        plt.close(fig)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

        fig.tight_layout()
        fig.savefig(self.graph_file_name)
        plt.close(fig)

    def __optimize(self, mini_batch: List[Experience]) -> None:
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(self.device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = self.policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor * self.target_dqn(new_states) \
                    .gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor * \
                           self.target_dqn(new_states).max(dim=1)[0]

        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def objective(trial: optuna.Trial, args: Namespace):
    dql = Agent(config_set_name=args.config_set, device_name=args.device, is_training=args.train,
                do_render=False if args.train else True, trial=trial)
    return dql.run()


def optimize(args: Namespace) -> None:
    study: optuna.Study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args), n_trails=100)
    best_trial = study.best_trial

    best_params_file_name = f'{args.config_set}_best_params.json'
    with open(best_params_file_name, 'w') as file:
        json.dump(best_trial.params, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--config_set', type=str, help='Config set name')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--optimize', help='Optimization mode', action='store_true')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device name (cpu or cuda)')
    args = parser.parse_args()

    if args.optimize:
        optimize(args)
    else:
        dql = Agent(config_set_name=args.config_set, device_name=args.device, is_training=True if args.train else False,
                    do_render=False if args.train else True)
        dql.run()
