import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, is_dueling_dqn_enabled: bool = True):
        super(DQN, self).__init__()

        self.is_dueling_dqn_enabled = is_dueling_dqn_enabled

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.is_dueling_dqn_enabled:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.is_dueling_dqn_enabled:
            # Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calc Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.output(x)

        return Q
