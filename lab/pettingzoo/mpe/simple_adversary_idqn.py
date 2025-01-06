import json
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_adversary_v3


@dataclass
class Configuration:
    max_cycles: int
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    batch_size: int
    replay_buffer_size: int
    steps_until_target_network_update: int
    tau: float


class Idqn:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self) -> Configuration:
        with open('simple_adversary_idqn_config.json') as file:
            data = json.load(file)
        return Configuration(**data)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
replay_buffer_size = 10000
tau = 0.001  # Soft update parameter for target network
update_target_every = 10  # Update target network every 10 steps

# Initialize environment and agents
env = simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
num_agents = len(env.possible_agents)
observation_space = env.observation_space('adversary_0').shape[0]
action_space = env.action_space('adversary_0').n

# Initialize IDQN networks and optimizers for each agent
agents = {}
for agent_id in env.possible_agents:
    agents[agent_id] = {
        'model': IDQN(observation_space, action_space),
        'optimizer': optim.Adam(agents[agent_id]['model'].parameters(), lr=learning_rate),
        'memory': deque(maxlen=replay_buffer_size),
        'target_model': IDQN(observation_space, action_space)
    }
    agents[agent_id]['target_model'].load_state_dict(agents[agent_id]['model'].state_dict())


# Epsilon-greedy action selection
def choose_action(agent_id, state):
    if np.random.rand() <= epsilon:
        return env.action_space(agent_id).sample()
    else:
        state = torch.FloatTensor(state)
        q_values = agents[agent_id]['model'](state)
        return torch.argmax(q_values).item()


# Learn from experience replay
def learn(agent_id):
    if len(agents[agent_id]['memory']) < batch_size:
        return

    batch = random.sample(agents[agent_id]['memory'], batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.BoolTensor(dones)

    q_values = agents[agent_id]['model'](states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = agents[agent_id]['target_model'](next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (~dones)

    loss = nn.MSELoss()(q_values, target_q_values)

    agents[agent_id]['optimizer'].zero_grad()
    loss.backward()
    agents[agent_id]['optimizer'].step()


# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    env.reset()
    total_rewards = {agent_id: 0 for agent_id in env.possible_agents}

    for agent_id in env.agent_iter():
        observation, reward, done, truncation, info = env.last()
        total_rewards[agent_id] += reward

        if done:
            action = None
        else:
            action = choose_action(agent_id, observation)

        env.step(action)

        if done or truncation:
            next_state = np.zeros_like(observation)
        else:
            next_state = observation

        agents[agent_id]['memory'].append((observation, action, reward, next_state, done))
        learn(agent_id)

        # Update target network
        if episode % update_target_every == 0:
            for target_param, local_param in zip(agents[agent_id]['target_model'].parameters(),
                                                 agents[agent_id]['model'].parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episode {episode + 1}, Total Rewards: {total_rewards}")

env.close()
