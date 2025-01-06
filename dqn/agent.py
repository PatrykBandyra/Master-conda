import argparse
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.types import Tensor

from dqn import DQN
from experience_buffer import ExperienceBuffer, Experience

DATE_FORMAT = '%m-%d %H:%M:%S'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(FORMATTER)
LOGGER.addHandler(stream_handler)

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib.use('Agg')  # Do not render plots on screen

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:
    def __init__(self, config_set_name: str, is_training: bool = True, do_render: bool = False):
        # Configuration ================================================================================================
        with open('config.yml', 'r') as file:
            all_config_sets = yaml.safe_load(file)
            config: Dict[str, Any] = all_config_sets[config_set_name]

        self.config_set_name: str = config_set_name

        self.env_id: str = config['env_id']
        self.learning_rate: float = config['learning_rate']  # alpha
        self.discount_factor: float = config['discount_factor']  # gamma
        self.steps_to_sync_target_net: int = config['steps_to_sync_target_net']
        self.exp_buffer_size: int = config['exp_buffer_size']
        self.mini_batch_size: int = config['mini_batch_size']
        self.epsilon_init: float = config['epsilon_init']
        self.epsilon_decay: float = config['epsilon_decay']
        self.epsilon_min: float = config['epsilon_min']
        self.episode_max_reward: float = config['episode_max_reward']
        self.episode_max_num: int = config['episode_max_num']
        self.fc1_nodes: int = config['fc1_nodes']
        self.env_make_params: Dict[Any, Any] = config.get('env_make_params', {})
        self.enable_double_dqn: bool = config['enable_double_dqn']
        self.enable_dueling_dqn: bool = config['enable_dueling_dqn']

        self.is_training = is_training

        # Environment ==================================================================================================
        self.env = gym.make(self.env_id, render_mode='human' if do_render else None, **self.env_make_params)
        self.exp_buffer = ExperienceBuffer(self.exp_buffer_size) if is_training else None

        # Neural Network ===============================================================================================
        self.policy_dqn = None
        self.target_dqn = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.epsilon = self.epsilon_init

        # Logging ======================================================================================================
        self.log_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}.log')
        self.model_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}.pt')
        self.graph_file_name: str = os.path.join(OUTPUT_DIR, f'{self.config_set_name}.png')

        file_handler = logging.FileHandler(self.log_file_name, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(file_handler)

    def run(self) -> None:
        step_count = 0
        best_reward = -9999999
        last_graph_update_time = datetime.now()
        epsilon_history: List[float] = []
        rewards_per_episode = []

        if self.is_training:
            LOGGER.info(f'Training starting for environment {self.env_id}')

        num_actions = self.env.action_space.n
        num_states = self.env.observation_space.shape[0]

        self.policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)

        if self.is_training:
            self.target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(DEVICE)
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)
        else:
            self.policy_dqn.load_state_dict(torch.load(self.model_file_name))
            self.policy_dqn.eval()

        for episode in range(self.episode_max_num):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)

            terminated = False
            episode_reward = 0.0

            # Gain experience
            while not terminated and episode_reward < self.episode_max_reward:
                action: Tensor = self.__select_action_epsilon_greedy(state)
                new_state, reward, terminated, truncated, info = self.env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=DEVICE)
                reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)

                if self.is_training:
                    self.exp_buffer.append(Experience((state, action, new_state, reward, terminated)))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if self.is_training:
                # Save model when new best reward is obtained.
                if episode_reward > best_reward:
                    LOGGER.info(f'New best reward {episode_reward:0.1f} ' +
                                f'({(episode_reward - best_reward) / best_reward * 100:+.1f}%) at episode {episode}, ' +
                                'saving model...')
                    torch.save(self.policy_dqn.state_dict(), self.model_file_name)
                    best_reward = episode_reward

                # Update graph
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(self.exp_buffer) > self.mini_batch_size:
                    mini_batch = self.exp_buffer.sample(self.mini_batch_size)
                    self.optimize(mini_batch)

                    # Decay epsilon
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(self.epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.steps_to_sync_target_net:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                        step_count = 0

    def __select_action_epsilon_greedy(self, state: Tensor) -> Tensor:
        if self.is_training and random.random() < self.epsilon:
            action = self.env.action_space.sample()
            return torch.tensor(action, dtype=torch.int64, device=DEVICE)
        else:
            with torch.no_grad():
                return self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.graph_file_name)
        plt.close(fig)

    def optimize(self, mini_batch: List[Experience]):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(DEVICE)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--config_set', type=str, help='Config set name')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(config_set_name=args.config_set, is_training=True if args.train else False,
                do_render=False if args.train else True)
    dql.run()
