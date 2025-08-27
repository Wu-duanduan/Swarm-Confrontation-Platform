import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math


# DQN网络
class DQNNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# 经验回放缓冲区（修改以存储联合经验）
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done):
        # 存储联合观察、联合动作、全局奖励、下一联合观察和完成标志
        self.buffer.append((obs, actions, reward, next_obs, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# DQN智能体（为每个智能体独立维护）
class DQNAgent:
    def __init__(self, params, agent_id):
        self.agent_id = agent_id  # 添加智能体ID
        self.q_net = DQNNetwork(
            obs_dim=params['obs_dim'],
            action_dim=params['action_dim'],
            hidden_dim=params['hidden_dim']
        )
        self.target_q_net = DQNNetwork(
            obs_dim=params['obs_dim'],
            action_dim=params['action_dim'],
            hidden_dim=params['hidden_dim']
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=params['lr'])
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.tau = params['tau']
        self.action_dim = params['action_dim']
        self.current_episode = 0

        self.save_dir = "../saved_models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_reward = -float('inf')

    def save_model(self, episode=None, reward=None):
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            save_path = os.path.join(self.save_dir, f"best_dqn_model_agent_vdn_{self.agent_id}.pth")
            torch.save({
                'q_net_state': self.q_net.state_dict(),
                'target_net_state': self.target_q_net.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'best_reward': self.best_reward,
                'episode': episode
            }, save_path)
            print(f"最佳模型已保存至 {save_path}，奖励: {reward:.2f}")

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.save_dir, f"best_dqn_model_agent_vdn_{self.agent_id}.pth")
        checkpoint = torch.load(model_path)
        self.q_net.load_state_dict(checkpoint['q_net_state'])
        self.target_q_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        print(f"已从 {model_path} 加载模型，最佳奖励: {self.best_reward:.2f}")

    def get_action(self, obs, eval_mode):
        if eval_mode:
            return self._greedy_action(obs)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return self._greedy_action(obs)

    def _greedy_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return torch.argmax(q_values).item()

    def decay_epsilon(self):
        if self.current_episode > 0:
            if self.current_episode <= 13:
                decay_rate = 0.8
            elif self.current_episode <= 27:
                decay_rate = 0.5
            else:
                decay_rate = 0.2
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon - decay_rate * (self.epsilon - self.epsilon_min) / 13
            )

    def update_target_network(self):
        for target_param, param in zip(self.target_q_net.parameters(),
                                      self.q_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )


# 多智能体VDN
class MultiAgentDQN:
    def __init__(self, num_agents, params):
        self.num_agents = num_agents
        self.params = params
        self.agents = [DQNAgent(params, agent_id=i) for i in range(num_agents)]
        self.memory = ReplayBuffer(params['buffer_size'])
        self.batch_size = params['batch_size']

    def get_actions(self, observations, eval_mode):
        return [self.agents[agent_id].get_action(obs, eval_mode)
                for agent_id, obs in enumerate(observations)]

    def update_training_progress(self, episode):
        for agent in self.agents:
            agent.current_episode = episode
            agent.decay_epsilon()

    def store_experiences(self, experiences):
        # experiences: list of (obs, actions, global_reward, next_obs, done)
        # Ensure experiences include all agents, even dead ones
        full_experiences = []
        alive_agent_ids = [i for i, exp in enumerate(experiences) if exp is not None]
        current_idx = 0
        for agent_id in range(self.num_agents):
            if current_idx < len(experiences) and agent_id == alive_agent_ids[current_idx]:
                # Use actual experience for alive agent
                full_experiences.append(experiences[current_idx])
                current_idx += 1
            else:
                # Placeholder for dead agent
                placeholder_obs = np.zeros(self.params['obs_dim'])  # Zero observation
                placeholder_action = 0  # Placeholder action
                placeholder_next_obs = np.zeros(self.params['obs_dim'])  # Zero next observation
                full_experiences.append(
                    (placeholder_obs, placeholder_action, experiences[0][2], placeholder_next_obs, experiences[0][4]))

        print(f"Storing experiences for {len(full_experiences)} agents, expected {self.num_agents} agents")
        self.memory.push(
            [exp[0] for exp in full_experiences],
            [exp[1] for exp in full_experiences],
            full_experiences[0][2],  # Global reward
            [exp[3] for exp in full_experiences],
            full_experiences[0][4]  # Global done
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        if len(obs[0]) != self.num_agents or len(actions[0]) != self.num_agents or len(next_obs[0]) != self.num_agents:
            print(
                f"Warning: Sampled batch has incorrect number of agents: obs={len(obs[0])}, actions={len(actions[0])}, next_obs={len(next_obs[0])}, expected={self.num_agents}")
            return None

        obs_tensors = [torch.FloatTensor(np.array(obs_i)) for obs_i in zip(*obs)]
        actions_tensors = [torch.LongTensor(np.array(act_i)) for act_i in zip(*actions)]
        next_obs_tensors = [torch.FloatTensor(np.array(next_obs_i)) for next_obs_i in zip(*next_obs)]  # Add this line
        if len(obs_tensors) != self.num_agents or len(actions_tensors) != self.num_agents or len(
                next_obs_tensors) != self.num_agents:
            print(
                f"Error: obs_tensors length={len(obs_tensors)}, actions_tensors length={len(actions_tensors)}, next_obs_tensors length={len(next_obs_tensors)}, expected={self.num_agents}")
            return None

        rewards_tensor = torch.FloatTensor(np.array(rewards))
        dones_tensor = torch.BoolTensor(np.array(dones))

        current_q_tot = 0
        for agent_id, agent in enumerate(self.agents):
            current_q = agent.q_net(obs_tensors[agent_id]).gather(1, actions_tensors[agent_id].unsqueeze(1))
            current_q_tot += current_q.squeeze()

        target_q_tot = 0
        with torch.no_grad():
            for agent_id, agent in enumerate(self.agents):
                target_q = agent.target_q_net(next_obs_tensors[agent_id]).max(1)[0]
                target_q_tot += target_q
            target_q_tot = rewards_tensor + self.params['gamma'] * target_q_tot * (~dones_tensor)

        loss = F.mse_loss(current_q_tot, target_q_tot)

        for agent in self.agents:
            agent.optimizer.zero_grad()
        loss.backward()
        for agent in self.agents:
            agent.optimizer.step()
            agent.update_target_network()

        return loss.item()

    def save_training_progress(self, reward_list):
        sns.set_theme()
        plt.figure(figsize=(10, 6))
        window_size = 10
        smooth_reward = np.convolve(reward_list, np.ones(window_size)/window_size, mode='valid')
        std_reward = [np.std(reward_list[i:i+window_size]) for i in range(len(reward_list)-window_size+1)]
        plt.plot(reward_list, color='#FF6F61', alpha=0.5, linewidth=1, label='Episode Reward')
        plt.plot(smooth_reward, color='#0F4C81', linewidth=2, label='Smoothed Reward (MA-10)')
        plt.fill_between(range(window_size-1, len(reward_list)),
                        (smooth_reward - std_reward),
                        (smooth_reward + std_reward),
                        color='#0F4C81', alpha=0.2)
        plt.title("train_reward_vdn", fontsize=16, pad=20)
        plt.xlabel("Episodes", fontsize=14, labelpad=10)
        plt.ylabel("Return", fontsize=14, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
        save_path = os.path.join(self.agents[0].save_dir, "vdn_training_progress.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"训练奖励曲线已保存至 {save_path}")