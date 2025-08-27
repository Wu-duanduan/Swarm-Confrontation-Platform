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


class MixingNetwork(nn.Module):
    def __init__(self, num_agents, hidden_dim):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, q_values, weights, biases):
        # q_values: [batch_size, num_agents]
        # weights[0]: [batch_size, num_agents, hidden_dim]
        # weights[1]: [batch_size, hidden_dim, 1]
        # biases[0]: [batch_size, hidden_dim]
        # biases[1]: [batch_size, 1]
        # 假设 q_values 为 0 表示死亡智能体
        mask = (q_values != 0).float()  # [batch_size, num_agents]
        q_values = q_values * mask  # 屏蔽死亡智能体的 Q 值
        x = F.elu(torch.bmm(q_values.unsqueeze(1), weights[0]).squeeze(1) + biases[0])  # [batch_size, hidden_dim]
        x = torch.bmm(x.unsqueeze(1), weights[1]).squeeze(1) + biases[1]  # [batch_size, 1]
        return x.squeeze(1)  # [batch_size]

class HyperNetwork(nn.Module):
    def __init__(self, state_dim, output_dim, embed_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, output_dim)
        self.output_dim = output_dim  # 添加用于调试

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return torch.abs(x)  # 确保单调性

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

    def push(self, obs, actions, reward, next_obs, done, state, next_state):
        self.buffer.append((obs, actions, reward, next_obs, done, state, next_state))

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
            save_path = os.path.join(self.save_dir, f"best_dqn_model_agent_qmix_{self.agent_id}.pth")
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
            model_path = os.path.join(self.save_dir, f"best_dqn_model_agent_qmix_{self.agent_id}.pth")
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
        self.state_dim = params['state_dim']
        self.mixing_network = MixingNetwork(num_agents, hidden_dim=32)
        self.hyper_w1 = HyperNetwork(self.state_dim, num_agents * 32)
        self.hyper_w2 = HyperNetwork(self.state_dim, 32)
        self.hyper_b1 = HyperNetwork(self.state_dim, 32)
        self.hyper_b2 = HyperNetwork(self.state_dim, 1)
        print(f"Hyper_w1 output_dim: {self.hyper_w1.output_dim}, expected: {num_agents * 32}")
        self.mixing_optimizer = optim.Adam(
            list(self.mixing_network.parameters()) +
            list(self.hyper_w1.parameters()) +
            list(self.hyper_w2.parameters()) +
            list(self.hyper_b1.parameters()) +
            list(self.hyper_b2.parameters()),
            lr=params['lr']
        )
        self.save_dir = "../saved_models"
        os.makedirs(self.save_dir, exist_ok=True)

    def get_actions(self, observations, eval_mode):
        return [self.agents[agent_id].get_action(obs, eval_mode)
                for agent_id, obs in enumerate(observations)]

    def update_training_progress(self, episode):
        for agent in self.agents:
            agent.current_episode = episode
            agent.decay_epsilon()

    def store_experiences(self, experiences, state, next_state):
        full_experiences = []
        alive_agent_ids = [i for i, exp in enumerate(experiences) if exp is not None]
        current_idx = 0
        for agent_id in range(self.num_agents):
            if current_idx < len(experiences) and agent_id == alive_agent_ids[current_idx]:
                full_experiences.append(experiences[current_idx])
                current_idx += 1
            else:
                placeholder_obs = np.zeros(self.params['obs_dim'])
                placeholder_action = 0
                placeholder_next_obs = np.zeros(self.params['obs_dim'])
                full_experiences.append(
                    (placeholder_obs, placeholder_action, experiences[0][2], placeholder_next_obs, experiences[0][4]))

        self.memory.push(
            [exp[0] for exp in full_experiences],
            [exp[1] for exp in full_experiences],
            full_experiences[0][2],
            [exp[3] for exp in full_experiences],
            full_experiences[0][4],
            state,
            next_state
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones, states, next_states = zip(*batch)
        if len(obs[0]) != self.num_agents or len(actions[0]) != self.num_agents or len(next_obs[0]) != self.num_agents:
            print(
                f"Warning: Sampled batch has incorrect number of agents: obs={len(obs[0])}, actions={len(actions[0])}, next_obs={len(next_obs[0])}, expected={self.num_agents}")
            return None

        obs_tensors = [torch.FloatTensor(np.array(obs_i)) for obs_i in zip(*obs)]
        actions_tensors = [torch.LongTensor(np.array(act_i)) for act_i in zip(*actions)]
        next_obs_tensors = [torch.FloatTensor(np.array(next_obs_i)) for next_obs_i in zip(*next_obs)]
        state_tensors = torch.FloatTensor(np.array(states))
        next_state_tensors = torch.FloatTensor(np.array(next_states))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        dones_tensor = torch.BoolTensor(np.array(dones))

        # 计算当前 Q 值
        q_values = []
        for agent_id, agent in enumerate(self.agents):
            q = agent.q_net(obs_tensors[agent_id]).gather(1, actions_tensors[agent_id].unsqueeze(1))
            q_values.append(q.squeeze(1))  # [batch_size]
        q_values = torch.stack(q_values, dim=1)  # [batch_size, num_agents]
        print(f"q_values shape: {q_values.shape}, expected: [{self.batch_size}, {self.num_agents}]")

        # 生成混合网络参数
        w1 = self.hyper_w1(state_tensors).view(self.batch_size, self.num_agents, 32)
        w2 = self.hyper_w2(state_tensors).view(self.batch_size, 32, 1)
        b1 = self.hyper_b1(state_tensors).view(self.batch_size, 32)
        b2 = self.hyper_b2(state_tensors).view(self.batch_size, 1)
        print(f"w1 shape before transpose: {w1.shape}, expected: [{self.batch_size}, {self.num_agents}, 32]")
        weights = [w1, w2]  # 移除转置，调整矩阵乘法顺序
        biases = [b1, b2]
        print(f"weights[0] shape: {weights[0].shape}, weights[1] shape: {weights[1].shape}")

        # 计算当前 Q_tot
        current_q_tot = self.mixing_network(q_values, weights, biases)  # [batch_size]
        print(f"current_q_tot shape: {current_q_tot.shape}, expected: [{self.batch_size}]")

        # 计算目标 Q 值
        target_q_values = []
        with torch.no_grad():
            for agent_id, agent in enumerate(self.agents):
                target_q = agent.target_q_net(next_obs_tensors[agent_id]).max(1)[0]
                target_q_values.append(target_q)  # [batch_size]
            target_q_values = torch.stack(target_q_values, dim=1)  # [batch_size, num_agents]

            # 生成目标混合网络参数
            target_w1 = self.hyper_w1(next_state_tensors).view(self.batch_size, self.num_agents, 32)
            target_w2 = self.hyper_w2(next_state_tensors).view(self.batch_size, 32, 1)
            target_b1 = self.hyper_b1(next_state_tensors).view(self.batch_size, 32)
            target_b2 = self.hyper_b2(next_state_tensors).view(self.batch_size, 1)
            target_weights = [target_w1, target_w2]
            target_biases = [target_b1, target_b2]

            # 计算目标 Q_tot
            target_q_tot = self.mixing_network(target_q_values, target_weights, target_biases)
            target_q_tot = rewards_tensor + self.params['gamma'] * target_q_tot * (~dones_tensor)

        # 计算损失
        loss = F.mse_loss(current_q_tot, target_q_tot)

        # 优化 Q 网络和混合网络
        for agent in self.agents:
            agent.optimizer.zero_grad()
        self.mixing_optimizer.zero_grad()
        loss.backward()
        for agent in self.agents:
            agent.optimizer.step()
            agent.update_target_network()
        self.mixing_optimizer.step()

        return loss.item()

    def save_mixing_network(self, episode=None, reward=None):
        if reward is not None and reward > self.agents[0].best_reward:
            save_path = os.path.join(self.save_dir, "best_qmix_mixing_network.pth")
            torch.save({
                'mixing_network_state': self.mixing_network.state_dict(),
                'hyper_w1_state': self.hyper_w1.state_dict(),
                'hyper_w2_state': self.hyper_w2.state_dict(),
                'hyper_b1_state': self.hyper_b1.state_dict(),
                'hyper_b2_state': self.hyper_b2.state_dict(),
                'mixing_optimizer_state': self.mixing_optimizer.state_dict(),
                'episode': episode
            }, save_path)
            print(f"最佳混合网络已保存至 {save_path}，奖励: {reward:.2f}")

    def load_mixing_network(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.save_dir, "best_qmix_mixing_network.pth")
        checkpoint = torch.load(model_path)
        self.mixing_network.load_state_dict(checkpoint['mixing_network_state'])
        self.hyper_w1.load_state_dict(checkpoint['hyper_w1_state'])
        self.hyper_w2.load_state_dict(checkpoint['hyper_w2_state'])
        self.hyper_b1.load_state_dict(checkpoint['hyper_b1_state'])
        self.hyper_b2.load_state_dict(checkpoint['hyper_b2_state'])
        self.mixing_optimizer.load_state_dict(checkpoint['mixing_optimizer_state'])
        print(f"已从 {model_path} 加载混合网络")

    def save_training_progress(self, reward_list):
        sns.set_theme()
        plt.figure(figsize=(10, 6))
        window_size = 10
        smooth_reward = np.convolve(reward_list, np.ones(window_size) / window_size, mode='valid')
        std_reward = [np.std(reward_list[i:i + window_size]) for i in range(len(reward_list) - window_size + 1)]
        plt.plot(reward_list, color='#FF6F61', alpha=0.5, linewidth=1, label='Episode Reward')
        plt.plot(smooth_reward, color='#0F4C81', linewidth=2, label='Smoothed Reward (MA-10)')
        plt.fill_between(range(window_size - 1, len(reward_list)),
                         (smooth_reward - std_reward),
                         (smooth_reward + std_reward),
                         color='#0F4C81', alpha=0.2)
        plt.title("train_reward_qmix", fontsize=16, pad=20)
        plt.xlabel("Episodes", fontsize=14, labelpad=10)
        plt.ylabel("Return", fontsize=14, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
        save_path = os.path.join(self.save_dir, "qmix_training_progress.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"训练奖励曲线已保存至 {save_path}")