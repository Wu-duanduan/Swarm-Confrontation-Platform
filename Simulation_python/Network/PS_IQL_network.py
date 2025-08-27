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


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# DQN强化学习智能体
class DQNAgent:
    def __init__(self, params):
        # 共享参数配置
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
        self.memory = ReplayBuffer(params['buffer_size'])
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.tau = params['tau']
        self.action_dim = params['action_dim']
        self.current_episode = 0
        # self.warmup_episodes = params['warmup_episodes']

        self.save_dir = "../saved_models"
        os.makedirs(self.save_dir, exist_ok=True)

        # 记录最佳奖励
        self.best_reward = -float('inf')

    def save_model(self, episode=None, reward=None):
        """只保存最佳模型"""
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            save_path = os.path.join(self.save_dir, "best_dqn_ps_model.pth")

            torch.save({
                'q_net_state': self.q_net.state_dict(),
                'target_net_state': self.target_q_net.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'best_reward': self.best_reward,
                'episode': episode
            }, save_path)
            print(f"最佳模型已保存至 {save_path}，奖励: {reward:.2f}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.q_net.load_state_dict(checkpoint['q_net_state'])
        self.target_q_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        print(f"已从 {model_path} 加载模型，最佳奖励: {self.best_reward:.2f}")

    def get_action(self, obs, agent_id, eval_mode):
        """基于DQN的动作选择策略"""
        if eval_mode:  # 评估模式直接取最优动作
            return self._greedy_action(obs)

        # 训练模式下的探索策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return self._greedy_action(obs)

    def _greedy_action(self, obs):
        """获取网络预测的最优动作"""
        # 转换为张量
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # [1, obs_dim]

        with torch.no_grad():
            q_values = self.q_net(obs_tensor)

        return torch.argmax(q_values).item()

    def decay_epsilon(self):
        """基于训练阶段的ε衰减策略"""
        # 使用分段线性衰减策略
        if self.current_episode > 0:
            # 在40个episode内完成衰减
            if self.current_episode <= 13:  # 前13个episode快速衰减
                decay_rate = 0.8
            elif self.current_episode <= 27:  # 中间14个episode中速衰减
                decay_rate = 0.5
            else:  # 最后13个episode慢速衰减
                decay_rate = 0.2

            # 应用分段线性衰减
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon - decay_rate * (self.epsilon - self.epsilon_min) / 13
            )

    def update_target_network(self):
        # 软更新目标网络
        for target_param, param in zip(self.target_q_net.parameters(),
                                       self.q_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def store_experience(self, experiences):
        # experiences: list of (obs, action, reward, next_obs, done) for all agents
        for agent_id, (obs, action, reward, next_obs, done) in enumerate(experiences):
            self.memory.push(obs, action, reward, next_obs, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        # 随机采样批处理数据
        batch = self.memory.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        # 转换为张量
        obs_tensor = torch.FloatTensor(np.array(obs))
        actions_tensor = torch.LongTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        next_obs_tensor = torch.FloatTensor(np.array(next_obs))
        dones_tensor = torch.BoolTensor(np.array(dones))

        # 计算当前Q值
        current_q = self.q_net(obs_tensor).gather(1, actions_tensor.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            target_q = rewards_tensor + self.gamma * self.target_q_net(next_obs_tensor).max(1)[0] * (~dones_tensor)

        # 计算损失
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.update_target_network()

        return loss.item()


# 多智能体DQN强化学习
class MultiAgentDQN:
    def __init__(self, num_agents, params):
        self.num_agents = num_agents
        self.params = params
        self.agent = DQNAgent(self.params)  # 所有智能体共享同一个agent

    def get_actions(self, observations, eval_mode):
        # 并行获取所有智能体动作
        return [self.agent.get_action(obs, agent_id, eval_mode) for agent_id, obs in enumerate(observations)]

    def update_training_progress(self, episode):
        """更新训练阶段状态"""
        self.agent.current_episode = episode
        self.agent.decay_epsilon()

    def store_experiences(self, all_experiences):
        # 存储多智能体经验
        self.agent.store_experience(all_experiences)

    def train(self):
        # 执行训练步骤
        loss = self.agent.train_step()
        # 衰减探索率
        self.agent.epsilon = max(
            self.params['epsilon_min'],
            self.agent.epsilon * self.params['epsilon_decay']
        )
        return loss

    def save_training_progress(self, reward_list):
        # 使用 seaborn 的样式
        sns.set_theme()

        plt.figure(figsize=(10, 6))
        window_size = 10

        # 计算移动平均和标准差
        smooth_reward = np.convolve(reward_list, np.ones(window_size)/window_size, mode='valid')
        std_reward = [np.std(reward_list[i:i+window_size]) for i in range(len(reward_list)-window_size+1)]

        # 绘制原始曲线（半透明线）
        plt.plot(reward_list, color='#FF6F61', alpha=0.5, linewidth=1, label='Episode Reward')

        # 绘制平滑曲线及置信区间
        plt.plot(smooth_reward,
                color='#0F4C81',
                linewidth=2,
                label='Smoothed Reward (MA-10)')
        plt.fill_between(range(window_size-1, len(reward_list)),
                        (smooth_reward - std_reward),
                        (smooth_reward + std_reward),
                        color='#0F4C81',
                        alpha=0.2)

        # 美化坐标轴和标题
        plt.title("train_reward", fontsize=16, pad=20)
        plt.xlabel("Episodes", fontsize=14, labelpad=10)
        plt.ylabel("Return", fontsize=14, labelpad=10)

        # 网格和刻度调整
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 图例美化
        plt.legend(loc='lower right', fontsize=12, framealpha=0.9)

        # 保存设置
        save_path = os.path.join(self.agent.save_dir, "dqn_ps_training_progress.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"训练奖励曲线已保存至 {save_path}")