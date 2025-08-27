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


# 演员网络
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)  # 输出动作概率


# 评论家网络
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim):
        super().__init__()
        self.input_dim = state_dim + action_dim * num_agents
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, actions):
        # state: [batch_size, state_dim]
        # actions: [batch_size, num_agents * action_dim] (one-hot 编码)
        x = torch.cat([state, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # [batch_size]


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done, state, next_state, action_probs):
        self.buffer.append((obs, actions, reward, next_obs, done, state, next_state, action_probs))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# COMA 智能体
class COMAAgent:
    def __init__(self, params, agent_id):
        self.agent_id = agent_id
        self.actor = ActorNetwork(
            obs_dim=params['obs_dim'],
            action_dim=params['action_dim'],
            hidden_dim=params['hidden_dim']
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['lr'])
        self.action_dim = params['action_dim']
        self.current_episode = 0
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.save_dir = "../saved_models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_reward = -float('inf')

    def get_action(self, obs, eval_mode):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
        if eval_mode:
            return torch.argmax(probs, dim=-1).item()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim), probs
        return torch.multinomial(probs, 1).item(), probs

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

    def save_model(self, episode=None, reward=None):
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            save_path = os.path.join(self.save_dir, f"best_dqn_model_agent_coma_{self.agent_id}.pth")
            torch.save({
                'actor_state': self.actor.state_dict(),
                'optimizer_state': self.actor_optimizer.state_dict(),
                'epsilon': self.epsilon,
                'best_reward': self.best_reward,
                'episode': episode
            }, save_path)
            print(f"最佳模型已保存至 {save_path}，奖励: {reward:.2f}")

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.save_dir, f"best_dqn_model_agent_coma_{self.agent_id}.pth")
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        print(f"已从 {model_path} 加载模型，最佳奖励: {self.best_reward:.2f}")


# 多智能体 COMA
class MultiAgentCOMA:
    def __init__(self, num_agents, params):
        self.num_agents = num_agents
        self.params = params
        self.agents = [COMAAgent(params, agent_id=i) for i in range(num_agents)]
        self.memory = ReplayBuffer(params['buffer_size'])
        self.batch_size = params['batch_size']
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.critic = CriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            hidden_dim=params['hidden_dim']
        )
        self.target_critic = CriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            hidden_dim=params['hidden_dim']
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['lr'])
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.save_dir = "../saved_models"
        os.makedirs(self.save_dir, exist_ok=True)

    def get_actions(self, observations, eval_mode):
        actions = []
        action_probs = []
        for agent_id, obs in enumerate(observations):
            result = self.agents[agent_id].get_action(obs, eval_mode)
            if eval_mode:
                actions.append(result)
                action_probs.append(None)
            else:
                action, probs = result
                actions.append(action)
                action_probs.append(probs)
        return actions, action_probs

    def update_training_progress(self, episode):
        for agent in self.agents:
            agent.current_episode = episode
            agent.decay_epsilon()

    def store_experiences(self, experiences, state, next_state, action_probs):
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
            next_state,
            action_probs
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones, states, next_states, action_probs = zip(*batch)
        if len(obs[0]) != self.num_agents or len(actions[0]) != self.num_agents or len(next_obs[0]) != self.num_agents:
            print(f"Warning: Sampled batch has incorrect number of agents: obs={len(obs[0])}, actions={len(actions[0])}, next_obs={len(next_obs[0])}, expected={self.num_agents}")
            return None

        obs_tensors = [torch.FloatTensor(np.array(obs_i)) for obs_i in zip(*obs)]
        actions_tensors = [torch.LongTensor(np.array(act_i)) for act_i in zip(*actions)]
        next_obs_tensors = [torch.FloatTensor(np.array(next_obs_i)) for next_obs_i in zip(*next_obs)]
        state_tensors = torch.FloatTensor(np.array(states))
        next_state_tensors = torch.FloatTensor(np.array(next_states))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        dones_tensor = torch.BoolTensor(np.array(dones))
        action_probs_tensors = [torch.stack([probs[i] for probs in action_probs if probs is not None]) for i in range(self.num_agents)]

        # 转换为 one-hot 动作
        actions_one_hot = torch.zeros(self.batch_size, self.num_agents * self.action_dim)
        for i in range(self.num_agents):
            actions_one_hot[:, i * self.action_dim:(i + 1) * self.action_dim] = F.one_hot(actions_tensors[i], num_classes=self.action_dim).float()

        # 计算评论家 Q 值
        q_values = self.critic(state_tensors, actions_one_hot)  # [batch_size]

        # 计算目标 Q 值
        with torch.no_grad():
            next_action_probs = [self.agents[i].actor(next_obs_tensors[i]) for i in range(self.num_agents)]
            next_actions = [torch.multinomial(probs, 1).squeeze(-1) for probs in next_action_probs]
            next_actions_one_hot = torch.zeros(self.batch_size, self.num_agents * self.action_dim)
            for i in range(self.num_agents):
                next_actions_one_hot[:, i * self.action_dim:(i + 1) * self.action_dim] = F.one_hot(next_actions[i], num_classes=self.action_dim).float()
            target_q = self.target_critic(next_state_tensors, next_actions_one_hot)
            target_q = rewards_tensor + self.gamma * target_q * (~dones_tensor)

        # 优化评论家
        critic_loss = F.mse_loss(q_values, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新目标评论家
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # 优化演员
        actor_loss = 0
        for agent_id in range(self.num_agents):
            current_probs = self.agents[agent_id].actor(obs_tensors[agent_id])  # [batch_size, action_dim]
            log_probs = torch.log(current_probs + 1e-8)  # [batch_size, action_dim]

            # 计算反事实基线
            advantages = torch.zeros(self.batch_size, device=q_values.device)
            for b in range(self.batch_size):
                baseline = 0
                for a in range(self.action_dim):
                    temp_actions_one_hot = actions_one_hot[b].clone()
                    temp_actions_one_hot[agent_id * self.action_dim:(agent_id + 1) * self.action_dim] = F.one_hot(torch.tensor(a), num_classes=self.action_dim).float()
                    q_temp = self.critic(state_tensors[b:b+1], temp_actions_one_hot.unsqueeze(0))
                    baseline += current_probs[b, a] * q_temp
                q_actual = self.critic(state_tensors[b:b+1], actions_one_hot[b:b+1])
                advantages[b] = q_actual - baseline

            # 策略梯度损失
            action_indices = actions_tensors[agent_id]  # [batch_size]
            selected_log_probs = log_probs[range(self.batch_size), action_indices]  # [batch_size]
            actor_loss += -torch.mean(selected_log_probs * advantages.detach())

        # 优化演员
        for agent in self.agents:
            agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        for agent in self.agents:
            agent.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def save_critic(self, episode=None, reward=None):
        if reward is not None and reward > self.agents[0].best_reward:
            save_path = os.path.join(self.save_dir, "best_coma_critic.pth")
            torch.save({
                'critic_state': self.critic.state_dict(),
                'target_critic_state': self.target_critic.state_dict(),
                'critic_optimizer_state': self.critic_optimizer.state_dict(),
                'episode': episode
            }, save_path)
            print(f"最佳评论家模型已保存至 {save_path}，奖励: {reward:.2f}")

    def load_critic(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.save_dir, "best_coma_critic.pth")
        checkpoint = torch.load(model_path)
        self.critic.load_state_dict(checkpoint['critic_state'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
        print(f"已从 {model_path} 加载评论家模型")

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
        plt.title("train_reward_coma", fontsize=16, pad=20)
        plt.xlabel("Episodes", fontsize=14, labelpad=10)
        plt.ylabel("Return", fontsize=14, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
        save_path = os.path.join(self.save_dir, "coma_training_progress.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"训练奖励曲线已保存至 {save_path}")