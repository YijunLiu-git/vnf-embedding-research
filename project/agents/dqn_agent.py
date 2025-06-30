import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.nn import global_mean_pool
from models.gnn_encoder import GNNEncoder
from utils.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, edge_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.edge_dim = edge_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.MSELoss()

        train_cfg = config["train"]
        gnn_cfg = config["gnn"]

        self.gamma = train_cfg["gamma"]
        self.epsilon = train_cfg["epsilon_start"]
        self.epsilon_decay = train_cfg["epsilon_decay"]
        self.epsilon_min = train_cfg["epsilon_min"]
        self.batch_size = train_cfg["batch_size"]
        self.target_update = train_cfg["target_update"]

        self.q_net = GNNEncoder(state_dim, edge_dim, gnn_cfg["hidden_dim"], gnn_cfg["output_dim"]).to(self.device)
        self.target_net = GNNEncoder(state_dim, edge_dim, gnn_cfg["hidden_dim"], gnn_cfg["output_dim"]).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=train_cfg["lr"])
        self.replay_buffer = ReplayBuffer(train_cfg["buffer_size"])
        self.episode = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            self.q_net.eval()
            with torch.no_grad():
                out = self.q_net(state.to(self.device))
                return out.argmax(dim=1)[0].item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从buffer采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 迁移到device
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)  # [batch_size, 1]
        dones = dones.to(self.device).unsqueeze(1)      # [batch_size, 1]

        # 当前Q网络预测值
        q_values_all = self.q_net(states)
        graph_ids = states.batch
        q_values = global_mean_pool(q_values_all, graph_ids)  # [batch_size, action_dim]
        q_values = q_values.gather(1, actions.unsqueeze(1))   # [batch_size, 1]

        # 目标Q网络预测值
        with torch.no_grad():
            next_q_values_all = self.target_net(next_states)
            next_graph_ids = next_states.batch
            next_q_values = global_mean_pool(next_q_values_all, next_graph_ids)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]  # [batch_size, 1]

        # TD target
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 计算loss
        loss = self.loss_fn(q_values, target_q_values)

        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_episode(self):
        self.episode += 1

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())