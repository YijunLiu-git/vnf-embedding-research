import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from models.gnn_encoder import GNNEncoder
from utils.replay_buffer import ReplayBuffer

class MultiDDQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, edge_dim, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = config["train"]["gamma"]
        self.batch_size = config["train"]["batch_size"]
        self.epsilon = config["train"]["epsilon_start"]
        self.epsilon_decay = config["train"]["epsilon_decay"]
        self.epsilon_min = config["train"]["epsilon_min"]
        self.target_update = config["train"]["target_update"]

        hidden_dim = config["gnn"]["hidden_dim"]
        output_dim = config["gnn"]["output_dim"]

        self.encoder = GNNEncoder(state_dim, edge_dim, hidden_dim, output_dim).to(self.device)
        self.target_encoder = GNNEncoder(state_dim, edge_dim, hidden_dim, output_dim).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=config["train"]["lr"])
        self.replay_buffer = ReplayBuffer(capacity=config["train"]["buffer_size"])

        self.action_dim = action_dim
        self.steps = 0

    def select_action(self, state):
        self.encoder.eval()
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.encoder(state).mean(dim=0)

        self.encoder.train()
        if torch.rand(1).item() < self.epsilon:
            actions = torch.randint(0, self.action_dim, (3,)).tolist()
        else:
            topk_actions = q_values.topk(3).indices.tolist()
            # 加入动作范围限制
            actions = [max(0, min(a, self.action_dim - 1)) for a in topk_actions]
        return actions

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        batch_q = []
        batch_target_q = []

        for i in range(self.batch_size):
            state = states[i].to(self.device)
            next_state = next_states[i].to(self.device)

            q_values = self.encoder(state).mean(dim=0)
            next_q_values = self.target_encoder(next_state).mean(dim=0)

            # 安全地取动作对应 Q 值
            q = sum([q_values[max(0, min(a, self.action_dim - 1))] for a in actions[i]]) / max(len(actions[i]), 1)
            max_q = next_q_values.max().detach()

            target_q = rewards[i] + (1 - dones[i]) * self.gamma * max_q

            batch_q.append(q)
            batch_target_q.append(target_q)

        loss = F.mse_loss(torch.stack(batch_q), torch.stack(batch_target_q))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_encoder.load_state_dict(self.encoder.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)