# agents/multi_dqn_agent.py
import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from models.gnn_encoder import GNNEncoder
from utils.replay_buffer import ReplayBuffer

class MultiDQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, edge_dim, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = config["train"]["gamma"]
        self.batch_size = config["train"]["batch_size"]
        self.epsilon = config["train"]["epsilon_start"]
        self.epsilon_decay = config["train"]["epsilon_decay"]
        self.epsilon_min = config["train"]["epsilon_min"]

        hidden_dim = config["gnn"]["hidden_dim"]
        output_dim = config["gnn"]["output_dim"]

        self.encoder = GNNEncoder(state_dim, edge_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=config["train"]["lr"])
        self.replay_buffer = ReplayBuffer(capacity=config["train"]["buffer_size"])

        self.action_dim = action_dim

    def select_action(self, state):
        self.encoder.eval()
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.encoder(state).mean(dim=0)

        self.encoder.train()
        if torch.rand(1).item() < self.epsilon:
            return [torch.randint(0, self.action_dim, (1,)).item() for _ in range(3)]
        else:
            return q_values.topk(3).indices.tolist()

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
            next_q_values = self.encoder(next_state).mean(dim=0).detach()

            q = sum([q_values[a] for a in actions[i]]) / len(actions[i])
            max_q = next_q_values.max()
            target_q = rewards[i] + (1 - dones[i]) * self.gamma * max_q

            batch_q.append(q)
            batch_target_q.append(target_q)

        loss = F.mse_loss(torch.stack(batch_q), torch.stack(batch_target_q))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
