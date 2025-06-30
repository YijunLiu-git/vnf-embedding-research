import torch
import torch.nn.functional as F
from torch import nn
import random
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from models.gnn_encoder import GNNEncoder
from utils.PPORolloutBuffer import PPORolloutBuffer as ReplayBuffer  

class PPOAgent:
    def __init__(self, state_dim, action_dim, edge_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.edge_dim = edge_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gnn_cfg = config["gnn"]
        train_cfg = config["train"]

        self.gamma = train_cfg["gamma"]
        self.eps_clip = train_cfg["eps_clip"]
        self.entropy_coef = train_cfg["entropy_coef"]
        self.batch_size = train_cfg["batch_size"]
        self.lr = train_cfg["lr"]

        self.policy_net = GNNEncoder(state_dim, edge_dim, gnn_cfg["hidden_dim"], gnn_cfg["output_dim"]).to(self.device)
        self.old_policy_net = GNNEncoder(state_dim, edge_dim, gnn_cfg["hidden_dim"], gnn_cfg["output_dim"]).to(self.device)
        self.output_layer = nn.Linear(gnn_cfg["output_dim"], action_dim).to(self.device)
        self.old_output_layer = nn.Linear(gnn_cfg["output_dim"], action_dim).to(self.device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.old_output_layer.load_state_dict(self.output_layer.state_dict())

        self.optimizer = torch.optim.Adam(list(self.policy_net.parameters()) + list(self.output_layer.parameters()), lr=self.lr)
        self.replay_buffer = ReplayBuffer(train_cfg["buffer_size"])
        self.episode = 0

    def select_action(self, state):
        self.old_policy_net.eval()
        self.old_output_layer.eval()
        with torch.no_grad():
            node_embeddings = self.old_policy_net(state.to(self.device))
            graph_embeddings = global_mean_pool(node_embeddings, state.batch)
            logits = self.old_output_layer(graph_embeddings)

            probs = F.softmax(logits, dim=-1)

            # Clamp to prevent NaN
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            if torch.isnan(probs).any():
                print("⚠️ WARNING: NaN detected in probs, fallback to uniform distribution.")
                probs = torch.ones_like(probs) / probs.size(-1)

            dist = Categorical(probs)
            action = dist.sample()

            return action.item(), dist.log_prob(action), dist.entropy()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.replay_buffer.add(state, action, reward, next_state, done, log_prob)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        samples = random.sample(self.replay_buffer.buffer, self.batch_size)
        states, actions, rewards, _, dones, log_probs_old = zip(*samples)

        states = Batch.from_data_list(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        log_probs_old = torch.stack(log_probs_old).detach().to(self.device)

        self.policy_net.train()
        self.output_layer.train()
        node_embeddings = self.policy_net(states)
        graph_embeddings = global_mean_pool(node_embeddings, states.batch)
        logits = self.output_layer(graph_embeddings)

        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = Categorical(probs)
        log_probs_new = dist.log_prob(actions)
        entropy = dist.entropy()

        ratios = torch.exp(log_probs_new - log_probs_old)
        advantages = rewards  # 可以替换成 GAE 等高级形式

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.old_output_layer.load_state_dict(self.output_layer.state_dict())

    def update_episode(self):
        self.episode += 1

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'output_layer': self.output_layer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.output_layer.load_state_dict(checkpoint['output_layer'])
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.old_output_layer.load_state_dict(self.output_layer.state_dict())