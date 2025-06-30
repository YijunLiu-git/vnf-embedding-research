# agents/multi_ppo_agent.py

import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from models.gnn_encoder import GNNEncoder

class MultiPPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, edge_dim, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_dim = config["gnn"]["hidden_dim"]
        output_dim = config["gnn"]["output_dim"]
        self.encoder = GNNEncoder(state_dim, edge_dim, hidden_dim, output_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=config["train"]["lr"])
        self.clip_epsilon = config["train"]["eps_clip"]
        self.entropy_coef = config["train"]["entropy_coef"]

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.action_dim = action_dim

    def select_action(self, state):
        state = state.to(self.device)
        logits = self.encoder(state).mean(dim=0)

        # 保证概率值合法
        probs = torch.softmax(logits, dim=0)
        action_dist = torch.distributions.Categorical(probs)

        actions = set()
        while len(actions) < 3:
            a = action_dist.sample().item()
            if 0 <= a < self.action_dim:
                actions.add(a)

        actions = list(actions)
        self.states.append(state)
        self.actions.append(actions)
        return actions

    def store_transition(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self):
        if len(self.rewards) == 0:
            return

        states = self.states
        actions = self.actions
        rewards = self.rewards
        dones = self.dones

        G = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + 0.99 * R
            G.insert(0, R)

        G = torch.tensor(G, dtype=torch.float32).to(self.device)

        loss = 0
        for state, act_list, ret in zip(states, actions, G):
            logits = self.encoder(state).mean(dim=0)
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)

            log_probs = torch.stack([dist.log_prob(torch.tensor(a).to(self.device)) for a in act_list])
            entropy = dist.entropy()

            ratio = torch.exp(log_probs - log_probs.detach())
            surr1 = ratio * ret
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * ret
            loss += -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()