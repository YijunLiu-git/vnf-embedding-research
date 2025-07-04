import gym
import torch
import numpy as np
from gym import spaces
import networkx as nx
from torch_geometric.data import Data

class VNFEmbeddingEnv(gym.Env):
    def __init__(self, graph, node_features, edge_features):
        super(VNFEmbeddingEnv, self).__init__()
        self.graph = graph
        self.node_features = node_features
        self.edge_features = edge_features
        self.state_dim = node_features.shape[1]
        self.edge_dim = edge_features.shape[1]
        self.action_dim = len(graph.nodes)

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_dim)

        self.edge_map = list(self.graph.edges)
        self.reset()

    def reset(self):
        self.current_node = np.random.choice(list(self.graph.nodes))
        return self._get_state()

    def _get_state(self):
        x = torch.tensor(self.node_features, dtype=torch.float32)
        edge_index = torch.tensor(np.array(self.edge_map).T, dtype=torch.long)
        edge_attr = torch.tensor(self.edge_features, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _get_edge_attr(self, u, v):
        for idx, (src, dst) in enumerate(self.edge_map):
            if (u == src and v == dst) or (u == dst and v == src):
                return self.edge_features[idx]
        return np.zeros(self.edge_dim, dtype=np.float32)

    def step(self, action):
        done = True
        node_list = list(self.graph.nodes)

        if action >= len(node_list):
            print(f"[ERROR] Action {action} out of range! Max={len(node_list)-1}")
            return self._get_state(), -10.0, True, False

        target_node = node_list[action]

        try:
            path = nx.shortest_path(self.graph, source=self.current_node, target=target_node)
            delay = 0.0
            bandwidths = []
            jitters = []
            losses = []

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_attr = self._get_edge_attr(u, v)
                bandwidths.append(edge_attr[0])      # 带宽
                delay += edge_attr[1]                # 延迟累加
                jitters.append(edge_attr[2])         # 抖动
                losses.append(edge_attr[3])          # 丢包率

            hops = len(path) - 1
            min_bandwidth = min(bandwidths) if bandwidths else 0.0
            avg_jitter = float(np.mean(jitters)) if jitters else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            success = (target_node != self.current_node)

        except nx.NetworkXNoPath:
            delay = 100.0
            min_bandwidth = 0.0
            hops = 0
            avg_jitter = 1.0
            avg_loss = 1.0
            success = False

        from rewards.reward_v3_composite import composite_reward
        reward = composite_reward(success, delay, min_bandwidth, hops)

        self.current_node = target_node
        next_state = self._get_state()

        return next_state, reward, done, success
