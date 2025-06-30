# env/topology_loader.py

import networkx as nx
import numpy as np

def generate_topology(num_nodes=20, prob=0.3):
    # 生成随机图：节点更多，连接概率更高
    G = nx.erdos_renyi_graph(n=num_nodes, p=prob, seed=None, directed=False)

    # 添加边特征（带宽、延迟、抖动、丢包）
    edge_features = []
    for u, v in G.edges():
        bandwidth = np.random.uniform(10, 100)
        delay = np.random.uniform(1, 10)
        jitter = np.random.uniform(0, 2)
        loss = np.random.uniform(0, 0.1)
        G.edges[u, v]['features'] = [bandwidth, delay, jitter, loss]
        edge_features.append(G.edges[u, v]['features'])

    node_features = np.random.rand(len(G.nodes), 8)  # 假设每个节点8维特征

    return G, node_features, np.array(edge_features)