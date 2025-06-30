import gym
import torch
import numpy as np
import networkx as nx
from gym import spaces
from torch_geometric.data import Data

from rewards.reward_v4_comprehensive_multi import compute_reward

class MultiVNFEmbeddingEnv(gym.Env):
    def __init__(self, graph, node_features, edge_features, reward_config, chain_length_range=(2, 5)):
        super(MultiVNFEmbeddingEnv, self).__init__()
        self.graph = graph
        self.node_features = node_features
        self.edge_features = edge_features
        self.reward_config = reward_config
        self.state_dim = node_features.shape[1]
        self.edge_dim = edge_features.shape[1]
        self.action_dim = len(graph.nodes)

        # 修复：观察空间应该包含更多信息
        # [node_features, current_vnf_requirements, remaining_vnfs, previous_embeddings]
        obs_dim = self.state_dim + 4  # +4 for VNF info
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_dim)

        self.edge_map = list(self.graph.edges)
        self.chain_length_range = chain_length_range

        # 状态记录
        self.embedding_map = {}
        self.service_chain = []
        self.current_vnf_index = 0  # 当前要嵌入的VNF索引
        self.vnf_requirements = []  # 每个VNF的资源需求

        self.reset()

    def reset(self):
        """重置环境，生成新的服务链"""
        chain_length = np.random.randint(*self.chain_length_range)
        self.service_chain = list(range(chain_length))  # VNF IDs: [0,1,2,...]
        
        # 为每个VNF生成资源需求
        self.vnf_requirements = []
        for _ in range(chain_length):
            requirements = {
                'cpu': np.random.uniform(0.1, 0.8),
                'memory': np.random.uniform(0.1, 0.8),
                'bandwidth': np.random.uniform(10, 50)
            }
            self.vnf_requirements.append(requirements)
        
        self.current_vnf_index = 0
        self.embedding_map.clear()
        
        return self._get_state()

    def _get_state(self):
        """获取当前状态 - 返回固定大小的向量"""
        if self.current_vnf_index >= len(self.service_chain):
            # 所有VNF已嵌入完成
            base_state = np.mean(self.node_features, axis=0)  # 平均节点特征
            vnf_info = np.zeros(4)
        else:
            # 当前节点资源状态 (简化：使用平均值)
            base_state = np.mean(self.node_features, axis=0)
            
            # 当前VNF信息
            current_vnf_req = self.vnf_requirements[self.current_vnf_index]
            vnf_info = np.array([
                current_vnf_req['cpu'],
                current_vnf_req['memory'], 
                current_vnf_req['bandwidth'],
                (len(self.service_chain) - self.current_vnf_index) / len(self.service_chain)  # 剩余比例
            ])
        
        state = np.concatenate([base_state, vnf_info])
        return state.astype(np.float32)

    def _get_edge_attr(self, u, v):
        """获取边属性"""
        for idx, (src, dst) in enumerate(self.edge_map):
            if (u == src and v == dst) or (u == dst and v == src):
                return self.edge_features[idx]
        return np.zeros(self.edge_dim, dtype=np.float32)

    def _check_resource_constraint(self, node_id, vnf_requirements):
        """检查节点资源是否满足VNF需求"""
        # 简化：假设节点特征的前3维是 [cpu, memory, available_bandwidth]
        if node_id >= len(self.node_features):
            return False
            
        node_resources = self.node_features[node_id]
        
        # 简单的资源检查（可以根据需要复杂化）
        if len(node_resources) >= 3:
            return (node_resources[0] >= vnf_requirements['cpu'] and 
                   node_resources[1] >= vnf_requirements['memory'])
        return True  # 如果没有资源信息，假设满足

    def step(self, action):
        """执行一个动作：为当前VNF选择嵌入节点"""
        # 检查动作有效性
        if action >= len(self.graph.nodes):
            reward = -10.0
            info = {"success": False, "error": "Invalid action"}
            return self._get_state(), reward, True, info

        # 检查是否所有VNF已嵌入完成
        if self.current_vnf_index >= len(self.service_chain):
            # 已完成，计算最终奖励
            reward, info = self._compute_final_reward()
            return self._get_state(), reward, True, info

        # 获取当前VNF和目标节点
        current_vnf = self.service_chain[self.current_vnf_index]
        target_node = action
        current_vnf_req = self.vnf_requirements[self.current_vnf_index]

        # 检查资源约束
        if not self._check_resource_constraint(target_node, current_vnf_req):
            # 资源不足，给予惩罚但不结束
            reward = -5.0
            info = {"success": False, "error": "Resource constraint violation"}
            return self._get_state(), reward, False, info

        # 检查节点是否已被占用
        if target_node in self.embedding_map.values():
            # 节点已被占用，给予惩罚
            reward = -3.0
            info = {"success": False, "error": "Node already occupied"}
            return self._get_state(), reward, False, info

        # 成功嵌入
        self.embedding_map[current_vnf] = target_node
        self.current_vnf_index += 1

        # 检查是否完成所有VNF嵌入
        done = (self.current_vnf_index >= len(self.service_chain))
        
        if done:
            # 计算最终奖励
            reward, info = self._compute_final_reward()
        else:
            # 中间奖励：成功嵌入一个VNF
            reward = 1.0
            info = {"success": True, "embedded_vnf": current_vnf, "at_node": target_node}

        next_state = self._get_state()
        return next_state, reward, done, info

    def _compute_final_reward(self):
        """计算完整服务链的最终奖励"""
        if len(self.embedding_map) != len(self.service_chain):
            # 未完全嵌入
            return -10.0, {"success": False, "sar": 0.0, "splat": float('inf')}

        # 计算服务链路径
        total_delay = 0.0
        total_bandwidth = float('inf')
        path_info = []
        
        for i in range(len(self.service_chain) - 1):
            vnf1 = self.service_chain[i]
            vnf2 = self.service_chain[i + 1]
            node1 = self.embedding_map[vnf1]
            node2 = self.embedding_map[vnf2]
            
            try:
                path = nx.shortest_path(self.graph, source=node1, target=node2)
                
                # 计算路径指标
                path_delay = 0.0
                path_bandwidths = []
                
                for j in range(len(path) - 1):
                    u, v = path[j], path[j + 1]
                    edge_attr = self._get_edge_attr(u, v)
                    if len(edge_attr) >= 2:
                        path_bandwidths.append(edge_attr[0])  # bandwidth
                        path_delay += edge_attr[1]  # delay
                
                min_bandwidth = min(path_bandwidths) if path_bandwidths else 0
                total_delay += path_delay
                total_bandwidth = min(total_bandwidth, min_bandwidth)
                
                path_info.append({
                    "delay": path_delay,
                    "bandwidth": min_bandwidth,
                    "hops": len(path) - 1,
                    "jitter": 0.1,  # 简化
                    "loss": 0.01   # 简化
                })
                
            except nx.NetworkXNoPath:
                # 无路径连接
                return -20.0, {"success": False, "error": "No path between VNFs"}

        # 计算SAR（服务接受率）
        sar = 1.0 if len(self.embedding_map) == len(self.service_chain) else 0.0
        
        # 计算SPLat（平均路径延迟）
        splat = total_delay / max(len(path_info), 1)
        
        # 使用原有奖励函数
        info = {
            "success": True,
            "paths": path_info,
            "sar": sar,
            "splat": splat,
            "embedding_map": self.embedding_map.copy()
        }
        
        reward = compute_reward(info, self.reward_config)
        
        # 添加SAR奖励
        if sar == 1.0:
            reward += 10.0  # 成功完成所有嵌入的奖励
        
        return reward, info

    def render(self, mode='human'):
        """可视化当前状态"""
        print(f"Service Chain: {self.service_chain}")
        print(f"Current VNF Index: {self.current_vnf_index}")
        print(f"Embedding Map: {self.embedding_map}")
        if self.current_vnf_index < len(self.vnf_requirements):
            print(f"Current VNF Requirements: {self.vnf_requirements[self.current_vnf_index]}")