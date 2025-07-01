# env/vnf_env_multi.py

import gym
import torch
import numpy as np
import networkx as nx
from gym import spaces
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Union, Any
import random

from rewards.reward_v4_comprehensive_multi import compute_reward

class MultiVNFEmbeddingEnv(gym.Env):
    """
    多VNF嵌入环境 - 修复版本
    
    核心功能：
    1. 序贯VNF嵌入决策（每次嵌入一个VNF）
    2. 图状态表示（支持边缘感知特征）
    3. 资源约束检查
    4. 正确的奖励计算和episode管理
    """
    
    def __init__(self, graph, node_features, edge_features, reward_config, chain_length_range=(2, 5)):
        super(MultiVNFEmbeddingEnv, self).__init__()
        
        # 网络拓扑和特征
        self.graph = graph
        self.node_features = node_features  # [num_nodes, node_feature_dim]
        self.edge_features = edge_features  # [num_edges, edge_feature_dim] 
        self.reward_config = reward_config
        
        # 环境配置
        self.chain_length_range = chain_length_range
        self.max_episode_steps = 20  # 防止无限循环
        
        # 特征维度
        self.state_dim = node_features.shape[1] if len(node_features.shape) > 1 else node_features.shape[0]
        self.edge_dim = edge_features.shape[1] if len(edge_features.shape) > 1 else edge_features.shape[0]
        self.action_dim = len(graph.nodes())
        
        # 动作和观察空间
        # 注意：这里定义为图数据空间，智能体会用GNN处理
        self.action_space = spaces.Discrete(self.action_dim)
        
        # 观察空间：为了兼容gym，定义一个大致的Box空间
        # 实际返回的是PyTorch Geometric Data对象
        max_nodes = len(graph.nodes())
        max_features = self.state_dim + 10  # 额外的VNF上下文信息
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(max_nodes * max_features,), 
            dtype=np.float32
        )
        
        # 边映射（用于快速查找边特征）
        self.edge_map = list(self.graph.edges())
        self.edge_index_map = {edge: idx for idx, edge in enumerate(self.edge_map)}
        
        # VNF嵌入状态
        self.service_chain = []           # 当前服务链的VNF列表
        self.vnf_requirements = []        # 每个VNF的资源需求
        self.current_vnf_index = 0        # 当前要嵌入的VNF索引
        self.embedding_map = {}           # VNF -> 节点的映射
        self.used_nodes = set()           # 已使用的节点
        self.step_count = 0               # 当前步数
        
        # 初始节点资源（复制原始资源）
        self.initial_node_resources = node_features.copy()
        self.current_node_resources = node_features.copy()
        
        print(f"🌍 VNF嵌入环境初始化完成:")
        print(f"   - 网络节点数: {len(graph.nodes())}")
        print(f"   - 网络边数: {len(graph.edges())}")
        print(f"   - 节点特征维度: {self.state_dim}")
        print(f"   - 边特征维度: {self.edge_dim}")
        
        self.reset()
    
    def reset(self) -> Data:
        """
        重置环境，生成新的服务链嵌入任务
        
        Returns:
            initial_state: 初始图状态
        """
        # 生成新的服务链
        chain_length = np.random.randint(*self.chain_length_range)
        self.service_chain = [f"VNF_{i}" for i in range(chain_length)]
        
        # 为每个VNF生成资源需求
        self.vnf_requirements = []
        for i in range(chain_length):
            # 生成CPU、内存、带宽需求
            cpu_req = np.random.uniform(0.1, 0.6)    # 10%-60% CPU
            memory_req = np.random.uniform(0.1, 0.5) # 10%-50% Memory  
            bandwidth_req = np.random.uniform(5, 25) # 5-25 Mbps
            
            self.vnf_requirements.append({
                'cpu': cpu_req,
                'memory': memory_req,
                'bandwidth': bandwidth_req,
                'vnf_type': i % 3  # 0: Firewall, 1: LoadBalancer, 2: Cache
            })
        
        # 重置嵌入状态
        self.current_vnf_index = 0
        self.embedding_map.clear()
        self.used_nodes.clear()
        self.step_count = 0
        
        # 重置节点资源
        self.current_node_resources = self.initial_node_resources.copy()
        
        print(f"\n🔄 新的嵌入任务:")
        print(f"   - 服务链长度: {len(self.service_chain)}")
        print(f"   - VNF需求: {[f'CPU:{req['cpu']:.2f}' for req in self.vnf_requirements]}")
        
        return self._get_state()
    
    def _get_state(self) -> Data:
        """
        获取当前图状态
        
        Returns:
            state: PyTorch Geometric Data对象，包含：
                - x: 增强的节点特征（原始特征 + 当前状态信息）
                - edge_index: 边索引
                - edge_attr: 边特征
                - vnf_context: 当前VNF需求信息
        """
        
        # 基础节点特征
        enhanced_node_features = self.current_node_resources.copy()
        
        # 添加节点状态信息
        num_nodes = len(self.graph.nodes())
        node_status = np.zeros((num_nodes, 4))  # [is_used, cpu_utilization, memory_utilization, vnf_count]
        
        for node_id in range(num_nodes):
            # 节点使用状态
            node_status[node_id, 0] = 1.0 if node_id in self.used_nodes else 0.0
            
            # 资源利用率（原始资源 - 当前可用资源）
            if self.initial_node_resources[node_id, 0] > 0:  # 避免除零
                cpu_util = 1.0 - (self.current_node_resources[node_id, 0] / self.initial_node_resources[node_id, 0])
                node_status[node_id, 1] = max(0.0, min(1.0, cpu_util))
            
            if len(self.initial_node_resources[node_id]) > 1 and self.initial_node_resources[node_id, 1] > 0:
                memory_util = 1.0 - (self.current_node_resources[node_id, 1] / self.initial_node_resources[node_id, 1])
                node_status[node_id, 2] = max(0.0, min(1.0, memory_util))
            
            # 节点上的VNF数量
            vnf_count = sum(1 for vnf, node in self.embedding_map.items() if node == node_id)
            node_status[node_id, 3] = vnf_count / 5.0  # 归一化（假设最多5个VNF）
        
        # 合并节点特征
        if len(enhanced_node_features.shape) == 1:
            enhanced_node_features = enhanced_node_features.reshape(-1, 1)
        enhanced_node_features = np.hstack([enhanced_node_features, node_status])
        
        # 构建图数据
        x = torch.tensor(enhanced_node_features, dtype=torch.float32)
        
        # 边索引和特征
        edge_index = torch.tensor(np.array(self.edge_map).T, dtype=torch.long)
        edge_attr = torch.tensor(self.edge_features, dtype=torch.float32)
        
        # VNF上下文信息
        if self.current_vnf_index < len(self.vnf_requirements):
            current_vnf_req = self.vnf_requirements[self.current_vnf_index]
            vnf_context = torch.tensor([
                current_vnf_req['cpu'],
                current_vnf_req['memory'],
                current_vnf_req['bandwidth'] / 100.0,  # 归一化
                current_vnf_req['vnf_type'] / 3.0,     # 归一化
                self.current_vnf_index / len(self.service_chain),  # 进度
                (len(self.service_chain) - self.current_vnf_index) / len(self.service_chain)  # 剩余比例
            ], dtype=torch.float32)
        else:
            vnf_context = torch.zeros(6, dtype=torch.float32)
        
        # 创建PyTorch Geometric Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            vnf_context=vnf_context
        )
        
        return data
    
    def step(self, action: int) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """
        执行一个动作：为当前VNF选择嵌入节点
        
        Args:
            action: 选择的节点ID
            
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.step_count += 1
        
        # 检查动作有效性
        if action >= self.action_dim:
            return self._handle_invalid_action(f"动作超出范围: {action} >= {self.action_dim}")
        
        # 检查是否所有VNF已嵌入完成
        if self.current_vnf_index >= len(self.service_chain):
            return self._handle_completion()
        
        # 获取当前VNF信息
        current_vnf = self.service_chain[self.current_vnf_index]
        current_vnf_req = self.vnf_requirements[self.current_vnf_index]
        target_node = action
        
        # 检查节点约束
        constraint_check = self._check_embedding_constraints(target_node, current_vnf_req)
        
        if not constraint_check['valid']:
            # 约束违反，给予惩罚但不结束episode
            reward = self._calculate_constraint_penalty(constraint_check['reason'])
            next_state = self._get_state()
            
            info = {
                'success': False,
                'constraint_violation': constraint_check['reason'],
                'current_vnf': current_vnf,
                'target_node': target_node,
                'step': self.step_count
            }
            
            return next_state, reward, False, info
        
        # 成功嵌入VNF
        self.embedding_map[current_vnf] = target_node
        self.used_nodes.add(target_node)
        
        # 更新节点资源
        self._update_node_resources(target_node, current_vnf_req)
        
        # 移动到下一个VNF
        self.current_vnf_index += 1
        
        # 检查是否完成所有VNF嵌入
        done = (self.current_vnf_index >= len(self.service_chain)) or (self.step_count >= self.max_episode_steps)
        
        if done and self.current_vnf_index >= len(self.service_chain):
            # 成功完成所有VNF嵌入
            reward, info = self._calculate_final_reward()
            info.update({
                'success': True,
                'embedding_completed': True,
                'total_steps': self.step_count
            })
        else:
            # 中间步骤奖励
            reward = self._calculate_intermediate_reward(current_vnf, target_node)
            info = {
                'success': True,
                'embedded_vnf': current_vnf,
                'target_node': target_node,
                'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
                'step': self.step_count
            }
        
        next_state = self._get_state()
        return next_state, reward, done, info
    
    def _check_embedding_constraints(self, node_id: int, vnf_req: Dict) -> Dict[str, Any]:
        """
        检查VNF嵌入约束
        
        Args:
            node_id: 目标节点ID
            vnf_req: VNF资源需求
            
        Returns:
            constraint_result: 约束检查结果
        """
        
        # 检查节点是否已被使用（如果不允许共享）
        if node_id in self.used_nodes:
            return {
                'valid': False,
                'reason': 'node_occupied',
                'details': f'节点 {node_id} 已被其他VNF使用'
            }
        
        # 检查CPU资源
        if self.current_node_resources[node_id, 0] < vnf_req['cpu']:
            return {
                'valid': False,
                'reason': 'insufficient_cpu',
                'details': f'节点 {node_id} CPU不足: 需要{vnf_req["cpu"]:.2f}, 可用{self.current_node_resources[node_id, 0]:.2f}'
            }
        
        # 检查内存资源（如果节点特征包含内存）
        if (len(self.current_node_resources[node_id]) > 1 and 
            self.current_node_resources[node_id, 1] < vnf_req['memory']):
            return {
                'valid': False,
                'reason': 'insufficient_memory',
                'details': f'节点 {node_id} 内存不足: 需要{vnf_req["memory"]:.2f}, 可用{self.current_node_resources[node_id, 1]:.2f}'
            }
        
        # 检查网络连通性（与已嵌入的VNF）
        connectivity_check = self._check_network_connectivity(node_id, vnf_req)
        if not connectivity_check['valid']:
            return connectivity_check
        
        return {
            'valid': True,
            'reason': 'all_constraints_satisfied'
        }
    
    def _check_network_connectivity(self, node_id: int, vnf_req: Dict) -> Dict[str, Any]:
        """检查网络连通性约束"""
        
        # 如果是第一个VNF，无需检查连通性
        if self.current_vnf_index == 0:
            return {'valid': True, 'reason': 'first_vnf'}
        
        # 检查与前一个VNF的连通性
        prev_vnf = self.service_chain[self.current_vnf_index - 1]
        prev_node = self.embedding_map.get(prev_vnf)
        
        if prev_node is None:
            return {'valid': True, 'reason': 'no_previous_embedding'}
        
        # 检查是否存在路径
        try:
            path = nx.shortest_path(self.graph, source=prev_node, target=node_id)
            
            # 检查路径带宽约束
            min_bandwidth = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_bandwidth = self._get_edge_bandwidth(u, v)
                min_bandwidth = min(min_bandwidth, edge_bandwidth)
            
            # 检查带宽是否满足需求
            if min_bandwidth < vnf_req['bandwidth']:
                return {
                    'valid': False,
                    'reason': 'insufficient_bandwidth',
                    'details': f'路径带宽不足: 需要{vnf_req["bandwidth"]:.1f}, 可用{min_bandwidth:.1f}'
                }
            
            return {'valid': True, 'reason': 'connectivity_satisfied'}
            
        except nx.NetworkXNoPath:
            return {
                'valid': False,
                'reason': 'no_network_path',
                'details': f'节点 {prev_node} 到 {node_id} 无连通路径'
            }
    
    def _get_edge_bandwidth(self, u: int, v: int) -> float:
        """获取边的可用带宽"""
        # 查找边特征
        if (u, v) in self.edge_index_map:
            edge_idx = self.edge_index_map[(u, v)]
        elif (v, u) in self.edge_index_map:
            edge_idx = self.edge_index_map[(v, u)]
        else:
            return 0.0  # 边不存在
        
        # 假设边特征的第一维是带宽
        return self.edge_features[edge_idx, 0] if len(self.edge_features[edge_idx]) > 0 else 100.0
    
    def _update_node_resources(self, node_id: int, vnf_req: Dict):
        """更新节点资源（扣除VNF消耗）"""
        self.current_node_resources[node_id, 0] -= vnf_req['cpu']
        
        if len(self.current_node_resources[node_id]) > 1:
            self.current_node_resources[node_id, 1] -= vnf_req['memory']
        
        # 确保资源不为负
        self.current_node_resources[node_id] = np.maximum(
            self.current_node_resources[node_id], 0.0
        )
    
    def _calculate_constraint_penalty(self, reason: str) -> float:
        """计算约束违反的惩罚"""
        penalty_map = {
            'node_occupied': -5.0,
            'insufficient_cpu': -8.0,
            'insufficient_memory': -6.0,
            'insufficient_bandwidth': -4.0,
            'no_network_path': -10.0
        }
        
        return penalty_map.get(reason, -3.0)
    
    def _calculate_intermediate_reward(self, vnf: str, node: int) -> float:
        """计算中间步骤奖励"""
        # 基础成功奖励
        base_reward = 2.0
        
        # 资源效率奖励
        efficiency_bonus = self._calculate_resource_efficiency_bonus(node)
        
        # 网络优化奖励
        network_bonus = self._calculate_network_optimization_bonus(node)
        
        return base_reward + efficiency_bonus + network_bonus
    
    def _calculate_resource_efficiency_bonus(self, node_id: int) -> float:
        """计算资源效率奖励"""
        # 奖励选择资源适配度高的节点
        if len(self.current_node_resources[node_id]) < 2:
            return 0.0
        
        cpu_utilization = 1.0 - (self.current_node_resources[node_id, 0] / self.initial_node_resources[node_id, 0])
        memory_utilization = 1.0 - (self.current_node_resources[node_id, 1] / self.initial_node_resources[node_id, 1])
        
        # 适中的利用率最好（70-90%）
        optimal_utilization = 0.8
        cpu_efficiency = 1.0 - abs(cpu_utilization - optimal_utilization)
        memory_efficiency = 1.0 - abs(memory_utilization - optimal_utilization)
        
        return (cpu_efficiency + memory_efficiency) * 0.5
    
    def _calculate_network_optimization_bonus(self, node_id: int) -> float:
        """计算网络优化奖励"""
        if self.current_vnf_index == 0:
            return 0.0
        
        # 奖励选择距离上一个VNF较近的节点
        prev_vnf = self.service_chain[self.current_vnf_index - 1]
        prev_node = self.embedding_map.get(prev_vnf)
        
        if prev_node is None:
            return 0.0
        
        try:
            path_length = nx.shortest_path_length(self.graph, source=prev_node, target=node_id)
            # 路径越短，奖励越高
            max_distance = 5  # 假设最大距离
            distance_bonus = max(0, (max_distance - path_length) / max_distance)
            return distance_bonus * 1.0
        except nx.NetworkXNoPath:
            return -2.0  # 无路径的惩罚
    
    def _calculate_final_reward(self) -> Tuple[float, Dict[str, Any]]:
        """计算完成所有VNF嵌入后的最终奖励"""
        
        # 计算服务链路径指标
        chain_metrics = self._calculate_chain_metrics()
        
        # 使用原有的奖励函数
        info = {
            'success': True,
            'paths': chain_metrics['paths'],
            'total_delay': chain_metrics['total_delay'],
            'min_bandwidth': chain_metrics['min_bandwidth'],
            'resource_utilization': chain_metrics['resource_utilization']
        }
        
        # 计算基础奖励
        base_reward = compute_reward(info, self.reward_config)
        
        # 添加完成奖励
        completion_bonus = 20.0  # 成功完成所有VNF嵌入的奖励
        
        # 效率奖励
        efficiency_bonus = self._calculate_overall_efficiency_bonus(chain_metrics)
        
        final_reward = base_reward + completion_bonus + efficiency_bonus
        
        # 更新info
        info.update({
            'base_reward': base_reward,
            'completion_bonus': completion_bonus,
            'efficiency_bonus': efficiency_bonus,
            'final_reward': final_reward,
            'sar': 1.0,  # 成功完成
            'splat': chain_metrics['avg_delay']
        })
        
        return final_reward, info
    
    def _calculate_chain_metrics(self) -> Dict[str, Any]:
        """计算服务链的网络指标"""
        paths = []
        total_delay = 0.0
        min_bandwidth = float('inf')
        
        # 计算相邻VNF之间的路径指标
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
                path_jitters = []
                path_losses = []
                
                for j in range(len(path) - 1):
                    u, v = path[j], path[j + 1]
                    edge_attr = self._get_edge_attr(u, v)
                    
                    if edge_attr is not None and len(edge_attr) >= 4:
                        path_bandwidths.append(edge_attr[0])  # bandwidth
                        path_delay += edge_attr[1]            # delay
                        path_jitters.append(edge_attr[2])     # jitter
                        path_losses.append(edge_attr[3])      # loss
                
                if path_bandwidths:
                    path_min_bw = min(path_bandwidths)
                    path_avg_jitter = np.mean(path_jitters)
                    path_avg_loss = np.mean(path_losses)
                else:
                    path_min_bw = 100.0  # 默认值
                    path_avg_jitter = 0.1
                    path_avg_loss = 0.01
                
                paths.append({
                    "delay": path_delay,
                    "bandwidth": path_min_bw,
                    "hops": len(path) - 1,
                    "jitter": path_avg_jitter,
                    "loss": path_avg_loss
                })
                
                total_delay += path_delay
                min_bandwidth = min(min_bandwidth, path_min_bw)
                
            except nx.NetworkXNoPath:
                # 无路径连接，这不应该发生（前面已检查）
                paths.append({
                    "delay": 999.0,
                    "bandwidth": 0.0,
                    "hops": 999,
                    "jitter": 1.0,
                    "loss": 1.0
                })
                total_delay += 999.0
                min_bandwidth = 0.0
        
        # 计算资源利用率
        total_cpu_used = sum(
            self.initial_node_resources[node, 0] - self.current_node_resources[node, 0]
            for node in self.used_nodes
        )
        total_cpu_available = sum(self.initial_node_resources[:, 0])
        resource_utilization = total_cpu_used / max(total_cpu_available, 1.0)
        
        return {
            'paths': paths,
            'total_delay': total_delay,
            'avg_delay': total_delay / max(len(paths), 1),
            'min_bandwidth': min_bandwidth,
            'resource_utilization': resource_utilization
        }
    
    def _calculate_overall_efficiency_bonus(self, metrics: Dict) -> float:
        """计算整体效率奖励"""
        # 资源利用率奖励（适中最好）
        util_bonus = 2.0 * (1.0 - abs(metrics['resource_utilization'] - 0.7))
        
        # 网络效率奖励（延迟越低越好）
        delay_bonus = max(0, 3.0 - metrics['avg_delay'] / 2.0)
        
        # 带宽效率奖励
        bandwidth_bonus = min(2.0, metrics['min_bandwidth'] / 20.0)
        
        return util_bonus + delay_bonus + bandwidth_bonus
    
    def _get_edge_attr(self, u: int, v: int) -> np.ndarray:
        """获取边属性"""
        if (u, v) in self.edge_index_map:
            edge_idx = self.edge_index_map[(u, v)]
        elif (v, u) in self.edge_index_map:
            edge_idx = self.edge_index_map[(v, u)]
        else:
            return np.array([100.0, 1.0, 0.1, 0.01])  # 默认边属性
        
        return self.edge_features[edge_idx]
    
    def _handle_invalid_action(self, reason: str) -> Tuple[Data, float, bool, Dict]:
        """处理无效动作"""
        return self._get_state(), -10.0, True, {
            'success': False,
            'error': reason,
            'step': self.step_count
        }
    
    def _handle_completion(self) -> Tuple[Data, float, bool, Dict]:
        """处理已完成的情况"""
        return self._get_state(), 0.0, True, {
            'success': True,
            'already_completed': True,
            'step': self.step_count
        }
    
    def get_valid_actions(self) -> List[int]:
        """
        获取当前状态下的有效动作
        
        Returns:
            valid_actions: 有效节点ID列表
        """
        if self.current_vnf_index >= len(self.service_chain):
            return []
        
        current_vnf_req = self.vnf_requirements[self.current_vnf_index]
        valid_actions = []
        
        for node_id in range(self.action_dim):
            constraint_check = self._check_embedding_constraints(node_id, current_vnf_req)
            if constraint_check['valid']:
                valid_actions.append(node_id)
        
        # 确保至少有一个有效动作
        if not valid_actions:
            # 如果没有有效动作，返回所有未使用的节点
            valid_actions = [i for i in range(self.action_dim) if i not in self.used_nodes]
            if not valid_actions:
                valid_actions = [0]  # 最后的回退选择
        
        return valid_actions
    
    def render(self, mode='human') -> None:
        """
        可视化当前环境状态
        
        Args:
            mode: 渲染模式
        """
        print(f"\n{'='*50}")
        print(f"📊 VNF嵌入环境状态 (步数: {self.step_count})")
        print(f"{'='*50}")
        
        # 服务链信息
        print(f"🔗 服务链: {' -> '.join(self.service_chain)}")
        print(f"📍 当前VNF: {self.current_vnf_index}/{len(self.service_chain)} - ", end="")
        if self.current_vnf_index < len(self.service_chain):
            current_vnf = self.service_chain[self.current_vnf_index]
            current_req = self.vnf_requirements[self.current_vnf_index]
            print(f"{current_vnf} (CPU:{current_req['cpu']:.2f}, MEM:{current_req['memory']:.2f})")
        else:
            print("已完成所有VNF嵌入")
        
        # 嵌入状态
        print(f"\n📍 已嵌入VNF:")
        for vnf, node in self.embedding_map.items():
            print(f"   {vnf} -> 节点 {node}")
        
        # 节点资源状态
        print(f"\n💾 节点资源状态:")
        for i in range(min(5, self.action_dim)):  # 只显示前5个节点
            is_used = "🔴" if i in self.used_nodes else "🟢"
            cpu_ratio = self.current_node_resources[i, 0] / self.initial_node_resources[i, 0]
            print(f"   节点 {i}: {is_used} CPU可用率 {cpu_ratio:.1%}")
        
        # 有效动作
        valid_actions = self.get_valid_actions()
        print(f"\n✅ 有效动作: {valid_actions[:10]}{'...' if len(valid_actions) > 10 else ''}")
        print(f"   有效动作数: {len(valid_actions)}/{self.action_dim}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取环境信息
        
        Returns:
            info: 环境状态信息
        """
        return {
            'service_chain_length': len(self.service_chain),
            'current_vnf_index': self.current_vnf_index,
            'embedding_progress': self.current_vnf_index / len(self.service_chain),
            'used_nodes': list(self.used_nodes),
            'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
            'step_count': self.step_count,
            'valid_actions_count': len(self.get_valid_actions()),
            'resource_utilization': self._get_current_resource_utilization()
        }
    
    def _get_current_resource_utilization(self) -> Dict[str, float]:
        """获取当前资源利用率"""
        if len(self.used_nodes) == 0:
            return {'cpu': 0.0, 'memory': 0.0}
        
        total_cpu_used = 0.0
        total_memory_used = 0.0
        total_cpu_capacity = 0.0
        total_memory_capacity = 0.0
        
        for node_id in range(self.action_dim):
            total_cpu_capacity += self.initial_node_resources[node_id, 0]
            if len(self.initial_node_resources[node_id]) > 1:
                total_memory_capacity += self.initial_node_resources[node_id, 1]
            
            cpu_used = self.initial_node_resources[node_id, 0] - self.current_node_resources[node_id, 0]
            total_cpu_used += max(0, cpu_used)
            
            if len(self.current_node_resources[node_id]) > 1:
                memory_used = self.initial_node_resources[node_id, 1] - self.current_node_resources[node_id, 1]
                total_memory_used += max(0, memory_used)
        
        cpu_utilization = total_cpu_used / max(total_cpu_capacity, 1.0)
        memory_utilization = total_memory_used / max(total_memory_capacity, 1.0) if total_memory_capacity > 0 else 0.0
        
        return {
            'cpu': cpu_utilization,
            'memory': memory_utilization
        }
    
    def seed(self, seed: int = None) -> List[int]:
        """
        设置随机种子
        
        Args:
            seed: 随机种子
            
        Returns:
            seeds: 使用的种子列表
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            return [seed]
        return []
    
    def close(self):
        """关闭环境"""
        pass


# 测试函数
def test_vnf_environment():
    """测试修复后的VNF嵌入环境"""
    print("🧪 测试VNF嵌入环境...")
    
    # 创建测试网络
    import networkx as nx
    G = nx.erdos_renyi_graph(n=10, p=0.4, seed=42)
    
    # 节点特征：[CPU, Memory]
    node_features = np.random.rand(10, 2) * 0.8 + 0.2  # 0.2-1.0之间
    
    # 边特征：[Bandwidth, Delay, Jitter, Loss]  
    edge_features = np.random.rand(len(G.edges()), 4)
    edge_features[:, 0] = edge_features[:, 0] * 80 + 20  # 带宽 20-100
    edge_features[:, 1] = edge_features[:, 1] * 5 + 1    # 延迟 1-6
    edge_features[:, 2] = edge_features[:, 2] * 0.5      # 抖动 0-0.5
    edge_features[:, 3] = edge_features[:, 3] * 0.05     # 丢包 0-0.05
    
    # 奖励配置
    reward_config = {
        "alpha": 0.5, "beta": 0.2, "gamma": 0.2, "delta": 0.1, "penalty": 1.0
    }
    
    # 创建环境
    env = MultiVNFEmbeddingEnv(
        graph=G,
        node_features=node_features,
        edge_features=edge_features,
        reward_config=reward_config,
        chain_length_range=(3, 5)
    )
    
    print(f"✅ 环境创建成功")
    
    # 测试重置
    state = env.reset()
    print(f"✅ 重置测试: 状态类型={type(state)}")
    print(f"   节点特征形状: {state.x.shape}")
    print(f"   边索引形状: {state.edge_index.shape}")
    print(f"   VNF上下文: {state.vnf_context}")
    
    # 测试多步交互
    total_reward = 0.0
    step_count = 0
    
    while step_count < 10:  # 最多10步
        env.render()
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("❌ 没有有效动作")
            break
        
        # 随机选择动作
        action = np.random.choice(valid_actions)
        print(f"\n🎯 选择动作: {action}")
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"   奖励: {reward:.2f}")
        print(f"   完成: {done}")
        print(f"   信息: {info.get('success', False)}")
        
        if done:
            print(f"\n🎉 Episode完成!")
            print(f"   总奖励: {total_reward:.2f}")
            print(f"   总步数: {step_count}")
            if 'sar' in info:
                print(f"   SAR: {info['sar']:.2f}")
            if 'splat' in info:
                print(f"   SPLat: {info['splat']:.2f}")
            break
    
    # 测试环境信息
    env_info = env.get_info()
    print(f"\n📊 环境信息:")
    for key, value in env_info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ VNF嵌入环境测试完成!")


if __name__ == "__main__":
    test_vnf_environment()