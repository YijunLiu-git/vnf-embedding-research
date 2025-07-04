# env/enhanced_vnf_env_multi.py - 增强的VNF嵌入环境

import gym
import torch
import numpy as np
import networkx as nx
from gym import spaces
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Union, Any
from rewards.reward_v4_comprehensive_multi import compute_reward
import random
from collections import defaultdict

class EdgeAwareStateComputer:
    """
    Edge-Aware状态计算器
    
    核心功能：
    1. 动态路径质量评估
    2. 网络拥塞状态建模
    3. VNF依赖关系分析
    4. 边重要性评分
    """
    
    def __init__(self, graph, edge_features):
        self.graph = graph
        self.edge_features = edge_features
        self.edge_map = list(graph.edges())
        self.edge_index_map = {edge: idx for idx, edge in enumerate(self.edge_map)}
        
        # 缓存机制
        self.path_cache = {}
        self.quality_cache = {}
        
        print(f"🔧 EdgeAware状态计算器初始化:")
        print(f"   - 图节点数: {len(graph.nodes())}")
        print(f"   - 图边数: {len(graph.edges())}")
        print(f"   - 边特征维度: {edge_features.shape}")
    
    def compute_enhanced_state(self, vnf_chain, current_embeddings, current_vnf_index):
        """
        计算增强的Edge-Aware状态
        
        Returns:
            enhanced_state: 包含路径质量、拥塞状态、依赖关系的增强状态
        """
        
        # 1. 计算路径质量矩阵
        path_quality_matrix = self._compute_path_quality_matrix()
        
        # 2. 分析网络拥塞状态
        congestion_state = self._analyze_network_congestion(current_embeddings)
        
        # 3. 构建VNF依赖关系
        dependency_info = self._analyze_vnf_dependencies(vnf_chain, current_vnf_index)
        
        # 4. 计算边重要性权重
        edge_importance = self._compute_edge_importance(
            path_quality_matrix, congestion_state, dependency_info
        )
        
        # 5. 生成网络状态向量
        network_state_vector = self._generate_network_state_vector(
            path_quality_matrix, congestion_state, edge_importance
        )
        
        enhanced_state = {
            'path_quality_matrix': path_quality_matrix,
            'congestion_state': congestion_state,
            'dependency_info': dependency_info,
            'edge_importance': edge_importance,
            'network_state_vector': network_state_vector
        }
        
        return enhanced_state
    
    def _compute_path_quality_matrix(self):
        """计算所有节点对之间的路径质量"""
        quality_matrix = {}
        
        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source == target:
                    continue
                
                cache_key = (min(source, target), max(source, target))
                if cache_key in self.quality_cache:
                    quality_matrix[(source, target)] = self.quality_cache[cache_key]
                    continue
                
                try:
                    # 计算最短路径
                    shortest_path = nx.shortest_path(self.graph, source, target)
                    path_quality = self._evaluate_path_quality(shortest_path)
                    
                    # 计算替代路径数量
                    try:
                        all_paths = list(nx.all_simple_paths(
                            self.graph, source, target, cutoff=6
                        ))
                        alternative_count = len(all_paths) - 1  # 减去最短路径
                    except:
                        alternative_count = 0
                    
                    quality_info = {
                        'quality_score': path_quality['quality_score'],
                        'bandwidth': path_quality['bandwidth'],
                        'latency': path_quality['latency'],
                        'jitter': path_quality['jitter'],
                        'packet_loss': path_quality['packet_loss'],
                        'hops': len(shortest_path) - 1,
                        'path': shortest_path,
                        'alternative_paths': alternative_count,
                        'reliability': path_quality['reliability']
                    }
                    
                    quality_matrix[(source, target)] = quality_info
                    self.quality_cache[cache_key] = quality_info
                    
                except nx.NetworkXNoPath:
                    quality_matrix[(source, target)] = {
                        'quality_score': 0.0,
                        'bandwidth': 0.0,
                        'latency': float('inf'),
                        'jitter': float('inf'),
                        'packet_loss': 1.0,
                        'hops': float('inf'),
                        'path': None,
                        'alternative_paths': 0,
                        'reliability': 0.0
                    }
        
        return quality_matrix
    
    def _evaluate_path_quality(self, path):
        """评估单条路径的综合质量"""
        if len(path) < 2:
            return {
                'quality_score': 0.0,
                'bandwidth': 0.0,
                'latency': 0.0,
                'jitter': 0.0,
                'packet_loss': 0.0,
                'reliability': 0.0
            }
        
        min_bandwidth = float('inf')
        total_latency = 0.0
        total_jitter = 0.0
        total_packet_loss = 0.0
        reliability_product = 1.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_attr = self._get_edge_attributes(u, v)
            
            # 提取边特征 [bandwidth, latency, jitter, packet_loss]
            bandwidth = edge_attr[0]
            latency = edge_attr[1]
            jitter = edge_attr[2]
            packet_loss = edge_attr[3]
            
            # 瓶颈带宽
            min_bandwidth = min(min_bandwidth, bandwidth)
            
            # 累积延迟和抖动
            total_latency += latency
            total_jitter += jitter
            
            # 累积丢包率（简化模型：1 - (1-p1)*(1-p2)*...）
            total_packet_loss = 1 - (1 - total_packet_loss) * (1 - packet_loss)
            
            # 可靠性
            edge_reliability = 1.0 - packet_loss
            reliability_product *= edge_reliability
        
        # 综合质量评分 (0-1之间，越高越好)
        # 归一化各个指标
        bandwidth_score = min(min_bandwidth / 100.0, 1.0)  # 假设100为满分带宽
        latency_score = max(0, 1.0 - total_latency / 100.0)  # 假设100ms为延迟上限
        jitter_score = max(0, 1.0 - total_jitter / 5.0)  # 假设5ms为抖动上限
        loss_score = 1.0 - min(total_packet_loss, 1.0)
        reliability_score = reliability_product
        
        # 加权综合评分
        quality_score = (
            bandwidth_score * 0.25 +
            latency_score * 0.25 +
            jitter_score * 0.2 +
            loss_score * 0.15 +
            reliability_score * 0.15
        )
        
        return {
            'quality_score': quality_score,
            'bandwidth': min_bandwidth,
            'latency': total_latency,
            'jitter': total_jitter,
            'packet_loss': total_packet_loss,
            'reliability': reliability_product
        }
    
    def _analyze_network_congestion(self, current_embeddings):
        """分析网络拥塞状态"""
        congestion_state = {
            'node_congestion': {},
            'edge_congestion': {},
            'hotspots': [],
            'bottlenecks': []
        }
        
        # 计算节点拥塞
        node_load = defaultdict(int)
        for vnf, node in current_embeddings.items():
            node_load[node] += 1
        
        max_load = max(node_load.values()) if node_load else 1
        for node in self.graph.nodes():
            load = node_load.get(node, 0)
            congestion_level = load / max(max_load, 1)
            congestion_state['node_congestion'][node] = congestion_level
            
            # 识别热点节点
            if congestion_level > 0.7:
                congestion_state['hotspots'].append(node)
        
        # 计算边拥塞
        edge_traffic = defaultdict(int)
        
        # 基于当前嵌入计算边流量
        vnf_nodes = list(current_embeddings.values())
        for i in range(len(vnf_nodes) - 1):
            source = vnf_nodes[i]
            target = vnf_nodes[i + 1]
            
            try:
                path = nx.shortest_path(self.graph, source, target)
                for j in range(len(path) - 1):
                    edge_key = tuple(sorted([path[j], path[j + 1]]))
                    edge_traffic[edge_key] += 1
            except nx.NetworkXNoPath:
                continue
        
        max_traffic = max(edge_traffic.values()) if edge_traffic else 1
        
        for u, v in self.graph.edges():
            edge_key = tuple(sorted([u, v]))
            traffic = edge_traffic.get(edge_key, 0)
            congestion_level = traffic / max(max_traffic, 1)
            congestion_state['edge_congestion'][(u, v)] = congestion_level
            
            # 识别瓶颈边
            if congestion_level > 0.8:
                congestion_state['bottlenecks'].append((u, v))
        
        return congestion_state
    
    def _analyze_vnf_dependencies(self, vnf_chain, current_vnf_index):
        """分析VNF依赖关系"""
        dependency_info = {
            'chain_progress': current_vnf_index / len(vnf_chain) if vnf_chain else 0,
            'remaining_vnfs': len(vnf_chain) - current_vnf_index,
            'dependency_strength': {},
            'critical_path_nodes': [],
            'flexibility_score': 0.0
        }
        
        if current_vnf_index > 0:
            # 计算与前序VNF的依赖强度
            for i in range(current_vnf_index):
                dependency_strength = 1.0 / (current_vnf_index - i)  # 距离越近依赖越强
                dependency_info['dependency_strength'][i] = dependency_strength
        
        # 计算灵活性评分（基于网络连通性）
        if current_vnf_index < len(vnf_chain):
            total_connectivity = 0
            for node in self.graph.nodes():
                connectivity = len(list(self.graph.neighbors(node)))
                total_connectivity += connectivity
            
            avg_connectivity = total_connectivity / len(self.graph.nodes())
            dependency_info['flexibility_score'] = min(avg_connectivity / 10.0, 1.0)
        
        return dependency_info
    
    def _compute_edge_importance(self, path_quality_matrix, congestion_state, dependency_info):
        """计算边重要性权重"""
        edge_importance = {}
        
        for u, v in self.graph.edges():
            importance_score = 0.0
            
            # 1. 基于路径质量的重要性
            paths_through_edge = 0
            quality_sum = 0.0
            
            for (source, target), path_info in path_quality_matrix.items():
                if path_info['path'] and len(path_info['path']) > 1:
                    path = path_info['path']
                    for i in range(len(path) - 1):
                        if (path[i] == u and path[i+1] == v) or (path[i] == v and path[i+1] == u):
                            paths_through_edge += 1
                            quality_sum += path_info['quality_score']
                            break
            
            if paths_through_edge > 0:
                avg_quality = quality_sum / paths_through_edge
                importance_score += avg_quality * 0.4
            
            # 2. 基于拥塞状态的重要性
            congestion_level = congestion_state['edge_congestion'].get((u, v), 0)
            # 拥塞越严重，重要性越高（需要更多关注）
            importance_score += congestion_level * 0.3
            
            # 3. 基于网络拓扑的重要性（中心性）
            try:
                edge_betweenness = nx.edge_betweenness_centrality(self.graph)
                betweenness_score = edge_betweenness.get((u, v), edge_betweenness.get((v, u), 0))
                importance_score += betweenness_score * 0.3
            except:
                importance_score += 0.1  # 默认值
            
            edge_importance[(u, v)] = min(importance_score, 1.0)
        
        return edge_importance
    
    def _generate_network_state_vector(self, path_quality_matrix, congestion_state, edge_importance):
        """生成网络状态向量用于GNN"""
        
        # 计算全局网络统计
        total_quality = 0.0
        total_paths = 0
        
        for path_info in path_quality_matrix.values():
            if path_info['quality_score'] > 0:
                total_quality += path_info['quality_score']
                total_paths += 1
        
        avg_network_quality = total_quality / max(total_paths, 1)
        
        # 计算拥塞统计
        node_congestion_levels = list(congestion_state['node_congestion'].values())
        edge_congestion_levels = list(congestion_state['edge_congestion'].values())
        
        avg_node_congestion = np.mean(node_congestion_levels) if node_congestion_levels else 0.0
        avg_edge_congestion = np.mean(edge_congestion_levels) if edge_congestion_levels else 0.0
        
        # 计算重要性统计
        importance_values = list(edge_importance.values())
        avg_edge_importance = np.mean(importance_values) if importance_values else 0.0
        
        # 网络连通性指标
        connectivity_score = nx.average_node_connectivity(self.graph) / len(self.graph.nodes())
        
        # 构建网络状态向量 [8维]
        network_state_vector = np.array([
            avg_network_quality,      # 平均网络质量
            avg_node_congestion,      # 平均节点拥塞
            avg_edge_congestion,      # 平均边拥塞
            avg_edge_importance,      # 平均边重要性
            connectivity_score,       # 连通性评分
            len(congestion_state['hotspots']) / len(self.graph.nodes()),  # 热点比例
            len(congestion_state['bottlenecks']) / len(self.graph.edges()),  # 瓶颈比例
            total_paths / (len(self.graph.nodes()) ** 2)  # 路径密度
        ], dtype=np.float32)
        
        return network_state_vector
    
    def _get_edge_attributes(self, u, v):
        """获取边属性"""
        if (u, v) in self.edge_index_map:
            edge_idx = self.edge_index_map[(u, v)]
        elif (v, u) in self.edge_index_map:
            edge_idx = self.edge_index_map[(v, u)]
        else:
            # 返回默认属性
            return np.array([50.0, 10.0, 1.0, 0.01], dtype=np.float32)
        
        return self.edge_features[edge_idx]


class EnhancedVNFEmbeddingEnv(gym.Env):
    """
    增强的VNF嵌入环境 - 集成Edge-Aware状态计算
    
    主要增强：
    1. 动态路径质量感知
    2. 网络拥塞状态建模
    3. 增强的状态表示
    4. 网络感知的动作选择
    """
    
    def __init__(self, graph, node_features, edge_features, reward_config, 
                 chain_length_range=(2, 5), config=None):
        super().__init__()
        
        self.config = config or {}
        self.graph = graph
        self._original_node_features = node_features.copy()
        self._original_edge_features = edge_features.copy()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_nodes = len(graph.nodes())
        self.reward_config = reward_config
        self.chain_length_range = chain_length_range
        self.max_episode_steps = config.get('train', {}).get('max_episode_steps', 20)
        
        # 🔧 新增：Edge-Aware状态计算器
        self.edge_aware_computer = EdgeAwareStateComputer(graph, edge_features)
        
        # 状态和动作空间
        self.state_dim = node_features.shape[1] if len(node_features.shape) > 1 else node_features.shape[0]
        self.edge_dim = edge_features.shape[1] if len(edge_features.shape) > 1 else edge_features.shape[0]
        self.action_dim = self.num_nodes
        
        self.action_space = spaces.Discrete(self.action_dim)
        max_nodes = self.num_nodes
        max_features = self.state_dim + 16  # 增加网络状态向量的维度
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_nodes * max_features,),
            dtype=np.float32
        )
        
        # 环境状态
        self.edge_map = list(self.graph.edges())
        self.edge_index_map = {edge: idx for idx, edge in enumerate(self.edge_map)}
        
        # VNF嵌入状态
        self.service_chain = []
        self.vnf_requirements = []
        self.current_vnf_index = 0
        self.embedding_map = {}
        self.used_nodes = set()
        self.step_count = 0
        self.initial_node_resources = node_features.copy()
        self.current_node_resources = node_features.copy()
        
        # 🔧 新增：增强状态缓存
        self.enhanced_state_cache = None
        self.last_enhanced_state_episode = -1
        
        # 场景相关
        self.current_scenario_name = "normal_operation"
        self.scenario_display_name = "正常运营期"
        self.scenario_applied = False
        
        print(f"🌍 增强VNF嵌入环境初始化完成:")
        print(f"   - 网络节点数: {self.num_nodes}")
        print(f"   - 网络边数: {len(self.graph.edges())}")
        print(f"   - 节点特征维度: {self.state_dim}")
        print(f"   - 边特征维度: {self.edge_dim}")
        print(f"   - Edge-Aware状态计算: 启用")
        
        self.reset()
    
    def apply_scenario_config(self, scenario_config):
        """应用场景配置（保持原有接口兼容性）"""
        try:
            self.current_scenario_name = scenario_config.get('scenario_name', 'unknown')
            
            scenario_display_names = {
                'normal_operation': '正常运营期',
                'peak_congestion': '高峰拥塞期', 
                'failure_recovery': '故障恢复期',
                'extreme_pressure': '极限压力期'
            }
            self.scenario_display_name = scenario_display_names.get(self.current_scenario_name, self.current_scenario_name)
            
            if 'vnf_requirements' in scenario_config:
                self._scenario_vnf_config = scenario_config['vnf_requirements'].copy()
            
            if 'topology' in scenario_config and 'node_resources' in scenario_config['topology']:
                node_res = scenario_config['topology']['node_resources']
                cpu_factor = node_res.get('cpu', 1.0)
                memory_factor = node_res.get('memory', 1.0)
                
                self.current_node_resources = self._original_node_features * cpu_factor
                self.initial_node_resources = self.current_node_resources.copy()
            
            if 'reward' in scenario_config:
                self.reward_config.update(scenario_config['reward'])
            
            self.scenario_applied = True
            print(f"✅ 增强环境场景配置应用成功: {self.scenario_display_name}")
            
        except Exception as e:
            print(f"⚠️ 增强环境场景配置应用出错: {e}")
    
    def reset(self) -> Data:
        """重置环境并计算增强状态"""
        try:
            # 基础重置逻辑（保持原有）
            if hasattr(self, '_scenario_vnf_config') and self._scenario_vnf_config:
                vnf_config = self._scenario_vnf_config.copy()
            else:
                vnf_config = self.config.get('vnf_requirements', {
                    'cpu_min': 0.03, 'cpu_max': 0.15,
                    'memory_min': 0.02, 'memory_max': 0.12,
                    'bandwidth_min': 3.0, 'bandwidth_max': 10.0,
                    'chain_length_range': (3, 6)
                })
            
            # 生成服务链
            chain_length_range = vnf_config.get('chain_length_range', (3, 6))
            chain_length = np.random.randint(chain_length_range[0], chain_length_range[1] + 1)
            self.service_chain = [f"VNF_{i}" for i in range(chain_length)]
            
            # 生成VNF需求
            self.vnf_requirements = []
            for i in range(chain_length):
                cpu_req = np.random.uniform(vnf_config['cpu_min'], vnf_config['cpu_max'])
                memory_req = np.random.uniform(vnf_config['memory_min'], vnf_config['memory_max'])
                bandwidth_req = np.random.uniform(vnf_config.get('bandwidth_min', 2.0), 
                                                vnf_config.get('bandwidth_max', 8.0))
                
                self.vnf_requirements.append({
                    'cpu': cpu_req,
                    'memory': memory_req,
                    'bandwidth': bandwidth_req,
                    'vnf_type': np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                })
            
            # 重置状态
            self.current_vnf_index = 0
            self.embedding_map.clear()
            self.used_nodes.clear()
            self.step_count = 0
            self.enhanced_state_cache = None
            
            print(f"\n🔄 增强环境重置 ({self.scenario_display_name}):")
            print(f"   服务链长度: {len(self.service_chain)}")
            
            return self._get_enhanced_state()
            
        except Exception as e:
            print(f"⚠️ 增强环境重置出错: {e}")
            # 回退到基础重置
            self.current_vnf_index = 0
            self.embedding_map.clear()
            self.used_nodes.clear()
            self.step_count = 0
            return self._get_basic_state()
    
    def _get_enhanced_state(self) -> Data:
        """🔧 核心改进：获取增强的Edge-Aware状态"""
        try:
            # 计算增强状态
            enhanced_state_info = self.edge_aware_computer.compute_enhanced_state(
                self.service_chain, self.embedding_map, self.current_vnf_index
            )
            
            # 基础节点特征 (保持8维)
            enhanced_node_features = self._compute_enhanced_node_features(enhanced_state_info)
            
            # 增强边特征
            enhanced_edge_features = self._compute_enhanced_edge_features(enhanced_state_info)
            
            # 构建PyG数据对象
            x = torch.tensor(enhanced_node_features, dtype=torch.float32)
            edge_index = torch.tensor(np.array(self.edge_map).T, dtype=torch.long)
            edge_attr = torch.tensor(enhanced_edge_features, dtype=torch.float32)
            
            # VNF上下文
            vnf_context = self._compute_vnf_context()
            
            # 🔧 新增：网络状态向量
            network_state_vector = torch.tensor(
                enhanced_state_info['network_state_vector'], dtype=torch.float32
            )
            
            return Data(
                x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr,
                vnf_context=vnf_context,
                network_state=network_state_vector,  # 新增网络状态
                enhanced_info=enhanced_state_info     # 新增完整增强信息
            )
            
        except Exception as e:
            print(f"⚠️ 增强状态计算失败，使用基础状态: {e}")
            return self._get_basic_state()
    
    def _compute_enhanced_node_features(self, enhanced_state_info):
        """计算增强的节点特征"""
        num_nodes = len(self.graph.nodes())
        enhanced_features = np.zeros((num_nodes, 8))
        
        # 基础特征 (前4维)
        enhanced_features[:, :4] = self.current_node_resources
        
        # 增强特征 (后4维)
        congestion_state = enhanced_state_info['congestion_state']
        
        for node_id in range(num_nodes):
            # 第5维: 节点占用状态
            enhanced_features[node_id, 4] = 1.0 if node_id in self.used_nodes else 0.0
            
            # 第6维: 节点拥塞级别
            enhanced_features[node_id, 5] = congestion_state['node_congestion'].get(node_id, 0.0)
            
            # 第7维: 节点连通性
            connectivity = len(list(self.graph.neighbors(node_id))) / (num_nodes - 1)
            enhanced_features[node_id, 6] = connectivity
            
            # 第8维: 节点在关键路径上的重要性
            importance = 0.0
            path_quality_matrix = enhanced_state_info['path_quality_matrix']
            
            for path_info in path_quality_matrix.values():
                if path_info['path'] and node_id in path_info['path']:
                    importance += path_info['quality_score']
            
            enhanced_features[node_id, 7] = min(importance / 10.0, 1.0)  # 归一化
        
        return enhanced_features
    
    def _compute_enhanced_edge_features(self, enhanced_state_info):
        """计算增强的边特征"""
        num_edges = len(self.edge_map)
        enhanced_features = np.zeros((num_edges, self.edge_dim + 2))  # 增加2维
        
        congestion_state = enhanced_state_info['congestion_state']
        edge_importance = enhanced_state_info['edge_importance']
        
        for i, (u, v) in enumerate(self.edge_map):
            # 基础边特征 (前4维)
            enhanced_features[i, :self.edge_dim] = self.edge_features[i]
            
            # 增强特征
            if self.edge_dim + 2 <= enhanced_features.shape[1]:
                # 第5维: 边拥塞级别
                congestion = congestion_state['edge_congestion'].get((u, v), 0.0)
                enhanced_features[i, self.edge_dim] = congestion
                
                # 第6维: 边重要性权重
                importance = edge_importance.get((u, v), 0.0)
                enhanced_features[i, self.edge_dim + 1] = importance
        
        return enhanced_features[:, :self.edge_dim]  # 保持原有维度输出
    
    def _compute_vnf_context(self):
        """计算VNF上下文"""
        if self.current_vnf_index < len(self.vnf_requirements):
            current_vnf_req = self.vnf_requirements[self.current_vnf_index]
            vnf_context = torch.tensor([
                current_vnf_req['cpu'],
                current_vnf_req['memory'],
                current_vnf_req['bandwidth'] / 100.0,
                current_vnf_req['vnf_type'] / 3.0,
                self.current_vnf_index / len(self.service_chain),
                (len(self.service_chain) - self.current_vnf_index) / len(self.service_chain)
            ], dtype=torch.float32)
        else:
            vnf_context = torch.zeros(6, dtype=torch.float32)
        
        return vnf_context
    
    def _get_basic_state(self) -> Data:
        """获取基础状态（兼容性保证）"""
        enhanced_node_features = self.current_node_resources.copy()
        num_nodes = len(self.graph.nodes())
        
        # 确保节点特征维度为8
        if enhanced_node_features.shape[1] < 8:
            padding_dims = 8 - enhanced_node_features.shape[1]
            padding = np.zeros((num_nodes, padding_dims))
            
            for node_id in range(num_nodes):
                if padding_dims >= 1:
                    padding[node_id, 0] = 1.0 if node_id in self.used_nodes else 0.0
                if padding_dims >= 2:
                    if self.initial_node_resources[node_id, 0] > 0:
                        cpu_util = 1.0 - (self.current_node_resources[node_id, 0] / self.initial_node_resources[node_id, 0])
                        padding[node_id, 1] = max(0.0, min(1.0, cpu_util))
            
            enhanced_node_features = np.hstack([enhanced_node_features, padding])
        
        x = torch.tensor(enhanced_node_features, dtype=torch.float32)
        edge_index = torch.tensor(np.array(self.edge_map).T, dtype=torch.long)
        edge_attr = torch.tensor(self.edge_features, dtype=torch.float32)
        vnf_context = self._compute_vnf_context()
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, vnf_context=vnf_context)
    
    def get_enhanced_valid_actions(self) -> List[int]:
        """🔧 核心改进：获取增强的有效动作（考虑网络质量）"""
        if self.current_vnf_index >= len(self.vnf_requirements):
            return []
        
        current_vnf_req = self.vnf_requirements[self.current_vnf_index]
        valid_actions = []
        
        # 获取增强状态信息
        if hasattr(self, 'enhanced_state_cache') and self.enhanced_state_cache:
            enhanced_info = self.enhanced_state_cache
        else:
            enhanced_info = self.edge_aware_computer.compute_enhanced_state(
                self.service_chain, self.embedding_map, self.current_vnf_index
            )
        
        for node in range(self.num_nodes):
            # 基础约束检查
            constraint_check = self._check_embedding_constraints(node, current_vnf_req)
            if not constraint_check['valid']:
                continue
            
            # 🔧 新增：网络质量约束检查
            if self._check_enhanced_network_constraints(node, current_vnf_req, enhanced_info):
                valid_actions.append(node)
        
        return valid_actions
    
    def _check_enhanced_network_constraints(self, node, vnf_req, enhanced_info):
        """🔧 新增：检查增强的网络质量约束"""
        if self.current_vnf_index == 0:
            return True  # 第一个VNF没有路径约束
        
        # 检查与前一个VNF的连接质量
        prev_vnf = self.service_chain[self.current_vnf_index - 1]
        prev_node = self.embedding_map.get(prev_vnf)
        
        if prev_node is None:
            return True
        
        # 从路径质量矩阵获取质量信息
        path_quality_matrix = enhanced_info['path_quality_matrix']
        path_info = path_quality_matrix.get((prev_node, node), {})
        
        # 设定质量阈值
        bandwidth_requirement = vnf_req.get('bandwidth', 0)
        latency_tolerance = 100.0  # ms
        quality_threshold = 0.3  # 最低质量评分
        
        # 检查路径质量
        if path_info:
            bandwidth_ok = path_info.get('bandwidth', 0) >= bandwidth_requirement
            latency_ok = path_info.get('latency', float('inf')) <= latency_tolerance
            quality_ok = path_info.get('quality_score', 0) >= quality_threshold
            
            return bandwidth_ok and latency_ok and quality_ok
        
        return False  # 无路径连接
    
    def _check_embedding_constraints(self, node: int, vnf_req: Dict) -> Dict[str, Any]:
        """检查节点是否满足VNF的资源约束"""
        cpu_req = vnf_req['cpu']
        mem_req = vnf_req['memory']
        
        if node in self.used_nodes:
            return {'valid': False, 'reason': 'node_occupied', 'details': f'节点 {node} 已被占用'}
        
        if self.current_node_resources[node, 0] < cpu_req:
            return {'valid': False, 'reason': 'insufficient_cpu', 
                   'details': f'节点 {node} CPU不足: 需要{cpu_req:.3f}, 可用{self.current_node_resources[node, 0]:.3f}'}
        
        if len(self.current_node_resources[node]) > 1 and self.current_node_resources[node, 1] < mem_req:
            return {'valid': False, 'reason': 'insufficient_memory', 
                   'details': f'节点 {node} 内存不足: 需要{mem_req:.3f}, 可用{self.current_node_resources[node, 1]:.3f}'}
        
        return {'valid': True, 'reason': None, 'details': None}
    
    def step(self, action: int) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """执行动作 - 使用增强状态"""
        self.step_count += 1
        
        if action >= self.action_dim:
            return self._handle_invalid_action(f"动作超出范围: {action} >= {self.action_dim}")
        
        if self.current_vnf_index >= len(self.service_chain):
            return self._handle_completion()
        
        current_vnf = self.service_chain[self.current_vnf_index]
        current_vnf_req = self.vnf_requirements[self.current_vnf_index]
        target_node = action
        
        constraint_check = self._check_embedding_constraints(target_node, current_vnf_req)
        
        if not constraint_check['valid']:
            penalty_factor = self.reward_config.get('constraint_penalty_factor', 1.0)
            base_penalty = self._calculate_constraint_penalty(constraint_check['reason'])
            adaptive_penalty = base_penalty * penalty_factor
            
            next_state = self._get_enhanced_state()
            return next_state, adaptive_penalty, False, {
                'success': False,
                'constraint_violation': constraint_check['reason'],
                'details': constraint_check['details'],
                'adaptive_penalty_factor': penalty_factor
            }
        
        # 执行嵌入
        self.embedding_map[current_vnf] = target_node
        self.used_nodes.add(target_node)
        self._update_node_resources(target_node, current_vnf_req)
        self.current_vnf_index += 1
        
        done = (self.current_vnf_index >= len(self.service_chain)) or (self.step_count >= self.max_episode_steps)
        
        if done and self.current_vnf_index >= len(self.service_chain):
            # 完成嵌入，计算最终奖励
            reward, info = self._calculate_enhanced_final_reward()
            
            info.update({
                'success': True,
                'embedding_completed': True,
                'total_steps': self.step_count,
                'enhanced_features_used': True
            })
        else:
            # 中间步骤奖励
            reward = self._calculate_enhanced_intermediate_reward(current_vnf, target_node)
            info = {
                'success': True,
                'embedded_vnf': current_vnf,
                'target_node': target_node,
                'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
                'step': self.step_count
            }
        
        next_state = self._get_enhanced_state()
        return next_state, reward, done, info
    
    def _calculate_enhanced_final_reward(self) -> Tuple[float, Dict[str, Any]]:
        """🔧 新增：计算增强的最终奖励"""
        try:
            # 获取增强状态信息
            enhanced_info = self.edge_aware_computer.compute_enhanced_state(
                self.service_chain, self.embedding_map, self.current_vnf_index
            )
            
            # 计算增强的链指标
            chain_metrics = self._calculate_enhanced_chain_metrics(enhanced_info)
            
            info = {
                'success': True,
                'paths': chain_metrics['paths'],
                'total_delay': chain_metrics['total_delay'],
                'min_bandwidth': chain_metrics['min_bandwidth'],
                'resource_utilization': chain_metrics['resource_utilization'],
                'avg_jitter': chain_metrics['avg_jitter'],
                'avg_loss': chain_metrics['avg_loss'],
                'is_edge_aware': True,  # 标记为edge-aware版本
                'path_quality_score': chain_metrics['avg_quality_score'],  # 新增
                'network_efficiency': chain_metrics['network_efficiency'],  # 新增
                'congestion_level': chain_metrics['congestion_level'],     # 新增
                'enhanced_features_used': True
            }
            
            # 使用增强奖励计算
            base_reward = self._compute_reward(info)
            completion_bonus = self.reward_config.get('completion_bonus', 15.0)
            
            # 🔧 新增：Edge-Aware特有奖励
            edge_aware_bonus = self._calculate_edge_aware_bonus(chain_metrics, enhanced_info)
            
            final_reward = float(base_reward) + float(completion_bonus) + float(edge_aware_bonus)
            
            info.update({
                'base_reward': base_reward,
                'completion_bonus': completion_bonus,
                'edge_aware_bonus': edge_aware_bonus,
                'final_reward': final_reward,
                'sar': len(self.embedding_map) / len(self.service_chain),
                'splat': chain_metrics.get('avg_delay', 0.0)
            })
            
            return final_reward, info
            
        except Exception as e:
            print(f"⚠️ 增强奖励计算失败: {e}")
            # 回退到基础奖励
            return self._calculate_basic_final_reward()
    
    def _calculate_enhanced_chain_metrics(self, enhanced_info):
        """计算增强的服务链指标"""
        paths = []
        total_delay = 0.0
        min_bandwidth = float('inf')
        total_jitter = 0.0
        total_loss = 0.0
        total_quality_score = 0.0
        
        path_quality_matrix = enhanced_info['path_quality_matrix']
        congestion_state = enhanced_info['congestion_state']
        
        if not self.embedding_map or len(self.embedding_map) < len(self.service_chain):
            return self._get_default_chain_metrics()
        
        for i in range(len(self.service_chain) - 1):
            vnf1 = self.service_chain[i]
            vnf2 = self.service_chain[i + 1]
            node1 = self.embedding_map.get(vnf1)
            node2 = self.embedding_map.get(vnf2)
            
            if node1 is None or node2 is None:
                continue
            
            # 从增强状态获取路径信息
            path_info = path_quality_matrix.get((node1, node2), {})
            
            if path_info and path_info.get('path'):
                path_data = {
                    "delay": path_info.get('latency', 0.0),
                    "bandwidth": path_info.get('bandwidth', 0.0),
                    "hops": path_info.get('hops', 0),
                    "jitter": path_info.get('jitter', 0.0),
                    "loss": path_info.get('packet_loss', 0.0),
                    "quality_score": path_info.get('quality_score', 0.0),
                    "reliability": path_info.get('reliability', 1.0)
                }
                
                paths.append(path_data)
                
                total_delay += path_data["delay"]
                min_bandwidth = min(min_bandwidth, path_data["bandwidth"])
                total_jitter += path_data["jitter"]
                total_loss += path_data["loss"]
                total_quality_score += path_data["quality_score"]
        
        # 计算资源利用率
        total_cpu_used = sum(self.initial_node_resources[node, 0] - self.current_node_resources[node, 0] 
                            for node in self.used_nodes)
        total_cpu_available = sum(self.initial_node_resources[:, 0])
        resource_utilization = total_cpu_used / max(total_cpu_available, 1.0)
        
        # 计算网络效率指标
        avg_node_congestion = np.mean(list(congestion_state['node_congestion'].values()))
        avg_edge_congestion = np.mean(list(congestion_state['edge_congestion'].values()))
        network_efficiency = 1.0 - (avg_node_congestion + avg_edge_congestion) / 2.0
        
        return {
            'paths': paths,
            'total_delay': total_delay,
            'avg_delay': total_delay / max(len(paths), 1) if total_delay != float('inf') else float('inf'),
            'min_bandwidth': min_bandwidth if min_bandwidth != float('inf') else 0.0,
            'resource_utilization': resource_utilization,
            'avg_jitter': total_jitter / max(len(paths), 1) if paths else 0.0,
            'avg_loss': total_loss / max(len(paths), 1) if paths else 0.0,
            'avg_quality_score': total_quality_score / max(len(paths), 1) if paths else 0.0,
            'network_efficiency': network_efficiency,
            'congestion_level': (avg_node_congestion + avg_edge_congestion) / 2.0
        }
    
    def _calculate_edge_aware_bonus(self, chain_metrics, enhanced_info):
        """🔧 新增：计算Edge-Aware特有奖励"""
        bonus = 0.0
        
        # 1. 路径质量奖励
        avg_quality = chain_metrics.get('avg_quality_score', 0.0)
        quality_bonus = avg_quality * 20.0  # 最高20分
        
        # 2. 网络效率奖励
        network_efficiency = chain_metrics.get('network_efficiency', 0.0)
        efficiency_bonus = network_efficiency * 15.0  # 最高15分
        
        # 3. 拥塞避免奖励
        congestion_level = chain_metrics.get('congestion_level', 0.0)
        congestion_bonus = (1.0 - congestion_level) * 10.0  # 最高10分
        
        # 4. 路径多样性奖励
        diversity_bonus = 0.0
        paths = chain_metrics.get('paths', [])
        if len(paths) > 1:
            hop_variance = np.var([p.get('hops', 0) for p in paths])
            diversity_bonus = min(hop_variance, 5.0)  # 最高5分
        
        total_bonus = quality_bonus + efficiency_bonus + congestion_bonus + diversity_bonus
        
        print(f"🎯 Edge-Aware奖励分解:")
        print(f"   路径质量奖励: {quality_bonus:.2f}")
        print(f"   网络效率奖励: {efficiency_bonus:.2f}")
        print(f"   拥塞避免奖励: {congestion_bonus:.2f}")
        print(f"   路径多样性奖励: {diversity_bonus:.2f}")
        print(f"   Edge-Aware总奖励: {total_bonus:.2f}")
        
        return total_bonus
    
    def _calculate_enhanced_intermediate_reward(self, vnf: str, node: int) -> float:
        """计算增强的中间步骤奖励"""
        try:
            base_reward = self.reward_config.get('base_reward', 10.0)
            
            # 🔧 新增：基于增强状态的奖励
            if hasattr(self, 'enhanced_state_cache') and self.enhanced_state_cache:
                enhanced_info = self.enhanced_state_cache
                
                # 路径质量奖励
                if self.current_vnf_index > 0:
                    prev_vnf = self.service_chain[self.current_vnf_index - 1]
                    prev_node = self.embedding_map.get(prev_vnf)
                    
                    if prev_node is not None:
                        path_quality_matrix = enhanced_info['path_quality_matrix']
                        path_info = path_quality_matrix.get((prev_node, node), {})
                        quality_score = path_info.get('quality_score', 0.0)
                        quality_bonus = quality_score * 5.0
                        base_reward += quality_bonus
                
                # 拥塞避免奖励
                congestion_state = enhanced_info['congestion_state']
                node_congestion = congestion_state['node_congestion'].get(node, 0.0)
                congestion_bonus = (1.0 - node_congestion) * 3.0
                base_reward += congestion_bonus
            
            return float(base_reward)
            
        except Exception as e:
            return self.reward_config.get('base_reward', 10.0)
    
    # 保持原有方法的兼容性
    def _calculate_basic_final_reward(self):
        """基础最终奖励计算（兼容性）"""
        try:
            chain_metrics = self._get_default_chain_metrics()
            
            info = {
                'success': True,
                'paths': chain_metrics['paths'],
                'total_delay': chain_metrics['total_delay'],
                'min_bandwidth': chain_metrics['min_bandwidth'],
                'resource_utilization': chain_metrics['resource_utilization'],
                'avg_jitter': chain_metrics['avg_jitter'],
                'avg_loss': chain_metrics['avg_loss'],
                'is_edge_aware': True
            }
            
            base_reward = self._compute_reward(info)
            completion_bonus = self.reward_config.get('completion_bonus', 15.0)
            final_reward = float(base_reward) + float(completion_bonus)
            
            info.update({
                'base_reward': base_reward,
                'completion_bonus': completion_bonus,
                'final_reward': final_reward,
                'sar': len(self.embedding_map) / len(self.service_chain),
                'splat': chain_metrics.get('avg_delay', 0.0)
            })
            
            return final_reward, info
            
        except Exception as e:
            default_reward = 50.0
            default_info = {
                'success': True,
                'base_reward': 10.0,
                'completion_bonus': 15.0,
                'final_reward': default_reward,
                'sar': 1.0,
                'splat': 0.0
            }
            return default_reward, default_info
    
    def _get_default_chain_metrics(self):
        """获取默认链指标"""
        paths = []
        total_delay = 0.0
        min_bandwidth = float('inf')
        
        for i in range(len(self.service_chain) - 1):
            vnf1 = self.service_chain[i]
            vnf2 = self.service_chain[i + 1]
            node1 = self.embedding_map.get(vnf1)
            node2 = self.embedding_map.get(vnf2)
            
            if node1 is None or node2 is None:
                continue
            
            try:
                path = nx.shortest_path(self.graph, source=node1, target=node2)
                path_delay = 0.0
                path_bandwidths = []
                
                for j in range(len(path) - 1):
                    u, v = path[j], path[j + 1]
                    edge_attr = self._get_edge_attr(u, v)
                    path_bandwidths.append(edge_attr[0])
                    path_delay += edge_attr[1]
                
                path_min_bw = min(path_bandwidths) if path_bandwidths else 0.0
                
                paths.append({
                    "delay": path_delay,
                    "bandwidth": path_min_bw,
                    "hops": len(path) - 1
                })
                
                total_delay += path_delay
                min_bandwidth = min(min_bandwidth, path_min_bw)
                
            except nx.NetworkXNoPath:
                continue
        
        total_cpu_used = sum(self.initial_node_resources[node, 0] - self.current_node_resources[node, 0] 
                            for node in self.used_nodes)
        total_cpu_available = sum(self.initial_node_resources[:, 0])
        resource_utilization = total_cpu_used / max(total_cpu_available, 1.0)
        
        return {
            'paths': paths,
            'total_delay': total_delay,
            'avg_delay': total_delay / max(len(paths), 1) if total_delay != float('inf') else float('inf'),
            'min_bandwidth': min_bandwidth,
            'resource_utilization': resource_utilization,
            'avg_jitter': 0.0,
            'avg_loss': 0.0
        }
    
    # 保持原有接口兼容性的其他方法
    def _update_node_resources(self, node_id: int, vnf_req: Dict):
        """更新节点资源"""
        self.current_node_resources[node_id, 0] -= vnf_req['cpu']
        if len(self.current_node_resources[node_id]) > 1:
            self.current_node_resources[node_id, 1] -= vnf_req['memory']
        self.current_node_resources[node_id] = np.maximum(self.current_node_resources[node_id], 0.0)
    
    def _calculate_constraint_penalty(self, reason: str) -> float:
        """计算约束违反的惩罚"""
        penalty_map = {
            'node_occupied': -5.0,
            'insufficient_cpu': -8.0,
            'insufficient_memory': -6.0,
            'insufficient_bandwidth': -4.0
        }
        return penalty_map.get(reason, -3.0)
    
    def _compute_reward(self, info: Dict) -> float:
        """计算奖励"""
        try:
            if 'total_vnfs' not in info:
                info['total_vnfs'] = len(self.service_chain)
            if 'deployed_vnfs' not in info:
                info['deployed_vnfs'] = len(self.embedding_map)
            if 'vnf_requests' not in info:
                info['vnf_requests'] = self.vnf_requirements
            
            reward = compute_reward(info, self.reward_config)
            
            if reward is None:
                reward = self.reward_config.get('base_reward', 10.0)
            
            return float(reward)
            
        except Exception as e:
            return self.reward_config.get('base_reward', 10.0)
    
    def _get_edge_attr(self, u: int, v: int) -> np.ndarray:
        """获取边属性"""
        if (u, v) in self.edge_index_map:
            edge_idx = self.edge_index_map[(u, v)]
        elif (v, u) in self.edge_index_map:
            edge_idx = self.edge_index_map[(v, u)]
        else:
            return np.array([100.0, 1.0, 0.1, 0.01])
        return self.edge_features[edge_idx]
    
    def _handle_invalid_action(self, reason: str) -> Tuple[Data, float, bool, Dict]:
        """处理无效动作"""
        return self._get_enhanced_state(), -10.0, True, {
            'success': False,
            'error': reason,
            'step': self.step_count
        }
    
    def _handle_completion(self) -> Tuple[Data, float, bool, Dict]:
        """处理已完成的情况"""
        return self._get_enhanced_state(), 0.0, True, {
            'success': True,
            'already_completed': True,
            'step': self.step_count
        }
    
    def get_valid_actions(self) -> List[int]:
        """获取有效动作（兼容性接口）"""
        return self.get_enhanced_valid_actions()
    
    def render(self, mode='human') -> None:
        """可视化当前环境状态"""
        display_name = getattr(self, 'scenario_display_name', self.current_scenario_name)
        
        print(f"\n{'='*60}")
        print(f"📊 增强VNF嵌入环境状态 (步数: {self.step_count}, 场景: {display_name})")
        print(f"{'='*60}")
        
        print(f"🔗 服务链: {' -> '.join(self.service_chain)}")
        print(f"📍 当前VNF: {self.current_vnf_index}/{len(self.service_chain)}")
        
        valid_actions = self.get_enhanced_valid_actions()
        print(f"✅ 增强有效动作数: {len(valid_actions)}/{self.action_dim}")
        
        if hasattr(self, 'enhanced_state_cache') and self.enhanced_state_cache:
            enhanced_info = self.enhanced_state_cache
            congestion_state = enhanced_info['congestion_state']
            print(f"🚦 网络状态:")
            print(f"   热点节点: {len(congestion_state['hotspots'])}")
            print(f"   瓶颈边: {len(congestion_state['bottlenecks'])}")
    
    def get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        base_info = {
            'service_chain_length': len(self.service_chain),
            'current_vnf_index': self.current_vnf_index,
            'embedding_progress': self.current_vnf_index / len(self.service_chain),
            'used_nodes': list(self.used_nodes),
            'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
            'step_count': self.step_count,
            'valid_actions_count': len(self.get_enhanced_valid_actions()),
            'current_scenario': self.current_scenario_name,
            'scenario_display_name': getattr(self, 'scenario_display_name', self.current_scenario_name),
            'enhanced_features_enabled': True
        }
        
        return base_info
    
    def seed(self, seed: int = None) -> List[int]:
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            return [seed]
        return []
    
    def close(self):
        """关闭环境"""
        pass


# 测试函数
def test_enhanced_vnf_env():
    """测试增强VNF环境"""
    print("🧪 测试增强VNF嵌入环境...")
    
    # 创建测试图
    import networkx as nx
    G = nx.erdos_renyi_graph(10, 0.3)
    
    # 创建测试特征
    node_features = np.random.rand(10, 4)
    edge_features = np.random.rand(len(G.edges()), 4)
    
    # 测试配置
    reward_config = {
        'base_reward': 10.0,
        'penalty': 20.0,
        'completion_bonus': 15.0
    }
    
    config = {
        'vnf_requirements': {
            'cpu_min': 0.1, 'cpu_max': 0.3,
            'memory_min': 0.1, 'memory_max': 0.3,
            'bandwidth_min': 5.0, 'bandwidth_max': 15.0,
            'chain_length_range': (3, 5)
        },
        'train': {'max_episode_steps': 20}
    }
    
    # 创建增强环境
    env = EnhancedVNFEmbeddingEnv(
        graph=G,
        node_features=node_features,
        edge_features=edge_features,
        reward_config=reward_config,
        config=config
    )
    
    print("✅ 增强VNF环境创建成功")
    
    # 测试重置
    state = env.reset()
    print(f"✅ 环境重置: 状态维度 {state.x.shape}")
    
    # 测试增强功能
    valid_actions = env.get_enhanced_valid_actions()
    print(f"✅ 增强有效动作: {len(valid_actions)} 个")
    
    # 测试步骤
    if valid_actions:
        action = valid_actions[0]
        next_state, reward, done, info = env.step(action)
        print(f"✅ 步骤测试: 奖励={reward:.2f}, 完成={done}")
        print(f"   增强特征: {info.get('enhanced_features_used', False)}")
    
    print("🎉 增强VNF环境测试完成!")


if __name__ == "__main__":
    test_enhanced_vnf_env()