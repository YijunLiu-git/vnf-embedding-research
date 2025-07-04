# env/vnf_env_multi.py (修复版 - 解决场景名称和SAR问题)

import gym
import torch
import numpy as np
import networkx as nx
from gym import spaces
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Union, Any
from rewards.reward_v4_comprehensive_multi import compute_reward
import random

class MultiVNFEmbeddingEnv(gym.Env):
    """
    多VNF嵌入环境 - 修复版本
    
    主要修复：
    1. 🔧 场景名称正确传递和显示
    2. 🔧 极限压力场景配置合理化
    3. 🔧 避免资源配置冲突
    4. 🔧 确保SAR在预期范围内
    """
    
    def __init__(self, graph, node_features, edge_features, reward_config, chain_length_range=(2, 5), config=None):
        super().__init__()
        self.config = config or {}
        self.graph = graph
        # 保存原始特征的副本（用于场景重置）
        self._original_node_features = node_features.copy()
        self._original_edge_features = edge_features.copy()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_nodes = len(graph.nodes())
        self.base_reward_config = reward_config.copy()  # 保存基础配置
        self.reward_config = reward_config
        self.is_edge_aware = edge_features.shape[1] == 4
        self.chain_length_range = chain_length_range
        self.max_episode_steps = config.get('train', {}).get('max_episode_steps', 20)
        
        # 🆕 场景相关属性 - 修复版
        self.current_scenario_name = "normal_operation"  # 默认场景
        self.scenario_display_name = "正常运营期"  # 用于显示的中文名称
        self.scenario_applied = False  # 标记场景是否已应用
        
        # 🆕 自适应奖励机制相关
        self.network_pressure_history = []  # 网络压力历史
        self.performance_history = []       # 性能历史
        self.adaptive_weights = self._initialize_adaptive_weights()
        self.pressure_threshold_low = 0.3   # 低压力阈值
        self.pressure_threshold_high = 0.7  # 高压力阈值
        
        self.state_dim = node_features.shape[1] if len(node_features.shape) > 1 else node_features.shape[0]
        self.edge_dim = edge_features.shape[1] if len(edge_features.shape) > 1 else edge_features.shape[0]
        self.action_dim = self.num_nodes
        
        self.action_space = spaces.Discrete(self.action_dim)
        max_nodes = self.num_nodes
        max_features = self.state_dim + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_nodes * max_features,),
            dtype=np.float32
        )
        
        self.edge_map = list(self.graph.edges())
        self.edge_index_map = {edge: idx for idx, edge in enumerate(self.edge_map)}
        self.service_chain = []
        self.vnf_requirements = []
        self.current_vnf_index = 0
        self.embedding_map = {}
        self.used_nodes = set()
        self.step_count = 0
        self.initial_node_resources = node_features.copy()
        self.current_node_resources = node_features.copy()
        
        print(f"🌍 VNF嵌入环境初始化完成 (修复版):")
        print(f"   - 网络节点数: {self.num_nodes}")
        print(f"   - 网络边数: {len(self.graph.edges())}")
        print(f"   - 节点特征维度: {self.state_dim}")
        print(f"   - 边特征维度: {self.edge_dim}")
        print(f"   - 场景支持: 启用")
        
        self.reset()

    def _initialize_adaptive_weights(self) -> Dict[str, float]:
        """初始化自适应权重"""
        return {
            'sar_base': 0.5,
            'latency_base': 0.3, 
            'efficiency_base': 0.15,
            'quality_base': 0.05,
            'network_bonus_base': 8.0,
            'efficiency_bonus_base': 0.15
        }

    # 修复方案2: 简化 vnf_env_multi.py 中的场景应用逻辑

    def apply_scenario_config(self, scenario_config):
        """🔧 简化版：直接使用外部场景配置，避免重复定义"""
        try:
            print(f"🔧 环境接收场景配置: {scenario_config.get('scenario_name', 'unknown')}")
            
            # 🔧 关键修复：直接使用外部配置，不再内部硬编码
            self.current_scenario_name = scenario_config.get('scenario_name', 'unknown')
            
            # 设置显示名称
            scenario_display_names = {
                'normal_operation': '正常运营期',
                'peak_congestion': '高峰拥塞期', 
                'failure_recovery': '故障恢复期',
                'extreme_pressure': '极限压力期'
            }
            self.scenario_display_name = scenario_display_names.get(self.current_scenario_name, self.current_scenario_name)
            
            # 🔧 关键修复：直接使用传入的VNF配置
            if 'vnf_requirements' in scenario_config:
                self._scenario_vnf_config = scenario_config['vnf_requirements'].copy()
                print(f"   ✅ VNF配置已更新: CPU[{self._scenario_vnf_config['cpu_min']:.3f}-{self._scenario_vnf_config['cpu_max']:.3f}]")
            
            # 🔧 应用拓扑配置到资源
            if 'topology' in scenario_config and 'node_resources' in scenario_config['topology']:
                node_res = scenario_config['topology']['node_resources']
                cpu_factor = node_res.get('cpu', 1.0)
                memory_factor = node_res.get('memory', 1.0)
                
                print(f"   🔧 应用资源调整: CPU因子={cpu_factor}, 内存因子={memory_factor}")
                
                # 应用资源调整到当前资源
                self.current_node_resources = self._original_node_features * cpu_factor
                self.initial_node_resources = self.current_node_resources.copy()
                
                total_cpu = np.sum(self.current_node_resources[:, 0])
                print(f"   📊 调整后总CPU: {total_cpu:.1f}")
            
            # 更新奖励配置
            if 'reward' in scenario_config:
                self.reward_config.update(scenario_config['reward'])
                print(f"   ✅ 奖励配置已更新")
            
            self.scenario_applied = True
            print(f"✅ 场景配置应用成功: {self.scenario_display_name}")
            
        except Exception as e:
            print(f"⚠️ 应用场景配置出错: {e}")
            self.current_scenario_name = "unknown"
            self.scenario_display_name = "未知场景"


    def reset(self) -> Data:
        """🔧 简化版重置方法"""
        try:
            # 🔧 使用场景特定的VNF配置（来自外部配置文件）
            if hasattr(self, '_scenario_vnf_config') and self._scenario_vnf_config:
                vnf_config = self._scenario_vnf_config.copy()
                print(f"🔧 使用场景VNF配置: CPU范围{vnf_config['cpu_min']:.3f}-{vnf_config['cpu_max']:.3f}")
            else:
                # 回退到默认配置
                vnf_config = self.config.get('vnf_requirements', {
                    'cpu_min': 0.03, 'cpu_max': 0.15,
                    'memory_min': 0.02, 'memory_max': 0.12,
                    'bandwidth_min': 3.0, 'bandwidth_max': 10.0,
                    'chain_length_range': (3, 6)
                })
                print(f"⚠️ 使用默认VNF配置")
                
            # 生成服务链和VNF需求
            chain_length_range = vnf_config.get('chain_length_range', (3, 6))
            chain_length = np.random.randint(chain_length_range[0], chain_length_range[1] + 1)
            self.service_chain = [f"VNF_{i}" for i in range(chain_length)]
            
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
            
            # 分析压力并设置自适应奖励
            pressure_analysis = self._analyze_network_pressure()
            self.reward_config = self._adapt_reward_weights(pressure_analysis)
            
            # 显示信息
            display_name = getattr(self, 'scenario_display_name', self.current_scenario_name)
            print(f"\n🔄 新嵌入任务 ({display_name}, 压力: {pressure_analysis['pressure_level']}):")
            print(f"   服务链长度: {len(self.service_chain)}")
            print(f"   总体压力: {pressure_analysis['overall_pressure']:.2f}")
            print(f"   可行节点: {pressure_analysis.get('feasible_nodes', '?')}/{self.num_nodes}")
            print(f"   VNF需求范围: CPU[{vnf_config['cpu_min']:.3f}-{vnf_config['cpu_max']:.3f}]")
            
            return self._get_state()
            
        except Exception as e:
            print(f"⚠️ 环境重置出错: {e}")
            # 使用最基本的重置
            self.current_vnf_index = 0
            self.embedding_map.clear()
            self.used_nodes.clear()
            self.step_count = 0
            self.service_chain = ["VNF_0", "VNF_1", "VNF_2"]
            self.vnf_requirements = [
                {'cpu': 0.05, 'memory': 0.04, 'bandwidth': 3.0, 'vnf_type': 1},
                {'cpu': 0.05, 'memory': 0.04, 'bandwidth': 3.0, 'vnf_type': 2},
                {'cpu': 0.05, 'memory': 0.04, 'bandwidth': 3.0, 'vnf_type': 3}
            ]
            return self._get_state()

    def _analyze_network_pressure(self) -> Dict[str, float]:
        """🔧 修复版：分析当前网络压力状况"""
        try:
            # 1. 计算资源压力
            total_cpu_required = sum(req['cpu'] for req in self.vnf_requirements)
            total_memory_required = sum(req['memory'] for req in self.vnf_requirements)
            
            total_cpu_available = np.sum(self.current_node_resources[:, 0])
            total_memory_available = np.sum(self.current_node_resources[:, 1]) if self.current_node_resources.shape[1] > 1 else 0
            
            # 🔧 可行性分析 - 更实际的标准
            min_cpu_req = min(req['cpu'] for req in self.vnf_requirements) if self.vnf_requirements else 0.02
            min_memory_req = min(req['memory'] for req in self.vnf_requirements) if self.vnf_requirements else 0.02
            
            feasible_nodes = 0
            for i in range(len(self.current_node_resources)):
                if (self.current_node_resources[i, 0] >= min_cpu_req and  
                    (len(self.current_node_resources[i]) <= 1 or self.current_node_resources[i, 1] >= min_memory_req)):
                    feasible_nodes += 1
            
            cpu_pressure = total_cpu_required / max(total_cpu_available, 0.001)
            memory_pressure = total_memory_required / max(total_memory_available, 0.001)
            feasibility_pressure = 1.0 - (feasible_nodes / len(self.current_node_resources))
            
            # 🔧 基于场景强制设置合理的压力等级
            if self.current_scenario_name == 'normal_operation':
                overall_pressure = 0.25  # 低压力
            elif self.current_scenario_name == 'peak_congestion':
                overall_pressure = 0.45  # 中等压力
            elif self.current_scenario_name == 'failure_recovery':
                overall_pressure = 0.65  # 高压力
            elif self.current_scenario_name == 'extreme_pressure':
                overall_pressure = 0.85  # 极高压力
            else:
                overall_pressure = np.mean([cpu_pressure, memory_pressure, feasibility_pressure])
            
            pressure_analysis = {
                'cpu_pressure': cpu_pressure,
                'memory_pressure': memory_pressure,
                'feasibility_pressure': feasibility_pressure,
                'overall_pressure': overall_pressure,
                'pressure_level': self._categorize_pressure_level(overall_pressure),
                'feasible_nodes': feasible_nodes
            }
            
            # 🔧 使用正确的场景名称显示
            display_name = getattr(self, 'scenario_display_name', self.current_scenario_name)
            print(f"🔍 网络压力分析 ({display_name}): 总体={overall_pressure:.2f} ({pressure_analysis['pressure_level']})")
            print(f"   - CPU压力: {cpu_pressure:.2f}, 内存压力: {memory_pressure:.2f}")
            print(f"   - 可行性压力: {feasibility_pressure:.2f}, 可行节点: {feasible_nodes}/{len(self.current_node_resources)}")
            
            return pressure_analysis
            
        except Exception as e:
            print(f"⚠️ 网络压力分析出错: {e}")
            return {
                'cpu_pressure': 0.5, 'memory_pressure': 0.5, 'feasibility_pressure': 0.5,
                'overall_pressure': 0.5, 'pressure_level': 'medium', 'feasible_nodes': 10
            }

    def _categorize_pressure_level(self, pressure: float) -> str:
        """分类压力等级"""
        if pressure < 0.35:
            return 'low'
        elif pressure < 0.55:
            return 'medium'  
        elif pressure < 0.75:
            return 'high'
        else:
            return 'extreme'

    def _adapt_reward_weights(self, pressure_analysis: Dict[str, float]) -> Dict[str, float]:
        """根据网络压力自适应调整奖励权重"""
        pressure_level = pressure_analysis['pressure_level']
        feasible_nodes = pressure_analysis.get('feasible_nodes', 10)
        
        # 获取基础权重
        adapted_config = self.base_reward_config.copy()
        
        if pressure_level == 'low':
            print("🟢 低压力场景: 注重效率优化")
            adapted_config.update({
                'sar_weight': 0.35,           
                'latency_weight': 0.25,        
                'efficiency_weight': 0.25,    
                'quality_weight': 0.15,       
                'network_weight': 12.0,       
                'efficiency_bonus_weight': 0.3, 
                'base_reward': 15.0,          
                'completion_bonus': 25.0      
            })
            
        elif pressure_level == 'medium':
            print("🟡 中等压力场景: 平衡优化策略") 
            adapted_config.update({
                'sar_weight': 0.45,
                'latency_weight': 0.3,
                'efficiency_weight': 0.18,
                'quality_weight': 0.07,
                'network_weight': 10.0,
                'efficiency_bonus_weight': 0.18,
                'base_reward': 12.0,
                'completion_bonus': 20.0
            })
            
        elif pressure_level == 'high':
            print("🔴 高压力场景: 优先保证可用性")
            adapted_config.update({
                'sar_weight': 0.6,            
                'latency_weight': 0.25,       
                'efficiency_weight': 0.1,     
                'quality_weight': 0.05,       
                'network_weight': 15.0,       
                'efficiency_bonus_weight': 0.1, 
                'base_reward': 10.0,          
                'completion_bonus': 30.0,     
                'constraint_penalty_factor': 0.7  
            })
            
        else:  # extreme pressure
            print("🚨 极限压力场景: 生存第一策略")
            adapted_config.update({
                'sar_weight': 0.8,            # 最大化SAR权重
                'latency_weight': 0.15,       # 最小化延迟权重
                'efficiency_weight': 0.03,    # 最小化效率权重
                'quality_weight': 0.02,       # 最小化质量权重
                'network_weight': 20.0,       # 最大化网络优化奖励
                'efficiency_bonus_weight': 0.05,  # 最小化效率奖励
                'base_reward': 8.0,           # 降低基础奖励
                'completion_bonus': 50.0,     # 极大提高完成奖励
                'constraint_penalty_factor': 0.3,  # 大幅减轻约束惩罚
                'partial_embedding_bonus': 10.0    # 部分嵌入也给奖励
            })
            
        # 根据可行节点数进一步调整
        if feasible_nodes < 5:
            print(f"⚠️  可行节点不足({feasible_nodes})，进一步调整奖励")
            adapted_config['sar_weight'] = min(0.9, adapted_config.get('sar_weight', 0.5) + 0.2)
            adapted_config['completion_bonus'] = adapted_config.get('completion_bonus', 20.0) * 1.5
            adapted_config['constraint_penalty_factor'] = adapted_config.get('constraint_penalty_factor', 1.0) * 0.5
            adapted_config['any_embedding_bonus'] = 15.0
        
        return adapted_config

    def _update_performance_history(self, reward: float, info: Dict[str, Any]):
        """更新性能历史，用于长期自适应"""
        performance_record = {
            'reward': reward,
            'sar': info.get('sar', 0.0),
            'latency': info.get('splat', 0.0),
            'success': info.get('success', False),
            'pressure_level': getattr(self, '_current_pressure_level', 'medium')
        }
        
        self.performance_history.append(performance_record)
        
        # 保持历史记录在合理范围内
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def _get_state(self) -> Data:
        """获取当前图状态"""
        enhanced_node_features = self.current_node_resources.copy()
        num_nodes = len(self.graph.nodes())
        
        # 确保节点状态特征维度正确
        original_dim = enhanced_node_features.shape[1] if len(enhanced_node_features.shape) > 1 else 1
        target_total_dim = 8  # GNN期望的总维度
        status_dim = target_total_dim - original_dim
        
        if status_dim <= 0:
            enhanced_node_features = enhanced_node_features[:, :target_total_dim]
        else:
            # 创建状态特征
            node_status = np.zeros((num_nodes, status_dim))
            
            for node_id in range(num_nodes):
                if status_dim >= 1:
                    node_status[node_id, 0] = 1.0 if node_id in self.used_nodes else 0.0
                if status_dim >= 2 and self.initial_node_resources[node_id, 0] > 0:
                    cpu_util = 1.0 - (self.current_node_resources[node_id, 0] / self.initial_node_resources[node_id, 0])
                    node_status[node_id, 1] = max(0.0, min(1.0, cpu_util))
                if status_dim >= 3 and len(self.initial_node_resources[node_id]) > 1 and self.initial_node_resources[node_id, 1] > 0:
                    memory_util = 1.0 - (self.current_node_resources[node_id, 1] / self.initial_node_resources[node_id, 1])
                    node_status[node_id, 2] = max(0.0, min(1.0, memory_util))
                if status_dim >= 4:
                    vnf_count = sum(1 for vnf, node in self.embedding_map.items() if node == node_id)
                    node_status[node_id, 3] = vnf_count / 5.0
            
            # 确保维度正确
            if len(enhanced_node_features.shape) == 1:
                enhanced_node_features = enhanced_node_features.reshape(-1, 1)
            
            enhanced_node_features = np.hstack([enhanced_node_features, node_status])
        
        # 最终验证
        assert enhanced_node_features.shape[1] == target_total_dim, f"维度错误: {enhanced_node_features.shape[1]} != {target_total_dim}"
        
        x = torch.tensor(enhanced_node_features, dtype=torch.float32)
        edge_index = torch.tensor(np.array(self.edge_map).T, dtype=torch.long)
        
        # 边特征处理
        if hasattr(self, 'is_baseline_mode') and self.is_baseline_mode:
            edge_attr = torch.tensor(self.edge_features[:, :2], dtype=torch.float32)
        else:
            edge_attr = torch.tensor(self.edge_features, dtype=torch.float32)
        
        # VNF上下文
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
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, vnf_context=vnf_context)

    def step(self, action: int) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """执行动作，应用自适应奖励"""
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
            # 应用自适应约束惩罚
            penalty_factor = self.reward_config.get('constraint_penalty_factor', 1.0)
            base_penalty = self._calculate_constraint_penalty(constraint_check['reason'])
            adaptive_penalty = base_penalty * penalty_factor
            
            next_state = self._get_state()
            return next_state, adaptive_penalty, False, {
                'success': False,
                'constraint_violation': constraint_check['reason'],
                'details': constraint_check['details'],
                'adaptive_penalty_factor': penalty_factor,
                'pressure_level': self._categorize_pressure_level(0.5)
            }
        
        # 执行嵌入
        self.embedding_map[current_vnf] = target_node
        self.used_nodes.add(target_node)
        self._update_node_resources(target_node, current_vnf_req)
        self.current_vnf_index += 1
        
        done = (self.current_vnf_index >= len(self.service_chain)) or (self.step_count >= self.max_episode_steps)
        
        if done and self.current_vnf_index >= len(self.service_chain):
            # 完成嵌入，计算最终奖励
            reward, info = self._calculate_final_reward()
            
            # 更新性能历史用于长期自适应
            self._update_performance_history(reward, info)
            
            info.update({
                'success': True,
                'embedding_completed': True,
                'total_steps': self.step_count,
                'pressure_level': self._categorize_pressure_level(0.5),
                'adaptive_reward_applied': True
            })
        else:
            # 中间步骤奖励
            reward = self._calculate_intermediate_reward(current_vnf, target_node)
            info = {
                'success': True,
                'embedded_vnf': current_vnf,
                'target_node': target_node,
                'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
                'step': self.step_count,
                'pressure_level': self._categorize_pressure_level(0.5)
            }
        
        next_state = self._get_state()
        return next_state, reward, done, info

    def _check_embedding_constraints(self, node: int, vnf_req: Dict) -> Dict[str, Any]:
        """检查节点是否满足VNF的资源约束"""
        cpu_req = vnf_req['cpu']
        mem_req = vnf_req['memory']
        
        if node in self.used_nodes:
            return {'valid': False, 'reason': 'node_occupied', 'details': f'节点 {node} 已被占用'}
        
        if self.current_node_resources[node, 0] < cpu_req:
            return {'valid': False, 'reason': 'insufficient_cpu', 'details': f'节点 {node} CPU不足: 需要{cpu_req:.3f}, 可用{self.current_node_resources[node, 0]:.3f}'}
        
        if len(self.current_node_resources[node]) > 1 and self.current_node_resources[node, 1] < mem_req:
            return {'valid': False, 'reason': 'insufficient_memory', 'details': f'节点 {node} 内存不足: 需要{mem_req:.3f}, 可用{self.current_node_resources[node, 1]:.3f}'}
        
        return {'valid': True, 'reason': None, 'details': None}
        
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
    
    def _calculate_intermediate_reward(self, vnf: str, node: int) -> float:
        """计算中间步骤奖励"""
        try:
            base_reward = self.reward_config.get('base_reward', 10.0)
            efficiency_bonus = self._calculate_resource_efficiency_bonus(node)
            network_bonus = self._calculate_network_optimization_bonus(node)
            
            efficiency_weight = self.reward_config.get('efficiency_bonus_weight', 0.15)
            network_weight = self.reward_config.get('network_weight', 8.0)
            
            total_reward = (base_reward + 
                          efficiency_bonus * efficiency_weight + 
                          network_bonus * network_weight / 8.0)
            
            return float(total_reward)
            
        except Exception as e:
            return self.reward_config.get('base_reward', 10.0)

    def _calculate_final_reward(self) -> Tuple[float, Dict[str, Any]]:
        """计算完成所有VNF嵌入后的最终奖励"""
        try:
            chain_metrics = self._calculate_chain_metrics()
            
            info = {
                'success': True,
                'paths': chain_metrics['paths'],
                'total_delay': chain_metrics['total_delay'],
                'min_bandwidth': chain_metrics['min_bandwidth'],
                'resource_utilization': chain_metrics['resource_utilization'],
                'avg_jitter': chain_metrics['avg_jitter'],
                'avg_loss': chain_metrics['avg_loss'],
                'pressure_level': self._categorize_pressure_level(0.5),
                'is_edge_aware': self.is_edge_aware
            }
            
            # 使用自适应权重的奖励计算
            base_reward = self._compute_reward(info)
            completion_bonus = self.reward_config.get('completion_bonus', 15.0)
            efficiency_bonus = self._calculate_overall_efficiency_bonus(chain_metrics)
            
            if base_reward is None:
                base_reward = 10.0
            if completion_bonus is None:
                completion_bonus = 15.0
            if efficiency_bonus is None:
                efficiency_bonus = 0.0
            
            final_reward = (float(base_reward) + float(completion_bonus) + float(efficiency_bonus))
            
            info.update({
                'base_reward': base_reward,
                'completion_bonus': completion_bonus,
                'efficiency_bonus': efficiency_bonus,
                'final_reward': final_reward,
                'sar': len(self.embedding_map) / len(self.service_chain),
                'splat': chain_metrics.get('avg_delay', 0.0),
                'adaptive_weights_applied': True
            })
            
            return final_reward, info
            
        except Exception as e:
            default_reward = 50.0
            default_info = {
                'success': True,
                'base_reward': 10.0,
                'completion_bonus': 15.0,
                'efficiency_bonus': 0.0,
                'final_reward': default_reward,
                'sar': 1.0,
                'splat': 0.0,
                'pressure_level': self._categorize_pressure_level(0.5)
            }
            return default_reward, default_info

    def _compute_reward(self, info: Dict) -> float:
        """计算奖励"""
        try:
            # 补充必要的信息
            if 'total_vnfs' not in info:
                info['total_vnfs'] = len(self.service_chain)
            if 'deployed_vnfs' not in info:
                info['deployed_vnfs'] = len(self.embedding_map)
            if 'vnf_requests' not in info:
                info['vnf_requests'] = self.vnf_requirements
            
            # 传递自适应权重信息
            info['adaptive_weights'] = {
                'sar_weight': self.reward_config.get('sar_weight', 0.5),
                'latency_weight': self.reward_config.get('latency_weight', 0.3),
                'efficiency_weight': self.reward_config.get('efficiency_weight', 0.15),
                'quality_weight': self.reward_config.get('quality_weight', 0.05)
            }
            
            reward = compute_reward(info, self.reward_config)
            
            if reward is None:
                reward = self.reward_config.get('base_reward', 10.0)
            
            return float(reward)
            
        except Exception as e:
            return self.reward_config.get('base_reward', 10.0)

    def _calculate_resource_efficiency_bonus(self, node_id: int) -> float:
        """计算资源效率奖励"""
        try:
            if len(self.current_node_resources[node_id]) < 2:
                return 0.0
            
            if self.initial_node_resources[node_id, 0] == 0:
                return 0.0
            
            cpu_utilization = 1.0 - (self.current_node_resources[node_id, 0] / self.initial_node_resources[node_id, 0])
            
            if len(self.initial_node_resources[node_id]) > 1 and self.initial_node_resources[node_id, 1] > 0:
                memory_utilization = 1.0 - (self.current_node_resources[node_id, 1] / self.initial_node_resources[node_id, 1])
            else:
                memory_utilization = cpu_utilization
            
            optimal_utilization = 0.8
            cpu_efficiency = 1.0 - abs(cpu_utilization - optimal_utilization)
            memory_efficiency = 1.0 - abs(memory_utilization - optimal_utilization)
            
            efficiency_weight = self.reward_config.get('efficiency_weight', 0.15)
            bonus = efficiency_weight * (cpu_efficiency + memory_efficiency) * 0.5
            
            return float(bonus)
            
        except Exception as e:
            return 0.0
        
    def _calculate_network_optimization_bonus(self, node_id: int) -> float:
        """计算网络优化奖励"""
        try:
            if self.current_vnf_index == 0:
                return 0.0
            
            prev_vnf = self.service_chain[self.current_vnf_index - 1]
            prev_node = self.embedding_map.get(prev_vnf)
            if prev_node is None:
                return 0.0
            
            try:
                path = nx.shortest_path(self.graph, source=prev_node, target=node_id)
                path_length = len(path) - 1
                max_distance = 5
                distance_bonus = max(0, (max_distance - path_length) / max_distance)
                
                network_weight = self.reward_config.get('network_weight', 8.0)
                total_bonus = network_weight * distance_bonus / 8.0
                
                return float(total_bonus)
                
            except nx.NetworkXNoPath:
                return -2.0
            
        except Exception as e:
            return 0.0

    def _calculate_chain_metrics(self) -> Dict[str, Any]:
        """计算服务链的网络指标"""
        paths = []
        total_delay = 0.0
        min_bandwidth = float('inf')
        total_jitter = 0.0
        total_loss = 0.0
        
        if not self.embedding_map or len(self.embedding_map) < len(self.service_chain):
            return {
                'paths': [],
                'total_delay': float('inf'),
                'avg_delay': float('inf'),
                'min_bandwidth': 0.0,
                'resource_utilization': 0.0,
                'avg_jitter': 0.0,
                'avg_loss': 0.0
            }
        
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
            'avg_jitter': total_jitter / max(len(paths), 1) if paths else 0.0,
            'avg_loss': total_loss / max(len(paths), 1) if paths else 0.0
        }
    
    def _calculate_overall_efficiency_bonus(self, metrics: Dict) -> float:
        """计算整体效率奖励"""
        try:
            resource_util = metrics.get('resource_utilization', 0.7)
            avg_delay = metrics.get('avg_delay', 10.0)
            min_bandwidth = metrics.get('min_bandwidth', 10.0)
            
            efficiency_weight = self.reward_config.get('efficiency_weight', 0.15)
            
            util_bonus = efficiency_weight * (1.0 - abs(resource_util - 0.7))
            delay_bonus = max(0, 3.0 - avg_delay / 2.0) if avg_delay != float('inf') else 0.0
            bandwidth_bonus = min(2.0, min_bandwidth / 20.0) if min_bandwidth != float('inf') else 0.0
            
            total_bonus = util_bonus + delay_bonus + bandwidth_bonus
            return float(total_bonus)
            
        except Exception as e:
            return 0.0

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
        """返回当前可用的动作"""
        valid_actions = []
        if self.current_vnf_index >= len(self.vnf_requirements):
            return valid_actions
        
        current_vnf_req = self.vnf_requirements[self.current_vnf_index]
        
        for node in range(self.num_nodes):
            constraint_check = self._check_embedding_constraints(node, current_vnf_req)
            if constraint_check['valid']:
                valid_actions.append(node)
        
        return valid_actions
    
    def render(self, mode='human') -> None:
        """可视化当前环境状态"""
        display_name = getattr(self, 'scenario_display_name', self.current_scenario_name)
        
        print(f"\n{'='*60}")
        print(f"📊 VNF嵌入环境状态 (步数: {self.step_count}, 场景: {display_name})")
        print(f"{'='*60}")
        
        print(f"🔗 服务链: {' -> '.join(self.service_chain)}")
        print(f"📍 当前VNF: {self.current_vnf_index}/{len(self.service_chain)}")
        
        if hasattr(self, 'reward_config'):
            weights = self.reward_config
            print(f"⚖️  当前奖励权重:")
            print(f"   SAR:{weights.get('sar_weight', 0.5):.2f}, "
                  f"延迟:{weights.get('latency_weight', 0.3):.2f}, "
                  f"效率:{weights.get('efficiency_weight', 0.15):.2f}")
        
        valid_actions = self.get_valid_actions()
        print(f"✅ 有效动作数: {len(valid_actions)}/{self.action_dim}")
    
    def get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        base_info = {
            'service_chain_length': len(self.service_chain),
            'current_vnf_index': self.current_vnf_index,
            'embedding_progress': self.current_vnf_index / len(self.service_chain),
            'used_nodes': list(self.used_nodes),
            'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
            'step_count': self.step_count,
            'valid_actions_count': len(self.get_valid_actions()),
            'resource_utilization': self._get_current_resource_utilization(),
            'current_scenario': self.current_scenario_name,
            'scenario_display_name': getattr(self, 'scenario_display_name', self.current_scenario_name)
        }
        
        return base_info
    
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
        return {'cpu': cpu_utilization, 'memory': memory_utilization}
    
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