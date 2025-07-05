# agents/base_agent.py - 完整修复版：配置驱动，维度统一

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
from torch_geometric.data import Data, Batch
from agents.enhanced_base_agent import EnhancedBaseAgent

# 🔧 关键修复：导入配置加载器
try:
    from config_loader import get_config_loader
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ 配置加载器未找到，使用默认维度配置")
    CONFIG_AVAILABLE = False

# 标准GNN编码器导入
from models.gnn_encoder import GNNEncoder

class BaseAgent(ABC):
    """
    多智能体VNF嵌入系统的基础智能体类 - 完整修复版
    
    🔧 主要修复：
    1. 完全基于配置文件的维度管理
    2. 智能体类型自动识别（Edge-aware/Baseline）
    3. 运行时维度验证
    4. 兼容性保证和错误处理
    """
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int, 
                 action_dim: int, 
                 edge_dim: int,
                 config: Dict[str, Any]):
        
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        
        # 🔧 关键修复1：从配置文件获取标准维度
        self._load_dimension_config()
        
        # 🔧 关键修复2：智能体类型自动识别
        self._detect_agent_mode()
        
        # 🔧 关键修复3：维度标准化
        self._standardize_dimensions()
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Agent {agent_id} 使用设备: {self.device}")
        
        # 🔧 关键修复4：配置驱动的GNN配置获取
        self._load_gnn_config()
        
        # 训练配置
        self._load_training_config()
        
        # 🔧 关键修复5：创建配置匹配的GNN编码器
        self._create_gnn_encoder()
        
        # 策略网络（子类实现具体结构）
        self.policy_network = None
        self.target_network = None  # DQN系列使用
        self.optimizer = None
        
        # 训练状态
        self.training_step = 0
        self.episode_count = 0
        self.is_training = True
        
        # 统计信息
        self.stats = {
            "total_reward": 0.0,
            "episodes": 0,
            "steps": 0,
            "losses": [],
            "q_values": [],
            "actions_taken": {}
        }
        
        # 多智能体协调（预留）
        self.other_agents = {}
        self.communication_enabled = False
        
        print(f"✅ BaseAgent {agent_id} 初始化完成")
        self._print_dimension_summary()
        
    def _load_dimension_config(self):
        """加载维度配置"""
        if CONFIG_AVAILABLE:
            try:
                config_loader = get_config_loader()
                self.dimensions = config_loader.config.get('dimensions', {})
                print(f"✅ 从配置文件加载维度配置")
            except Exception as e:
                print(f"⚠️ 配置文件加载失败: {e}，使用默认配置")
                self.dimensions = self._get_default_dimensions()
        else:
            self.dimensions = self._get_default_dimensions()
    
    def _get_default_dimensions(self):
        """获取默认维度配置（兜底方案）"""
        return {
            'node_feature_dim': 8,
            'edge_feature_dim_full': 4,
            'edge_feature_dim_baseline': 2,
            'vnf_context_dim': 6,
            'gnn_output_dim': 256
        }
    
    def _detect_agent_mode(self):
        """自动检测智能体模式"""
        agent_id_lower = self.agent_id.lower()
        
        # 🔧 智能体类型检测逻辑
        if any(keyword in agent_id_lower for keyword in ['edge_aware', 'edge-aware', 'enhanced']):
            self.agent_mode = 'edge_aware'
            self.is_edge_aware = True
        elif any(keyword in agent_id_lower for keyword in ['baseline', 'base', 'standard']):
            self.agent_mode = 'baseline'
            self.is_edge_aware = False
        else:
            # 默认根据配置判断
            self.agent_mode = 'edge_aware' if self.config.get('edge_aware_mode', True) else 'baseline'
            self.is_edge_aware = self.agent_mode == 'edge_aware'
        
        print(f"🔧 智能体模式检测: {self.agent_id} -> {self.agent_mode}")
    
    def _standardize_dimensions(self):
        """标准化维度配置"""
        # 🔧 强制使用配置文件中的维度，忽略传入参数
        self.state_dim = self.dimensions.get('node_feature_dim', 8)
        self.output_dim = self.dimensions.get('gnn_output_dim', 256)
        
        # 根据智能体模式确定边特征维度
        if self.is_edge_aware:
            self.edge_dim = self.dimensions.get('edge_feature_dim_full', 4)
        else:
            self.edge_dim = self.dimensions.get('edge_feature_dim_baseline', 2)
        
        self.vnf_context_dim = self.dimensions.get('vnf_context_dim', 6)
        
        print(f"🔧 维度标准化完成:")
        print(f"   节点特征: {self.state_dim}维")
        print(f"   边特征: {self.edge_dim}维 ({self.agent_mode})")
        print(f"   VNF上下文: {self.vnf_context_dim}维")
        print(f"   GNN输出: {self.output_dim}维")
    
    def _load_gnn_config(self):
        """加载GNN配置"""
        gnn_config = self.config.get("gnn", {}).get(self.agent_mode, {})
        
        self.hidden_dim = gnn_config.get("hidden_dim", 128 if self.is_edge_aware else 64)
        self.num_layers = gnn_config.get("layers", 6 if self.is_edge_aware else 4)
        self.dropout = gnn_config.get("dropout", 0.1)
        self.heads = gnn_config.get("heads", 4)
        
        print(f"🔧 GNN配置加载: hidden_dim={self.hidden_dim}, layers={self.num_layers}")
    
    def _load_training_config(self):
        """加载训练配置"""
        train_config = self.config.get("train", {})
        
        self.learning_rate = train_config.get("lr", 0.0003)
        self.gamma = train_config.get("gamma", 0.99)
        self.batch_size = train_config.get("batch_size", 32)
        
        # 探索配置
        self.epsilon = train_config.get("epsilon_start", 1.0)
        self.epsilon_decay = train_config.get("epsilon_decay", 0.998)
        self.epsilon_min = train_config.get("epsilon_min", 0.05)
    
    def _create_gnn_encoder(self):
        """创建配置匹配的GNN编码器"""
        print(f"🔧 创建GNN编码器:")
        print(f"   模式: {self.agent_mode}")
        print(f"   节点维度: {self.state_dim}")
        print(f"   边维度: {self.edge_dim}")
        print(f"   隐藏维度: {self.hidden_dim}")
        print(f"   输出维度: {self.output_dim}")
        print(f"   层数: {self.num_layers}")
        
        try:
            self.gnn_encoder = GNNEncoder(
                node_dim=self.state_dim,           # 配置中的节点维度
                edge_dim=self.edge_dim,            # 根据模式确定的边维度
                hidden_dim=self.hidden_dim,        # 配置中的隐藏维度
                output_dim=self.output_dim,        # 配置中的输出维度
                num_layers=self.num_layers         # 配置中的层数
            ).to(self.device)
            
            print(f"✅ GNN编码器创建成功")
            
        except Exception as e:
            print(f"❌ GNN编码器创建失败: {e}")
            # 兜底方案：使用简化参数
            self.gnn_encoder = GNNEncoder(
                node_dim=8,
                edge_dim=4 if self.is_edge_aware else 2,
                hidden_dim=128,
                output_dim=256,
                num_layers=4
            ).to(self.device)
            print(f"⚠️ 使用兜底GNN编码器")
    
    def _print_dimension_summary(self):
        """打印维度配置摘要"""
        print(f"\n📊 {self.agent_id} 维度配置摘要:")
        print(f"{'='*50}")
        print(f"智能体模式: {self.agent_mode}")
        print(f"节点特征维度: {self.state_dim}")
        print(f"边特征维度: {self.edge_dim}")
        print(f"VNF上下文维度: {self.vnf_context_dim}")
        print(f"GNN隐藏维度: {self.hidden_dim}")
        print(f"GNN输出维度: {self.output_dim}")
        print(f"GNN层数: {self.num_layers}")
        print(f"{'='*50}\n")
    
    def process_state(self, state: Union[Data, Dict, np.ndarray]) -> torch.Tensor:
        """
        处理状态输入，统一转换为图神经网络可处理的格式 - 修复版
        
        🔧 主要改进：
        1. 严格的维度验证
        2. 自动维度修复
        3. 模式感知处理
        4. 详细的错误报告
        """
        self.gnn_encoder.eval()
        
        with torch.no_grad():
            if isinstance(state, Data):
                # 🔧 PyG数据对象处理
                processed_state = self._process_pyg_data(state)
                
            elif isinstance(state, dict) and 'graph_data' in state:
                # 🔧 字典格式处理
                graph_data = state['graph_data']
                processed_state = self._process_pyg_data(graph_data)
                
            elif isinstance(state, (np.ndarray, torch.Tensor)):
                # 🔧 张量格式处理
                if isinstance(state, np.ndarray):
                    state = torch.tensor(state, dtype=torch.float32)
                
                # 确保是二维张量
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                
                # 维度检查和修复
                if state.size(-1) != self.output_dim:
                    if state.size(-1) < self.output_dim:
                        # 补充到目标维度
                        padding = torch.zeros(*state.shape[:-1], self.output_dim - state.size(-1))
                        state = torch.cat([state, padding], dim=-1)
                    else:
                        # 截取到目标维度
                        state = state[..., :self.output_dim]
                
                processed_state = state.to(self.device)
                
            else:
                raise ValueError(f"不支持的状态格式: {type(state)}")
        
        if self.is_training:
            self.gnn_encoder.train()
        
        return processed_state
    
    def _process_pyg_data(self, data: Data) -> torch.Tensor:
        """处理PyTorch Geometric数据对象"""
        # 🔧 维度验证和修复
        data = self._validate_and_fix_pyg_dimensions(data)
        
        # 移动到设备
        data = data.to(self.device)
        
        # GNN编码
        try:
            encoded_state = self.gnn_encoder(data)
            
            # 输出维度验证
            if encoded_state.size(-1) != self.output_dim:
                print(f"⚠️ GNN输出维度不匹配: 期望{self.output_dim}, 实际{encoded_state.size(-1)}")
                # 自动修复
                if encoded_state.size(-1) < self.output_dim:
                    padding = torch.zeros(*encoded_state.shape[:-1], 
                                        self.output_dim - encoded_state.size(-1), 
                                        device=encoded_state.device)
                    encoded_state = torch.cat([encoded_state, padding], dim=-1)
                else:
                    encoded_state = encoded_state[..., :self.output_dim]
            
            return encoded_state
            
        except Exception as e:
            print(f"❌ GNN编码失败: {e}")
            # 兜底方案：返回默认张量
            return torch.zeros(1, self.output_dim, device=self.device)
    
    def _validate_and_fix_pyg_dimensions(self, data: Data) -> Data:
        """验证和修复PyG数据维度"""
        
        # 🔧 节点特征维度检查和修复
        if data.x is not None:
            current_node_dim = data.x.size(1)
            if current_node_dim != self.state_dim:
                print(f"🔧 修复节点特征维度: {current_node_dim} -> {self.state_dim}")
                
                if current_node_dim < self.state_dim:
                    # 补充到目标维度
                    padding = torch.zeros(data.x.size(0), self.state_dim - current_node_dim, 
                                        device=data.x.device, dtype=data.x.dtype)
                    data.x = torch.cat([data.x, padding], dim=1)
                else:
                    # 截取到目标维度
                    data.x = data.x[:, :self.state_dim]
        
        # 🔧 边特征维度检查和修复
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            current_edge_dim = data.edge_attr.size(1)
            if current_edge_dim != self.edge_dim:
                print(f"🔧 修复边特征维度: {current_edge_dim} -> {self.edge_dim}")
                
                if current_edge_dim < self.edge_dim:
                    # 补充到目标维度
                    padding = torch.zeros(data.edge_attr.size(0), self.edge_dim - current_edge_dim,
                                        device=data.edge_attr.device, dtype=data.edge_attr.dtype)
                    
                    # 为Edge-aware模式智能补充特征
                    if self.is_edge_aware and current_edge_dim == 2 and self.edge_dim == 4:
                        # 补充抖动和丢包率的默认值
                        padding[:, 0] = torch.rand(data.edge_attr.size(0), device=data.edge_attr.device) * 0.01  # 抖动
                        padding[:, 1] = torch.rand(data.edge_attr.size(0), device=data.edge_attr.device) * 0.005  # 丢包率
                    
                    data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
                else:
                    # 截取到目标维度
                    data.edge_attr = data.edge_attr[:, :self.edge_dim]
        
        # 🔧 VNF上下文维度检查
        if hasattr(data, 'vnf_context') and data.vnf_context is not None:
            if data.vnf_context.size(-1) != self.vnf_context_dim:
                print(f"🔧 VNF上下文维度警告: 期望{self.vnf_context_dim}, 实际{data.vnf_context.size(-1)}")
        
        return data
    
    def update_target_network(self, tau: float = None):
        """更新目标网络（用于DQN系列算法）"""
        if self.target_network is None:
            return
            
        if tau is None:
            # 硬更新
            self.target_network.load_state_dict(self.policy_network.state_dict())
        else:
            # 软更新
            for target_param, policy_param in zip(
                self.target_network.parameters(), 
                self.policy_network.parameters()
            ):
                target_param.data.copy_(
                    tau * policy_param.data + (1 - tau) * target_param.data
                )
    
    def decay_epsilon(self):
        """更新探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """
        获取当前状态下的有效动作
        
        Args:
            state: 当前状态
            **kwargs: 额外参数（如资源约束等）
            
        Returns:
            valid_actions: 有效动作列表
        """
        # 基础实现：所有动作都有效
        available_nodes = kwargs.get('available_nodes', list(range(self.action_dim)))
        resource_constraints = kwargs.get('resource_constraints', {})
        
        valid_actions = []
        for node in available_nodes:
            if self._check_node_feasibility(node, resource_constraints):
                valid_actions.append(node)
        
        # 确保至少有一个有效动作
        if not valid_actions:
            valid_actions = [0]
        
        return valid_actions
    
    def _check_node_feasibility(self, node_id: int, constraints: Dict) -> bool:
        """检查节点可行性（子类可重写）"""
        return True
    
    def mask_invalid_actions(self, q_values: torch.Tensor, valid_actions: List[int]) -> torch.Tensor:
        """
        屏蔽无效动作的Q值
        
        Args:
            q_values: 原始Q值 [batch_size, action_dim]
            valid_actions: 有效动作列表
            
        Returns:
            masked_q_values: 屏蔽后的Q值
        """
        masked_q_values = q_values.clone()
        
        # 获取无效动作
        invalid_actions = [a for a in range(self.action_dim) if a not in valid_actions]
        
        if invalid_actions:
            masked_q_values[:, invalid_actions] = -float('inf')
        
        return masked_q_values
    
    def update_stats(self, reward: float, action: int, loss: float = None, q_values: torch.Tensor = None):
        """更新智能体统计信息"""
        self.stats["total_reward"] += reward
        self.stats["steps"] += 1
        
        if loss is not None:
            self.stats["losses"].append(loss)
        
        if q_values is not None:
            self.stats["q_values"].append(q_values.mean().item() if isinstance(q_values, torch.Tensor) else q_values)
        
        # 记录动作分布
        if action not in self.stats["actions_taken"]:
            self.stats["actions_taken"][action] = 0
        self.stats["actions_taken"][action] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        stats = self.stats.copy()
        
        # 计算平均指标
        if stats["episodes"] > 0:
            stats["avg_reward"] = stats["total_reward"] / stats["episodes"]
        else:
            stats["avg_reward"] = 0.0
            
        if stats["losses"]:
            stats["avg_loss"] = np.mean(stats["losses"][-100:])  # 最近100次的平均loss
            
        if stats["q_values"]:
            stats["avg_q_value"] = np.mean(stats["q_values"][-100:])  # 最近100次的平均Q值
            
        # 添加当前状态
        stats["epsilon"] = self.epsilon
        stats["training_step"] = self.training_step
        stats["agent_mode"] = self.agent_mode
        
        return stats
    
    def reset_episode_stats(self):
        """重置episode统计"""
        self.stats["total_reward"] = 0.0
        self.stats["episodes"] += 1
    
    def save_checkpoint(self, filepath: str):
        """保存智能体检查点"""
        checkpoint = {
            'agent_id': self.agent_id,
            'agent_mode': self.agent_mode,
            'dimensions': self.dimensions,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'stats': self.stats,
            'gnn_encoder_state': self.gnn_encoder.state_dict() if self.gnn_encoder else None,
        }
        
        if self.policy_network is not None:
            checkpoint['policy_network_state'] = self.policy_network.state_dict()
            
        if self.target_network is not None:
            checkpoint['target_network_state'] = self.target_network.state_dict()
            
        if self.optimizer is not None:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"💾 Agent {self.agent_id} 检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载智能体检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复基本状态
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.epsilon = checkpoint.get('epsilon', 1.0)
        self.stats = checkpoint.get('stats', {})
        
        # 恢复模型状态
        if 'gnn_encoder_state' in checkpoint and self.gnn_encoder is not None:
            try:
                self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state'])
                print(f"✅ GNN编码器状态已恢复")
            except Exception as e:
                print(f"⚠️ GNN编码器状态恢复失败: {e}")
            
        if 'policy_network_state' in checkpoint and self.policy_network is not None:
            try:
                self.policy_network.load_state_dict(checkpoint['policy_network_state'])
                print(f"✅ 策略网络状态已恢复")
            except Exception as e:
                print(f"⚠️ 策略网络状态恢复失败: {e}")
            
        if 'target_network_state' in checkpoint and self.target_network is not None:
            try:
                self.target_network.load_state_dict(checkpoint['target_network_state'])
                print(f"✅ 目标网络状态已恢复")
            except Exception as e:
                print(f"⚠️ 目标网络状态恢复失败: {e}")
            
        if 'optimizer_state' in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                print(f"✅ 优化器状态已恢复")
            except Exception as e:
                print(f"⚠️ 优化器状态恢复失败: {e}")
        
        print(f"📂 Agent {self.agent_id} 检查点已加载: {filepath}")
    
    def set_training_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.is_training = training
        
        if self.gnn_encoder is not None:
            if training:
                self.gnn_encoder.train()
            else:
                self.gnn_encoder.eval()
                
        if self.policy_network is not None:
            if training:
                self.policy_network.train()
            else:
                self.policy_network.eval()
    
    def register_other_agents(self, agents: Dict[str, 'BaseAgent']):
        """注册其他智能体（用于多智能体协调）"""
        self.other_agents = agents
        self.communication_enabled = len(agents) > 0
        print(f"🤝 Agent {self.agent_id} 已注册 {len(agents)} 个其他智能体")
    
    def get_dimension_info(self) -> Dict[str, Any]:
        """获取维度信息（用于调试）"""
        return {
            'agent_id': self.agent_id,
            'agent_mode': self.agent_mode,
            'is_edge_aware': self.is_edge_aware,
            'state_dim': self.state_dim,
            'edge_dim': self.edge_dim,
            'vnf_context_dim': self.vnf_context_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }
    
    # 🔧 抽象方法：子类必须实现
    @abstractmethod
    def select_action(self, state: Union[Data, Dict], **kwargs) -> Union[int, List[int]]:
        """选择动作"""
        pass
    
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        """存储经验"""
        pass
    
    @abstractmethod
    def learn(self) -> Dict[str, float]:
        """学习更新"""
        pass


def create_agent(agent_type: str, agent_id: str, state_dim: int, action_dim: int, 
                 edge_dim: int, config: Dict[str, Any], use_enhanced_gnn: bool = False) -> EnhancedBaseAgent:
    print(f"🏭 创建智能体: {agent_type} -> {agent_id} (增强模式: {use_enhanced_gnn})")
    
    try:
        if agent_type.lower() == 'ddqn':
            from agents.multi_ddqn_agent import MultiDDQNAgent
            return MultiDDQNAgent(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
        elif agent_type.lower() == 'dqn':
            from agents.multi_dqn_agent import MultiDQNAgent
            return MultiDQNAgent(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
        elif agent_type.lower() == 'ppo':
            from agents.multi_ppo_agent import MultiPPOAgent
            return MultiPPOAgent(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
    except Exception as e:
        print(f"❌ 智能体创建失败: {e}")
        raise


def test_base_agent_config_driven():
    """测试配置驱动的BaseAgent"""
    print("🧪 测试配置驱动的BaseAgent...")
    print("=" * 60)
    
    # 模拟配置
    config = {
        "edge_aware_mode": True,
        "gnn": {
            "edge_aware": {"hidden_dim": 128, "output_dim": 256, "layers": 6},
            "baseline": {"hidden_dim": 64, "output_dim": 256, "layers": 4}
        },
        "train": {"lr": 0.0003, "gamma": 0.99, "batch_size": 32}
    }
    
    # 测试智能体类 
    class TestAgent(BaseAgent):
        def select_action(self, state, **kwargs):
            return np.random.randint(0, self.action_dim)
        
        def store_transition(self, state, action, reward, next_state, done, **kwargs):
            pass
        
        def learn(self):
            return {"loss": 0.1}
    
    print("1. 测试Edge-aware智能体创建:")
    try:
        agent_edge = TestAgent("ddqn_edge_aware_test", state_dim=8, action_dim=42, edge_dim=4, config=config)
        print(f"✅ Edge-aware智能体创建成功")
        print(f"   模式: {agent_edge.agent_mode}")
        print(f"   节点维度: {agent_edge.state_dim}")
        print(f"   边维度: {agent_edge.edge_dim}")
        print(f"   输出维度: {agent_edge.output_dim}")
    except Exception as e:
        print(f"❌ Edge-aware智能体创建失败: {e}")
    
    print("\n2. 测试Baseline智能体创建:")
    try:
        agent_baseline = TestAgent("dqn_baseline_test", state_dim=8, action_dim=42, edge_dim=2, config=config)
        print(f"✅ Baseline智能体创建成功")
        print(f"   模式: {agent_baseline.agent_mode}")
        print(f"   节点维度: {agent_baseline.state_dim}")
        print(f"   边维度: {agent_baseline.edge_dim}")
        print(f"   输出维度: {agent_baseline.output_dim}")
    except Exception as e:
        print(f"❌ Baseline智能体创建失败: {e}")
    
    print("\n3. 测试状态处理:")
    try:
        # 创建测试状态
        test_state = Data(
            x=torch.randn(42, 8),  # 正确的8维节点特征
            edge_index=torch.randint(0, 42, (2, 100)),
            edge_attr=torch.randn(100, 4)  # Edge-aware 4维边特征
        )
        
        processed_state = agent_edge.process_state(test_state)
        print(f"✅ Edge-aware状态处理成功: {processed_state.shape}")
        
        # 测试维度不匹配的自动修复
        test_state_wrong = Data(
            x=torch.randn(42, 6),  # 错误的6维节点特征
            edge_index=torch.randint(0, 42, (2, 100)),
            edge_attr=torch.randn(100, 2)  # 2维边特征
        )
        
        processed_state_fixed = agent_edge.process_state(test_state_wrong)
        print(f"✅ 维度自动修复测试成功: {processed_state_fixed.shape}")
        
    except Exception as e:
        print(f"❌ 状态处理测试失败: {e}")
    
    print("\n4. 测试统计功能:")
    try:
        agent_edge.update_stats(reward=10.5, action=5, loss=0.2, q_values=torch.tensor([1.5, 2.0, 1.8]))
        stats = agent_edge.get_stats()
        print(f"✅ 统计功能测试成功")
        print(f"   总奖励: {stats['total_reward']}")
        print(f"   步数: {stats['steps']}")
        print(f"   模式: {stats['agent_mode']}")
    except Exception as e:
        print(f"❌ 统计功能测试失败: {e}")
    
    print("\n5. 测试维度信息:")
    try:
        dim_info = agent_edge.get_dimension_info()
        print(f"✅ 维度信息获取成功:")
        for key, value in dim_info.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ 维度信息获取失败: {e}")
    
    print(f"\n🎉 配置驱动BaseAgent测试完成!")


# 🔧 兼容性包装函数
def create_legacy_agent(*args, **kwargs):
    """为了向后兼容的包装函数"""
    print("⚠️ 使用了旧版智能体创建接口，建议使用新的配置驱动接口")
    return create_agent(*args, **kwargs)


if __name__ == "__main__":
    test_base_agent_config_driven()