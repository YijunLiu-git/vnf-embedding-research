# agents/base_agent.py - 修复版

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
from torch_geometric.data import Data, Batch

# 🔧 关键修复：只使用标准GNN编码器，暂时禁用Enhanced版本
from models.gnn_encoder import GNNEncoder

class BaseAgent(ABC):
    """
    多智能体VNF嵌入系统的基础智能体类 - 修复版
    
    主要修复：
    1. 🔧 禁用enhanced_gnn_encoder，只使用标准GNNEncoder
    2. 🔧 确保维度配置一致性
    3. 🔧 简化初始化流程，避免复杂的特征融合
    """
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int, 
                 action_dim: int, 
                 edge_dim: int,
                 config: Dict[str, Any]):
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.edge_dim = edge_dim
        self.config = config
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Agent {agent_id} 使用设备: {self.device}")
        
        # 🔧 修复：简化GNN配置获取逻辑
        if "edge_aware" in agent_id:
            gnn_config = config.get("gnn", {}).get("edge_aware", {})
            print(f"   📊 Edge-aware模式: edge_dim={edge_dim}")
        else:
            gnn_config = config.get("gnn", {}).get("baseline", {})
            print(f"   📊 Baseline模式: edge_dim={edge_dim}")
        
        self.hidden_dim = gnn_config.get("hidden_dim", 128)
        self.output_dim = gnn_config.get("output_dim", 256)
        self.num_layers = gnn_config.get("layers", 4)
        
        # 训练配置
        self.learning_rate = config.get("train", {}).get("lr", 0.001)
        self.gamma = config.get("train", {}).get("gamma", 0.99)
        self.batch_size = config.get("train", {}).get("batch_size", 32)
        
        # 探索配置
        self.epsilon = config.get("train", {}).get("epsilon_start", 1.0)
        self.epsilon_decay = config.get("train", {}).get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("train", {}).get("epsilon_min", 0.01)
        
        # 🔧 关键修复：使用标准GNN编码器，确保维度匹配
        print(f"   🔧 创建GNN编码器: node_dim={state_dim}, edge_dim={edge_dim}")
        print(f"      hidden_dim={self.hidden_dim}, output_dim={self.output_dim}")
        
        self.gnn_encoder = GNNEncoder(
            node_dim=state_dim,           # 8维节点特征
            edge_dim=edge_dim,            # edge_aware=4, baseline=2
            hidden_dim=self.hidden_dim,   # 配置中的hidden_dim
            output_dim=self.output_dim,   # 统一256维输出
            num_layers=self.num_layers    # 配置中的层数
        ).to(self.device)
        
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
        
        print(f"✅ GNN编码器创建成功")
        
    def process_state(self, state: Union[Data, Dict, np.ndarray]) -> torch.Tensor:
        """
        处理状态输入，统一转换为图神经网络可处理的格式 - 修复版
        
        Args:
            state: 可以是PyG Data对象、字典或numpy数组
            
        Returns:
            processed_state: 处理后的状态tensor [1, output_dim]
        """
        self.gnn_encoder.eval()
        
        with torch.no_grad():
            if isinstance(state, Data):
                # 🔧 关键修复：验证输入维度
                if state.x.size(1) != self.state_dim:
                    print(f"⚠️ 节点特征维度不匹配: 期望{self.state_dim}, 实际{state.x.size(1)}")
                    # 尝试自动修复
                    if state.x.size(1) < self.state_dim:
                        padding = torch.zeros(state.x.size(0), self.state_dim - state.x.size(1), 
                                            device=state.x.device)
                        state.x = torch.cat([state.x, padding], dim=1)
                    else:
                        state.x = state.x[:, :self.state_dim]
                
                if hasattr(state, 'edge_attr') and state.edge_attr is not None:
                    if state.edge_attr.size(1) != self.edge_dim:
                        print(f"⚠️ 边特征维度不匹配: 期望{self.edge_dim}, 实际{state.edge_attr.size(1)}")
                        # 尝试自动修复
                        if state.edge_attr.size(1) < self.edge_dim:
                            padding = torch.zeros(state.edge_attr.size(0), 
                                                self.edge_dim - state.edge_attr.size(1), 
                                                device=state.edge_attr.device)
                            state.edge_attr = torch.cat([state.edge_attr, padding], dim=1)
                        else:
                            state.edge_attr = state.edge_attr[:, :self.edge_dim]
                
                state = state.to(self.device)
                encoded_state = self.gnn_encoder(state)
                
            elif isinstance(state, dict) and 'graph_data' in state:
                graph_data = state['graph_data'].to(self.device)
                encoded_state = self.gnn_encoder(graph_data)
                
            elif isinstance(state, (np.ndarray, torch.Tensor)):
                if isinstance(state, np.ndarray):
                    state = torch.tensor(state, dtype=torch.float32)
                encoded_state = state.unsqueeze(0).to(self.device)
                
            else:
                raise ValueError(f"不支持的状态格式: {type(state)}")
        
        if self.is_training:
            self.gnn_encoder.train()
        
        # 🔧 验证输出维度
        if encoded_state.size(-1) != self.output_dim:
            print(f"⚠️ GNN输出维度不匹配: 期望{self.output_dim}, 实际{encoded_state.size(-1)}")
            
        return encoded_state
    
    def update_target_network(self, tau: float = None):
        """
        更新目标网络（用于DQN系列算法）
        
        Args:
            tau: 软更新参数，None表示硬更新
        """
        if self.target_network is None:
            return
            
        if tau is None:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        else:
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
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """
        获取当前状态下的有效动作
        
        Args:
            state: 当前状态
            **kwargs: 额外参数（如资源约束等）
            
        Returns:
            valid_actions: 有效动作列表
        """
        return list(range(self.action_dim))
    
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
            self.stats["q_values"].append(q_values.mean().item())
        
        if action not in self.stats["actions_taken"]:
            self.stats["actions_taken"][action] = 0
        self.stats["actions_taken"][action] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        stats = self.stats.copy()
        
        if stats["episodes"] > 0:
            stats["avg_reward"] = stats["total_reward"] / stats["episodes"]
        else:
            stats["avg_reward"] = 0.0
            
        if stats["losses"]:
            stats["avg_loss"] = np.mean(stats["losses"][-100:])
            
        if stats["q_values"]:
            stats["avg_q_value"] = np.mean(stats["q_values"][-100:])
            
        stats["epsilon"] = self.epsilon
        
        return stats
    
    def reset_episode_stats(self):
        """重置episode统计"""
        self.stats["total_reward"] = 0.0
        self.stats["episodes"] += 1
    
    def save_checkpoint(self, filepath: str):
        """保存智能体检查点"""
        checkpoint = {
            'agent_id': self.agent_id,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'stats': self.stats,
            'gnn_encoder_state': self.gnn_encoder.state_dict(),
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
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
        self.stats = checkpoint['stats']
        
        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state'])
        
        if 'policy_network_state' in checkpoint and self.policy_network is not None:
            self.policy_network.load_state_dict(checkpoint['policy_network_state'])
            
        if 'target_network_state' in checkpoint and self.target_network is not None:
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            
        if 'optimizer_state' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"📂 Agent {self.agent_id} 检查点已加载: {filepath}")
    
    def set_training_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.is_training = training
        
        if training:
            self.gnn_encoder.train()
            if self.policy_network is not None:
                self.policy_network.train()
        else:
            self.gnn_encoder.eval()
            if self.policy_network is not None:
                self.policy_network.eval()
    
    def register_other_agents(self, agents: Dict[str, 'BaseAgent']):
        """注册其他智能体（用于多智能体协调）"""
        self.other_agents = agents
        self.communication_enabled = len(agents) > 0
        print(f"🤝 Agent {self.agent_id} 已注册 {len(agents)} 个其他智能体")
    
    @abstractmethod
    def select_action(self, state: Union[Data, Dict], **kwargs) -> Union[int, List[int]]:
        pass
    
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        pass
    
    @abstractmethod
    def learn(self) -> Dict[str, float]:
        pass

def create_agent(agent_type: str, agent_id: str, state_dim: int, action_dim: int, 
                edge_dim: int, config: Dict[str, Any]) -> BaseAgent:
    """
    工厂函数：创建指定类型的智能体 - 修复版
    
    Args:
        agent_type: 智能体类型 ('ddqn', 'dqn', 'ppo')
        agent_id: 智能体ID
        state_dim: 状态维度
        action_dim: 动作维度
        edge_dim: 边特征维度
        config: 配置字典
        
    Returns:
        agent: 创建的智能体实例
    """
    
    print(f"🏭 创建智能体: {agent_type} -> {agent_id}")
    print(f"   参数: state_dim={state_dim}, action_dim={action_dim}, edge_dim={edge_dim}")
    
    if agent_type.lower() == 'ddqn':
        from agents.multi_ddqn_agent import MultiDDQNAgent
        return MultiDDQNAgent(agent_id, state_dim, action_dim, edge_dim, config)
    elif agent_type.lower() == 'dqn':
        from agents.multi_dqn_agent import MultiDQNAgent
        return MultiDQNAgent(agent_id, state_dim, action_dim, edge_dim, config)
    elif agent_type.lower() == 'ppo':
        from agents.multi_ppo_agent import MultiPPOAgent
        return MultiPPOAgent(agent_id, state_dim, action_dim, edge_dim, config)
    else:
        raise ValueError(f"不支持的智能体类型: {agent_type}")

def test_base_agent_fixed():
    """测试修复版BaseAgent基础功能"""
    print("🧪 测试修复版BaseAgent基础功能...")
    
    config = {
        "gnn": {
            "edge_aware": {"hidden_dim": 64, "output_dim": 256, "layers": 4},
            "baseline": {"hidden_dim": 64, "output_dim": 256, "layers": 4}
        },
        "train": {"lr": 0.001, "gamma": 0.99, "batch_size": 16}
    }
    
    class TestAgent(BaseAgent):
        def __init__(self, agent_id, state_dim, action_dim, edge_dim, config):
            super().__init__(agent_id, state_dim, action_dim, edge_dim, config)
            
        def select_action(self, state, **kwargs):
            return np.random.randint(0, self.action_dim)
            
        def store_transition(self, state, action, reward, next_state, done, **kwargs):
            pass
            
        def learn(self):
            return {"loss": 0.1}
    
    # 测试Edge-aware智能体
    agent_edge = TestAgent("test_agent_edge_aware", state_dim=8, action_dim=10, edge_dim=4, config=config)
    
    # 测试Baseline智能体
    agent_baseline = TestAgent("test_agent_baseline", state_dim=8, action_dim=10, edge_dim=2, config=config)
    
    # 创建测试状态
    test_state = Data(
        x=torch.randn(10, 8),  # 10个节点，8维特征
        edge_index=torch.randint(0, 10, (2, 20)),  # 20条边
        edge_attr=torch.randn(20, 4)  # 4维边特征
    )
    
    # 测试Edge-aware处理
    processed_state_edge = agent_edge.process_state(test_state)
    print(f"✅ Edge-aware状态处理: 输入{test_state.x.shape} -> 输出{processed_state_edge.shape}")
    
    # 测试Baseline处理（需要调整边特征维度）
    test_state_baseline = Data(
        x=torch.randn(10, 8),  # 10个节点，8维特征
        edge_index=torch.randint(0, 10, (2, 20)),  # 20条边
        edge_attr=torch.randn(20, 2)  # 2维边特征
    )
    
    processed_state_baseline = agent_baseline.process_state(test_state_baseline)
    print(f"✅ Baseline状态处理: 输入{test_state_baseline.x.shape} -> 输出{processed_state_baseline.shape}")
    
    print(f"✅ 修复版BaseAgent测试完成!")
    print(f"   Edge-aware智能体创建成功: {agent_edge.agent_id}")
    print(f"   Baseline智能体创建成功: {agent_baseline.agent_id}")
    print(f"   输出维度统一: {processed_state_edge.shape[-1]} = {processed_state_baseline.shape[-1]}")

if __name__ == "__main__":
    test_base_agent_fixed()