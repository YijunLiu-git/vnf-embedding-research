# agents/multi_ddqn_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Any
from torch_geometric.data import Data

from agents.base_agent import BaseAgent
from utils.replay_buffer import PrioritizedReplayBuffer

class DDQNNetwork(nn.Module):
    """
    双深度Q网络 - 策略网络和目标网络的共同结构
    
    架构：
    GNNEncoder -> 全连接层 -> Q值输出
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512):
        super(DDQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Q值网络
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 网络初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state_embedding: GNN编码后的状态 [batch_size, input_dim]
            
        Returns:
            q_values: Q值 [batch_size, action_dim]
        """
        return self.q_network(state_embedding)


class MultiDDQNAgent(BaseAgent):
    """
    多智能体双深度Q学习智能体
    
    特性：
    1. 使用GNN编码器处理图状态
    2. 双Q网络减少过估计
    3. 优先级经验回放
    4. VNF嵌入专用的动作选择逻辑
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, edge_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, edge_dim, config)
        
        # DDQN特定配置
        self.target_update_freq = config.get("train", {}).get("target_update", 100)
        self.double_q = True  # 启用双Q学习
        
        # 网络架构
        network_input_dim = self.output_dim  # GNNEncoder的输出维度
        
        # 策略网络（主网络）
        self.policy_network = DDQNNetwork(
            input_dim=network_input_dim,
            action_dim=action_dim,
            hidden_dim=config.get("network", {}).get("hidden_dim", 512)
        ).to(self.device)
        
        # 目标网络
        self.target_network = DDQNNetwork(
            input_dim=network_input_dim,
            action_dim=action_dim,
            hidden_dim=config.get("network", {}).get("hidden_dim", 512)
        ).to(self.device)
        
        # 初始化目标网络
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # 优先级经验回放
        buffer_size = config.get("train", {}).get("buffer_size", 10000)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=0.6,  # 优先级指数
            beta=0.4    # 重要性采样指数
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )
        
        print(f"🚀 DDQN Agent {agent_id} 初始化完成")
        print(f"   - 状态维度: {state_dim} -> GNN编码 -> {network_input_dim}")
        print(f"   - 动作维度: {action_dim}")
        print(f"   - 设备: {self.device}")
    
    def select_action(self, state: Union[Data, Dict], valid_actions: List[int] = None, **kwargs) -> int:
        """
        选择动作 - 修复为单动作选择
        
        Args:
            state: 当前状态（图数据或编码后状态）
            valid_actions: 有效动作列表
            **kwargs: 额外参数
            
        Returns:
            action: 选择的单个动作
        """
        
        # 处理状态
        if isinstance(state, Data):
            state_embedding = self.process_state(state)
        else:
            state_embedding = state.to(self.device) if isinstance(state, torch.Tensor) else torch.tensor(state, device=self.device)
            if state_embedding.dim() == 1:
                state_embedding = state_embedding.unsqueeze(0)
        
        # 获取有效动作
        if valid_actions is None:
            valid_actions = self.get_valid_actions(state, **kwargs)
        
        # ε-贪婪策略
        if self.is_training and np.random.random() < self.epsilon:
            # 探索：从有效动作中随机选择
            action = np.random.choice(valid_actions)
        else:
            # 利用：选择Q值最高的有效动作
            self.policy_network.eval()
            with torch.no_grad():
                q_values = self.policy_network(state_embedding)
                
                # 屏蔽无效动作
                masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
                action = masked_q_values.argmax(dim=1).item()
            
            if self.is_training:
                self.policy_network.train()
        
        return action
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """
        获取VNF嵌入场景下的有效动作
        
        这里可以根据具体的VNF嵌入约束来实现，比如：
        - 资源约束：节点CPU/内存是否足够
        - 连通性约束：节点是否可达
        - 策略约束：某些节点是否被策略禁止
        """
        # 基础实现：所有动作都有效
        # 在实际应用中，这里应该检查：
        # 1. 节点资源是否满足当前VNF需求
        # 2. 节点是否已被其他VNF占用
        # 3. 网络连通性是否满足要求
        
        available_nodes = kwargs.get('available_nodes', list(range(self.action_dim)))
        resource_constraints = kwargs.get('resource_constraints', {})
        
        valid_actions = []
        for node in available_nodes:
            # 这里添加具体的约束检查逻辑
            if self._check_node_feasibility(node, resource_constraints):
                valid_actions.append(node)
        
        # 确保至少有一个有效动作
        if not valid_actions:
            valid_actions = [0]  # 默认选择节点0
        
        return valid_actions
    
    def _check_node_feasibility(self, node_id: int, constraints: Dict) -> bool:
        """检查节点是否满足VNF嵌入约束"""
        # 简化实现，实际应用中需要更复杂的逻辑
        return True
    
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        """
        存储经验到优先级回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            **kwargs: 额外信息（如TD误差）
        """
        
        # 计算初始优先级（可以基于奖励或TD误差）
        priority = kwargs.get('priority', abs(reward) + 1e-6)
        
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            priority=priority,
            agent_id=self.agent_id,
            **kwargs
        )
    
    def learn(self) -> Dict[str, float]:
        """
        DDQN学习更新
        
        Returns:
            learning_info: 学习统计信息
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_value": 0.0}
        
        # 优先级采样
        batch_data = self.replay_buffer.sample(self.batch_size, device=self.device)
        states, actions, rewards, next_states, dones, weights, indices = batch_data
        
        # 处理状态编码
        if isinstance(states, Data):
            # 图数据：使用GNN编码
            state_embeddings = self.gnn_encoder(states)
            next_state_embeddings = self.gnn_encoder(next_states)
        else:
            # 已编码状态
            state_embeddings = states
            next_state_embeddings = next_states
        
        # 当前Q值
        current_q_values = self.policy_network(state_embeddings)
        
        # 处理动作索引
        if isinstance(actions, torch.Tensor):
            action_indices = actions
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            # 处理列表形式的动作
            action_indices = torch.tensor([a if isinstance(a, int) else a[0] for a in actions], 
                                        dtype=torch.long, device=self.device)
            current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # 双Q学习：使用策略网络选择动作，目标网络评估价值
        with torch.no_grad():
            # 策略网络选择下一状态的最优动作
            next_q_values_policy = self.policy_network(next_state_embeddings)
            next_actions = next_q_values_policy.argmax(dim=1, keepdim=True)
            
            # 目标网络评估选定动作的价值
            next_q_values_target = self.target_network(next_state_embeddings)
            next_q = next_q_values_target.gather(1, next_actions).squeeze(1)
            
            # 计算目标Q值
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # 计算TD误差
        td_errors = torch.abs(current_q - target_q)
        
        # 重要性采样加权损失
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # 更新探索率
        self.decay_epsilon()
        
        # 更新统计信息
        avg_q_value = current_q.mean().item()
        first_action = action_indices[0].item() if len(action_indices) > 0 else 0
        self.update_stats(
            reward=rewards.mean().item(),
            action=first_action,
            loss=loss.item(),
            q_values=torch.tensor([avg_q_value])
        )
        
        return {
            "loss": loss.item(),
            "q_value": avg_q_value,
            "td_error": td_errors.mean().item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
            "epsilon": self.epsilon
        }
    
    def compute_td_error(self, state, action, reward, next_state, done) -> float:
        """计算TD误差用于优先级更新"""
        with torch.no_grad():
            if isinstance(state, Data):
                state_emb = self.gnn_encoder(state.to(self.device))
                next_state_emb = self.gnn_encoder(next_state.to(self.device))
            else:
                state_emb = torch.tensor(state, device=self.device).unsqueeze(0)
                next_state_emb = torch.tensor(next_state, device=self.device).unsqueeze(0)
            
            current_q = self.policy_network(state_emb)[0, action]
            
            if not done:
                next_action = self.policy_network(next_state_emb).argmax(dim=1)
                next_q = self.target_network(next_state_emb).gather(1, next_action.unsqueeze(1)).squeeze()
                target_q = reward + self.gamma * next_q
            else:
                target_q = torch.tensor(reward, device=self.device)
            
            td_error = abs(current_q - target_q).item()
        
        return td_error
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'gnn_encoder': self.gnn_encoder.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }, filepath)
        print(f"💾 DDQN模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        
        print(f"📂 DDQN模型已加载: {filepath}")


# 测试函数
def test_ddqn_agent():
    """测试DDQN智能体"""
    print("🧪 测试DDQN智能体...")
    
    config = {
        "gnn": {"hidden_dim": 64, "output_dim": 128},
        "train": {"lr": 0.001, "gamma": 0.99, "batch_size": 16, "target_update": 10},
        "network": {"hidden_dim": 256}
    }
    
    agent = MultiDDQNAgent("test_ddqn", state_dim=8, action_dim=10, edge_dim=4, config=config)
    
    # 测试动作选择
    test_state = torch.randn(1, 128)
    action = agent.select_action(test_state)
    print(f"✅ 动作选择测试: {action}")
    
    # 测试经验存储
    agent.store_transition(test_state, action, 1.0, test_state, False)
    print(f"✅ 经验存储测试: 缓冲区大小 {len(agent.replay_buffer)}")
    
    # 添加更多经验以测试学习
    for i in range(20):
        state = torch.randn(1, 128)
        action = i % 10
        reward = np.random.random()
        next_state = torch.randn(1, 128)
        done = False
        agent.store_transition(state, action, reward, next_state, done)
    
    # 测试学习
    learning_info = agent.learn()
    print(f"✅ 学习测试: Loss={learning_info['loss']:.4f}, Q值={learning_info['q_value']:.4f}")
    
    print("✅ DDQN智能体测试完成!")


if __name__ == "__main__":
    test_ddqn_agent()