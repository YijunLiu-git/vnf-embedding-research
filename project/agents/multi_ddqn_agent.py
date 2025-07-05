# agents/multi_ddqn_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Any
from torch_geometric.data import Data
from agents.enhanced_base_agent import EnhancedBaseAgent
from utils.replay_buffer import PrioritizedReplayBuffer

class DDQNNetwork(nn.Module):
    """
    双深度Q网络 - 策略网络和目标网络的共同结构
    
    架构：
    输入（GNN编码后的状态嵌入） -> 全连接层 -> Q值输出
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
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

class MultiDDQNAgent(EnhancedBaseAgent):
    """
    多智能体双深度Q学习智能体
    
    特性：
    1. 使用GNN编码器处理图状态（支持增强GNN）
    2. 双Q网络减少过估计
    3. 优先级经验回放
    4. VNF嵌入专用的动作选择逻辑
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, edge_dim: int, config: Dict[str, Any], use_enhanced_gnn: bool = False):
        super().__init__(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
        
        # DDQN特定配置
        self.target_update_freq = config.get("train", {}).get("target_update", 100)
        self.double_q = True
        
        # 网络架构
        network_input_dim = self.output_dim
        
        # 策略网络
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
            alpha=0.6,
            beta=0.4
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )
        
        print(f"🚀 DDQN Agent {agent_id} 初始化完成（增强模式: {use_enhanced_gnn}）")
        print(f"   - 状态维度: {state_dim} -> GNN编码 -> {network_input_dim}")
        print(f"   - 动作维度: {action_dim}")
        print(f"   - 设备: {self.device}")
    
    def select_action(self, state: Union[Data, Dict], valid_actions: List[int] = None, **kwargs) -> int:
        state_embedding = self.process_state(state)
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        
        if valid_actions is None:
            valid_actions = self.get_valid_actions(state, **kwargs)
        
        if self.is_training and np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            self.policy_network.eval()
            with torch.no_grad():
                q_values = self.policy_network(state_embedding)
                masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
                action = masked_q_values.argmax(dim=1).item()
            if self.is_training:
                self.policy_network.train()
        
        return action
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """
        获取VNF嵌入场景下的有效动作
        """
        available_nodes = kwargs.get('available_nodes', list(range(self.action_dim)))
        resource_constraints = kwargs.get('resource_constraints', {})
        
        valid_actions = []
        for node in available_nodes:
            if self._check_node_feasibility(node, resource_constraints):
                valid_actions.append(node)
        
        if not valid_actions:
            valid_actions = [0]
        
        return valid_actions
    
    def _check_node_feasibility(self, node_id: int, constraints: Dict) -> bool:
        """检查节点是否满足VNF嵌入约束"""
        return True
    
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        """
        存储经验到优先级回放缓冲区
        """
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
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_value": 0.0}
        
        batch_data = self.replay_buffer.sample(self.batch_size, device=self.device)
        states, actions, rewards, next_states, dones, weights, indices = batch_data
        
        if isinstance(states, Data):
            state_embeddings = self.gnn_encoder(states)
            next_state_embeddings = self.gnn_encoder(next_states)
        else:
            state_embeddings = states
            next_state_embeddings = next_states
        
        current_q_values = self.policy_network(state_embeddings)
        
        if isinstance(actions, torch.Tensor):
            action_indices = actions
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            action_indices = torch.tensor([a if isinstance(a, int) else a[0] for a in actions], 
                                        dtype=torch.long, device=self.device)
            current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values_policy = self.policy_network(next_state_embeddings)
            next_actions = next_q_values_policy.argmax(dim=1, keepdim=True)
            next_q_values_target = self.target_network(next_state_embeddings)
            next_q = next_q_values_target.gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        td_errors = torch.abs(current_q - target_q)
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        self.decay_epsilon()
        
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

def test_ddqn_agent():
    print("🧪 测试DDQN智能体（增强版）...")
    
    from config_loader import get_scenario_config
    config = get_scenario_config('normal_operation')
    
    agent = MultiDDQNAgent(
        agent_id="test_ddqn_enhanced",
        state_dim=config['dimensions']['node_feature_dim'],
        action_dim=config['topology']['node_counts']['total'],
        edge_dim=config['dimensions']['edge_feature_dim_full'],
        config=config,
        use_enhanced_gnn=True
    )
    
    # 测试动作选择
    test_state = Data(
        x=torch.randn(config['topology']['node_counts']['total'], config['dimensions']['node_feature_dim']),
        edge_index=torch.randint(0, config['topology']['node_counts']['total'], (2, 100)),
        edge_attr=torch.randn(100, config['dimensions']['edge_feature_dim_full']),
        vnf_context=torch.randn(config['dimensions']['vnf_context_dim'])
    )
    action = agent.select_action(test_state)
    print(f"✅ 动作选择测试: {action}")
    
    # 测试经验存储和学习
    for i in range(20):
        state = Data(
            x=torch.randn(config['topology']['node_counts']['total'], config['dimensions']['node_feature_dim']),
            edge_index=torch.randint(0, config['topology']['node_counts']['total'], (2, 100)),
            edge_attr=torch.randn(100, config['dimensions']['edge_feature_dim_full']),
            vnf_context=torch.randn(config['dimensions']['vnf_context_dim'])
        )
        action = i % config['topology']['node_counts']['total']
        reward = np.random.random()
        next_state = Data(
            x=torch.randn(config['topology']['node_counts']['total'], config['dimensions']['node_feature_dim']),
            edge_index=torch.randint(0, config['topology']['node_counts']['total'], (2, 100)),
            edge_attr=torch.randn(100, config['dimensions']['edge_feature_dim_full']),
            vnf_context=torch.randn(config['dimensions']['vnf_context_dim'])
        )
        done = False
        agent.store_transition(state, action, reward, next_state, done)
    
    learning_info = agent.learn()
    print(f"✅ 学习测试: Loss={learning_info['loss']:.4f}, Q值={learning_info['q_value']:.4f}")
    print("✅ 增强DDQN智能体测试完成!")