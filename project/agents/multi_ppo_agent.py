# agents/multi_ppo_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Any, Tuple
from torch_geometric.data import Data
from collections import namedtuple
from agents.enhanced_base_agent import EnhancedBaseAgent

# PPO经验数据结构
PPOExperience = namedtuple('PPOExperience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'log_prob', 'value', 'advantage', 'return_'
])

class PPONetwork(nn.Module):
    """
    PPO网络 - Actor-Critic架构
    
    包含：
    1. Actor网络：输出动作概率分布
    2. Critic网络：输出状态价值函数
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        # 处理 shared_layers（nn.Sequential）
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        # 处理 actor 和 critic（nn.Linear）
        for layer in [self.actor, self.critic]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state_embedding: GNN编码后的状态 [batch_size, input_dim]
            
        Returns:
            logits: 动作logits [batch_size, action_dim]
            values: 状态价值 [batch_size, 1]
        """
        shared_features = self.shared_layers(state_embedding)
        logits = self.actor(shared_features)
        values = self.critic(shared_features)
        return logits, values
    
    def get_action_and_value(self, state_embedding: torch.Tensor, action: torch.Tensor = None):
        """
        获取动作概率和价值
        
        Args:
            state_embedding: 状态编码
            action: 动作（可选，用于计算特定动作的log概率）
            
        Returns:
            action: 采样的动作
            log_prob: 动作的log概率
            entropy: 策略熵
            value: 状态价值
        """
        logits, values = self.forward(state_embedding)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_log_prob, entropy, values.squeeze(-1)

class MultiPPOAgent(EnhancedBaseAgent):
    """
    多智能体近端策略优化（PPO）智能体
    
    特性：
    1. Actor-Critic架构
    2. 裁剪策略损失防止过大更新
    3. 广义优势估计（GAE）
    4. 使用GNN编码器处理图状态（支持增强GNN）
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, edge_dim: int, config: Dict[str, Any], use_enhanced_gnn: bool = False):
        super().__init__(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
        
        # PPO特定配置
        self.clip_epsilon = config.get("train", {}).get("eps_clip", 0.2)
        self.entropy_coef = config.get("train", {}).get("entropy_coef", 0.01)
        self.value_coef = config.get("train", {}).get("value_coef", 0.5)
        self.max_grad_norm = config.get("train", {}).get("max_grad_norm", 0.5)
        self.gae_lambda = config.get("train", {}).get("gae_lambda", 0.95)
        self.ppo_epochs = config.get("train", {}).get("ppo_epochs", 4)
        self.mini_batch_size = config.get("train", {}).get("mini_batch_size", 64)
        
        # 网络架构
        network_input_dim = self.output_dim
        
        # PPO网络
        self.policy_network = PPONetwork(
            input_dim=network_input_dim,
            action_dim=action_dim,
            hidden_dim=config.get("network", {}).get("hidden_dim", 512)
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )
        
        # 经验存储
        self.experiences = []
        self.rollout_length = config.get("train", {}).get("rollout_length", 128)
        
        # PPO统计
        self.ppo_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": []
        }
        
        print(f"🚀 PPO Agent {agent_id} 初始化完成（增强模式: {use_enhanced_gnn}）")
        print(f"   - 状态维度: {state_dim} -> GNN编码 -> {network_input_dim}")
        print(f"   - 动作维度: {action_dim}")
        print(f"   - 裁剪参数: {self.clip_epsilon}")
        print(f"   - 设备: {self.device}")
    
    def select_action(self, state: Union[Data, Dict], valid_actions: List[int] = None, **kwargs) -> int:
        state_embedding = self.process_state(state)
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        
        if valid_actions is None:
            valid_actions = self.get_valid_actions(state, **kwargs)
        
        self.policy_network.eval()
        with torch.no_grad():
            logits, values = self.policy_network(state_embedding)
            masked_logits = self._mask_invalid_logits(logits, valid_actions)
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            if self.is_training:
                log_prob = dist.log_prob(action)
                self._store_action_info(state, action.item(), log_prob.item(), values.item())
        
        if self.is_training:
            self.policy_network.train()
        
        return action.item()
    
    def _mask_invalid_logits(self, logits: torch.Tensor, valid_actions: List[int]) -> torch.Tensor:
        """屏蔽无效动作的logits"""
        masked_logits = logits.clone()
        invalid_actions = [a for a in range(self.action_dim) if a not in valid_actions]
        if invalid_actions:
            masked_logits[:, invalid_actions] = -float('inf')
        return masked_logits
    
    def _store_action_info(self, state, action: int, log_prob: float, value: float):
        """存储动作选择时的信息"""
        self._last_action_info = {
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'value': value
        }
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """获取VNF嵌入场景下的有效动作"""
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
        存储经验（PPO使用on-policy学习）
        """
        if hasattr(self, '_last_action_info'):
            log_prob = self._last_action_info['log_prob']
            value = self._last_action_info['value']
        else:
            with torch.no_grad():
                if isinstance(state, Data):
                    state_emb = self.process_state(state)
                else:
                    state_emb = torch.tensor(state, device=self.device).unsqueeze(0)
                
                _, log_prob, _, value = self.policy_network.get_action_and_value(
                    state_emb, torch.tensor([action], device=self.device)
                )
                log_prob = log_prob.item()
                value = value.item()
        
        experience = PPOExperience(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
            log_prob=log_prob,
            value=value,
            advantage=0.0,
            return_=0.0
        )
        
        self.experiences.append(experience)
    
    def _compute_gae(self):
        """计算广义优势估计（GAE）"""
        if len(self.experiences) == 0:
            return
        
        returns = []
        advantages = []
        
        if not self.experiences[-1].done:
            with torch.no_grad():
                if isinstance(self.experiences[-1].next_state, Data):
                    last_state_emb = self.process_state(self.experiences[-1].next_state)
                else:
                    last_state_emb = torch.tensor(self.experiences[-1].next_state, device=self.device).unsqueeze(0)
                
                _, last_value = self.policy_network(last_state_emb)
                last_value = last_value.item()
        else:
            last_value = 0.0
        
        gae = 0.0
        for i in reversed(range(len(self.experiences))):
            exp = self.experiences[i]
            
            if i == len(self.experiences) - 1:
                next_value = last_value
            else:
                next_value = self.experiences[i + 1].value
            
            delta = exp.reward + self.gamma * next_value * (1 - exp.done) - exp.value
            gae = delta + self.gamma * self.gae_lambda * (1 - exp.done) * gae
            return_ = gae + exp.value
            
            returns.insert(0, return_)
            advantages.insert(0, gae)
        
        advantages_tensor = torch.tensor(advantages, device=self.device)
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            advantages = advantages_tensor.tolist()
        
        for i, exp in enumerate(self.experiences):
            self.experiences[i] = exp._replace(
                advantage=advantages[i],
                return_=returns[i]
            )
    
    def update_stats(self, reward: float, action: int, loss: float, **kwargs):
        """
        更新 PPO 智能体的统计信息
        """
        super().update_stats(reward=reward, action=action, loss=loss)
        
        if not hasattr(self, 'ppo_stats'):
            self.ppo_stats = {
                "policy_loss": [],
                "value_loss": [],
                "entropy": []
            }
        
        if "policy_loss" in kwargs:
            self.ppo_stats["policy_loss"].append(kwargs["policy_loss"])
        if "value_loss" in kwargs:
            self.ppo_stats["value_loss"].append(kwargs["value_loss"])
        if "entropy" in kwargs:
            self.ppo_stats["entropy"].append(kwargs["entropy"])
    
    def learn(self) -> Dict[str, float]:
        if len(self.experiences) < self.mini_batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        self._compute_gae()
        
        states = [exp.state for exp in self.experiences]
        actions = [exp.action for exp in self.experiences]
        old_log_probs = [exp.log_prob for exp in self.experiences]
        advantages = [exp.advantage for exp in self.experiences]
        returns = [exp.return_ for exp in self.experiences]
        
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        batch_size = len(self.experiences)
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            
            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                batch_states = [states[i] for i in batch_indices.tolist()]
                if all(isinstance(s, Data) for s in batch_states):
                    from torch_geometric.data import Batch
                    states_batch = Batch.from_data_list(batch_states).to(self.device)
                    batch_state_embeddings = self.gnn_encoder(states_batch)
                else:
                    states_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s 
                                            for s in batch_states]).to(self.device)
                    batch_state_embeddings = states_tensor
                
                _, new_log_probs, entropy, values = self.policy_network.get_action_and_value(
                    batch_state_embeddings, batch_actions
                )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, batch_returns)
                
                total_loss = (policy_loss + 
                            self.value_coef * value_loss - 
                            self.entropy_coef * entropy.mean())
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
                    max_norm=self.max_grad_norm
                )
                
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        self.lr_scheduler.step()
        self.training_step += 1
        
        avg_reward = np.mean([exp.reward for exp in self.experiences])
        first_action = actions[0] if actions else 0
        self.experiences.clear()
        
        self.update_stats(
            reward=avg_reward,
            action=first_action,
            loss=total_policy_loss,
            policy_loss=total_policy_loss,
            value_loss=total_value_loss,
            entropy=total_entropy
        )
        
        return {
            "loss": total_policy_loss / max(self.ppo_epochs, 1),
            "policy_loss": total_policy_loss / max(self.ppo_epochs, 1),
            "value_loss": total_value_loss / max(self.ppo_epochs, 1),
            "entropy": total_entropy / max(self.ppo_epochs, 1),
            "lr": self.lr_scheduler.get_last_lr()[0]
        }
    
    def should_update(self) -> bool:
        """判断是否应该进行PPO更新"""
        return len(self.experiences) >= self.rollout_length
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'gnn_encoder': self.gnn_encoder.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
        }, filepath)
        print(f"💾 PPO模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        print(f"📂 PPO模型已加载: {filepath}")

def test_ppo_agent():
    print("🧪 测试PPO智能体（增强版）...")
    
    from config_loader import get_scenario_config
    config = get_scenario_config('normal_operation')
    
    agent = MultiPPOAgent(
        agent_id="test_ppo_enhanced",
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
    valid_actions = list(range(config['topology']['node_counts']['total']))
    action = agent.select_action(test_state, valid_actions=valid_actions)
    print(f"✅ 动作选择测试: {action}")
    
    # 测试经验存储和学习
    for i in range(40):
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
        done = (i % 20 == 0)
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if agent.should_update():
            learning_info = agent.learn()
            print(f"✅ PPO学习更新: Policy Loss={learning_info['policy_loss']:.4f}, "
                  f"Value Loss={learning_info['value_loss']:.4f}, "
                  f"Entropy={learning_info['entropy']:.4f}")
            break
    
    print("✅ 增强PPO智能体测试完成!")