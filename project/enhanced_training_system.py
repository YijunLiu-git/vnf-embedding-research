
import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import argparse

# 导入现有组件
from env.enhanced_vnf_env_multi import EnhancedVNFEmbeddingEnv  # 使用您现有的增强环境
from env.topology_loader import generate_topology
from agents.base_agent import create_agent
from utils.logger import Logger
from config_loader import get_scenario_config, load_config

class SafeEnhancedTrainer:
    """
    安全的增强训练器 - 避免维度冲突
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        print(f"🛡️ 初始化安全增强训练系统...")
        
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.episodes = self.config['train']['episodes']
        self.agent_types = ['ddqn', 'dqn', 'ppo']
        
        self.results_dir = f"safe_enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.current_scenario = "normal_operation"
        
        # 设置组件
        self._setup_safe_components()
        
        print(f"✅ 安全增强训练系统初始化完成")
    
    def _setup_safe_components(self):
        """安全设置组件"""
        
        # 1. 生成拓扑
        full_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'dimensions': self.config['dimensions']
        }
        self.graph, self.node_features, self.edge_features = generate_topology(config=full_config)
        
        # 2. 创建环境
        reward_config = self.config['reward']
        chain_length_range = tuple(self.config['vnf_requirements']['chain_length_range'])
        
        env_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'reward': self.config['reward'],
            'train': self.config['train'],
            'dimensions': self.config['dimensions']
        }
        
        # 使用您现有的增强环境
        self.env_edge_aware = EnhancedVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=reward_config,
            chain_length_range=chain_length_range,
            config=env_config.copy()
        )
        
        self.env_baseline = EnhancedVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=reward_config,
            chain_length_range=chain_length_range,
            config=env_config.copy()
        )
        self.env_baseline.is_baseline_mode = True
        
        # 3. 创建智能体（不替换编码器，使用标准版本）
        expected_node_dim = self.config['dimensions']['node_feature_dim']
        action_dim = len(self.graph.nodes())
        
        print(f"🤖 创建安全智能体:")
        
        # Edge-aware智能体
        self.agents_edge_aware = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_safe_edge_aware"
            edge_dim = self.config['gnn']['edge_aware']['edge_dim']
            
            agent = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=expected_node_dim,
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
            
            self.agents_edge_aware[agent_type] = agent
            print(f"   ✅ {agent_id}: 标准GNN编码器")
        
        # Baseline智能体
        self.agents_baseline = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_safe_baseline"
            edge_dim = self.config['gnn']['baseline']['edge_dim']
            
            self.agents_baseline[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=expected_node_dim,
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
            print(f"   ✅ {agent_id}: 标准GNN编码器")
        
        # 4. 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.loggers = {}
        
        for agent_type in self.agent_types:
            self.loggers[f"{agent_type}_edge_aware"] = Logger(
                log_dir=os.path.join(self.results_dir, f"safe_{agent_type}_edge_aware_{timestamp}")
            )
            self.loggers[f"{agent_type}_baseline"] = Logger(
                log_dir=os.path.join(self.results_dir, f"safe_{agent_type}_baseline_{timestamp}")
            )
    
    def train_safe_episode(self, agent, env, agent_id: str) -> Dict[str, Any]:
        """安全训练单个episode"""
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        success = False
        info = {}
        
        max_steps = 20
        
        while step_count < max_steps:
            try:
                # 获取有效动作
                if hasattr(env, 'get_enhanced_valid_actions'):
                    valid_actions = env.get_enhanced_valid_actions()
                else:
                    valid_actions = env.get_valid_actions()
                
                if not valid_actions:
                    info = {'success': False, 'reason': 'no_valid_actions'}
                    break
                
                # 选择动作
                action = agent.select_action(state, valid_actions=valid_actions)
                if action not in valid_actions:
                    action = np.random.choice(valid_actions)
                
                # 执行动作
                next_state, reward, done, step_info = env.step(action)
                
                # 存储经验
                agent.store_transition(state, action, reward, next_state, done)
                
                # 更新状态
                state = next_state
                total_reward += reward
                step_count += 1
                
                # 学习更新
                try:
                    if hasattr(agent, 'should_update') and agent.should_update():
                        learning_info = agent.learn()
                    elif hasattr(agent, 'replay_buffer') and len(getattr(agent, 'replay_buffer', [])) >= 16:
                        learning_info = agent.learn()
                except Exception as e:
                    print(f"⚠️ 学习更新失败: {e}")
                
                if done:
                    success = step_info.get('success', False)
                    info = step_info
                    break
                    
            except Exception as e:
                print(f"⚠️ 步骤执行失败: {e}")
                break
        
        # 计算统计
        sar = 1.0 if success else 0.0
        splat = info.get('splat', info.get('avg_delay', float('inf'))) if success else float('inf')
        
        return {
            'total_reward': total_reward,
            'steps': step_count,
            'success': success,
            'sar': sar,
            'splat': splat,
            'info': info,
            'scenario': self.current_scenario
        }
    
    def train_safe(self):
        """安全训练主循环"""
        print(f"\n🛡️ 开始安全增强训练...")
        
        performance_results = {
            'edge_aware': {agent: {'sar': [], 'splat': [], 'rewards': []} for agent in self.agent_types},
            'baseline': {agent: {'sar': [], 'splat': [], 'rewards': []} for agent in self.agent_types}
        }
        
        # 应用初始场景
        initial_scenario_config = get_scenario_config(1)
        self.env_edge_aware.apply_scenario_config(initial_scenario_config)
        self.env_baseline.apply_scenario_config(initial_scenario_config)
        
        for episode in range(1, self.episodes + 1):
            # 更新场景
            if episode <= 25:
                new_scenario = "normal_operation"
            elif episode <= 50:
                new_scenario = "peak_congestion"
            elif episode <= 75:
                new_scenario = "failure_recovery"
            else:
                new_scenario = "extreme_pressure"
            
            if new_scenario != self.current_scenario:
                print(f"🎯 场景切换: {self.current_scenario} → {new_scenario}")
                self.current_scenario = new_scenario
                scenario_config = get_scenario_config(episode)
                self.env_edge_aware.apply_scenario_config(scenario_config)
                self.env_baseline.apply_scenario_config(scenario_config)
            
            print(f"Episode {episode}/{self.episodes} - {self.current_scenario}")
            
            # 训练Edge-aware智能体
            for agent_type in self.agent_types:
                agent = self.agents_edge_aware[agent_type]
                env = self.env_edge_aware
                episode_stats = self.train_safe_episode(agent, env, f"{agent_type}_edge_aware")
                
                performance_results['edge_aware'][agent_type]['sar'].append(episode_stats['sar'])
                performance_results['edge_aware'][agent_type]['splat'].append(episode_stats['splat'])
                performance_results['edge_aware'][agent_type]['rewards'].append(episode_stats['total_reward'])
                
                # 记录日志
                if f"{agent_type}_edge_aware" in self.loggers:
                    self.loggers[f"{agent_type}_edge_aware"].log_episode(episode, episode_stats)
            
            # 训练Baseline智能体
            for agent_type in self.agent_types:
                agent = self.agents_baseline[agent_type]
                env = self.env_baseline
                episode_stats = self.train_safe_episode(agent, env, f"{agent_type}_baseline")
                
                performance_results['baseline'][agent_type]['sar'].append(episode_stats['sar'])
                performance_results['baseline'][agent_type]['splat'].append(episode_stats['splat'])
                performance_results['baseline'][agent_type]['rewards'].append(episode_stats['total_reward'])
                
                # 记录日志
                if f"{agent_type}_baseline" in self.loggers:
                    self.loggers[f"{agent_type}_baseline"].log_episode(episode, episode_stats)
            
            # 打印进度
            if episode % 10 == 0:
                print(f"📊 Episode {episode} 进度报告:")
                for variant in ['edge_aware', 'baseline']:
                    recent_sar = np.mean([performance_results[variant][agent]['sar'][-5:] for agent in self.agent_types])
                    print(f"  {variant} 平均SAR (最近5轮): {recent_sar:.3f}")
        
        print(f"\n✅ 安全增强训练完成!")
        print(f"📁 结果保存在: {self.results_dir}")
        
        return performance_results

def main_safe():
    """安全训练主函数"""
    parser = argparse.ArgumentParser(description='安全增强Edge-Aware训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    args = parser.parse_args()
    
    trainer = SafeEnhancedTrainer(config_path=args.config)
    
    if args.episodes:
        trainer.episodes = args.episodes
    
    results = trainer.train_safe()
    
    print(f"\n🎉 安全训练完成!")
    return results

if __name__ == "__main__":
    main_safe()