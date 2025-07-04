# main_quiet.py - 精简版主训练脚本，大幅减少输出

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import argparse

from env.vnf_env_multi import MultiVNFEmbeddingEnv
from env.topology_loader import generate_topology
from agents.base_agent import create_agent
from config_loader import get_scenario_config, load_config

class QuietTrainer:
    """
    精简版多智能体训练器
    
    特点：
    - 最少输出信息
    - 只显示关键进度
    - 专注于结果
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.episodes = self.config['train']['episodes']
        self.agent_types = ['ddqn', 'dqn', 'ppo']
        
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 场景信息
        self.scenarios = {
            'normal_operation': {'name': '正常运营期', 'range': [1, 25]},
            'peak_congestion': {'name': '高峰拥塞期', 'range': [26, 50]},
            'failure_recovery': {'name': '故障恢复期', 'range': [51, 75]},
            'extreme_pressure': {'name': '极限压力期', 'range': [76, 100]}
        }
        
        self._setup_environments()
        self._setup_agents()
        
        print(f"🚀 训练器初始化完成 | Episodes: {self.episodes} | 智能体: {len(self.agent_types)} × 2种模式")
    
    def _setup_environments(self):
        """设置环境"""
        print("🌐 初始化环境...", end="")
        
        full_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'dimensions': self.config['dimensions']
        }
        
        self.graph, self.node_features, self.edge_features = generate_topology(config=full_config)
        
        env_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'reward': self.config['reward'],
            'train': self.config['train'],
            'dimensions': self.config['dimensions']
        }
        
        # Edge-aware环境
        self.env_edge_aware = MultiVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=self.config['reward'],
            chain_length_range=tuple(self.config['vnf_requirements']['chain_length_range']),
            config=env_config.copy()
        )
        
        # Baseline环境
        self.env_baseline = MultiVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=self.config['reward'],
            chain_length_range=tuple(self.config['vnf_requirements']['chain_length_range']),
            config=env_config.copy()
        )
        self.env_baseline.is_baseline_mode = True
        
        print(" ✅")
    
    def _setup_agents(self):
        """设置智能体"""
        print("🤖 初始化智能体...", end="")
        
        state_dim = self.config['dimensions']['node_feature_dim']
        action_dim = len(self.graph.nodes())
        
        # Edge-aware智能体
        self.agents_edge_aware = {}
        for agent_type in self.agent_types:
            self.agents_edge_aware[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=f"{agent_type}_edge_aware",
                state_dim=state_dim,
                action_dim=action_dim,
                edge_dim=self.config['gnn']['edge_aware']['edge_dim'],
                config=self.config
            )
        
        # Baseline智能体
        self.agents_baseline = {}
        for agent_type in self.agent_types:
            self.agents_baseline[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=f"{agent_type}_baseline",
                state_dim=state_dim,
                action_dim=action_dim,
                edge_dim=self.config['gnn']['baseline']['edge_dim'],
                config=self.config
            )
        
        print(" ✅")
    
    def _get_current_scenario(self, episode: int) -> str:
        """获取当前场景"""
        if episode <= 25:
            return 'normal_operation'
        elif episode <= 50:
            return 'peak_congestion'
        elif episode <= 75:
            return 'failure_recovery'
        else:
            return 'extreme_pressure'
    
    def _train_episode(self, agent, env, agent_id: str) -> Dict[str, Any]:
        """训练单个episode - 静默版"""
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        success = False
        info = {}
        
        max_steps = getattr(env, 'max_episode_steps', 20)
        
        while step_count < max_steps:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action = agent.select_action(state, valid_actions=valid_actions)
            if action not in valid_actions:
                action = np.random.choice(valid_actions)
            
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 静默学习
            try:
                if hasattr(agent, 'should_update') and agent.should_update():
                    agent.learn()
            except:
                pass
            
            if done:
                success = info.get('success', False)
                break
        
        return {
            'reward': total_reward,
            'success': success,
            'sar': 1.0 if success else 0.0,
            'scenario': getattr(env, 'current_scenario_name', 'unknown')
        }
    
    def train(self):
        """主训练循环 - 精简版"""
        print(f"\n🎯 开始训练 | 目标: {self.episodes} episodes")
        print("=" * 60)
        
        # 结果存储
        results = {
            'edge_aware': {agent: {'rewards': [], 'sar': [], 'scenarios': []} 
                          for agent in self.agent_types},
            'baseline': {agent: {'rewards': [], 'sar': [], 'scenarios': []} 
                        for agent in self.agent_types}
        }
        
        current_scenario = None
        
        for episode in range(1, self.episodes + 1):
            # 检查场景切换
            new_scenario = self._get_current_scenario(episode)
            if new_scenario != current_scenario:
                current_scenario = new_scenario
                scenario_info = self.scenarios[current_scenario]
                
                print(f"\n🎯 Episode {episode}: {scenario_info['name']}")
                
                # 应用场景配置
                scenario_config = get_scenario_config(episode)
                self.env_edge_aware.apply_scenario_config(scenario_config)
                self.env_baseline.apply_scenario_config(scenario_config)
            
            # 训练所有智能体 - 静默模式
            for agent_type in self.agent_types:
                # Edge-aware
                stats = self._train_episode(
                    self.agents_edge_aware[agent_type], 
                    self.env_edge_aware, 
                    f"{agent_type}_edge_aware"
                )
                results['edge_aware'][agent_type]['rewards'].append(stats['reward'])
                results['edge_aware'][agent_type]['sar'].append(stats['sar'])
                results['edge_aware'][agent_type]['scenarios'].append(stats['scenario'])
                
                # Baseline
                stats = self._train_episode(
                    self.agents_baseline[agent_type], 
                    self.env_baseline, 
                    f"{agent_type}_baseline"
                )
                results['baseline'][agent_type]['rewards'].append(stats['reward'])
                results['baseline'][agent_type]['sar'].append(stats['sar'])
                results['baseline'][agent_type]['scenarios'].append(stats['scenario'])
            
            # 每25个episode显示进度
            if episode % 25 == 0:
                self._print_progress(episode, current_scenario, results)
        
        # 最终分析
        self._final_analysis(results)
        return results
    
    def _print_progress(self, episode: int, scenario: str, results: Dict):
        """精简进度显示"""
        scenario_name = self.scenarios[scenario]['name']
        window = 25
        start_idx = max(0, episode - window)
        
        print(f"\n📊 Episode {episode} | {scenario_name}")
        print("-" * 40)
        
        for variant in ['edge_aware', 'baseline']:
            avg_sar = np.mean([
                np.mean(results[variant][agent]['sar'][start_idx:episode]) 
                for agent in self.agent_types
            ])
            avg_reward = np.mean([
                np.mean(results[variant][agent]['rewards'][start_idx:episode]) 
                for agent in self.agent_types
            ])
            
            print(f"{variant.upper():11} | SAR: {avg_sar:.3f} | Reward: {avg_reward:.1f}")
    
    def _final_analysis(self, results: Dict):
        """最终结果分析"""
        print(f"\n🎯 最终结果汇总")
        print("=" * 60)
        
        summary_data = []
        
        for scenario_key, scenario_info in self.scenarios.items():
            print(f"\n📋 {scenario_info['name']}:")
            
            # 计算该场景的episode范围
            start_ep, end_ep = scenario_info['range']
            episode_indices = list(range(start_ep-1, end_ep))  # 转换为0-based索引
            
            for variant in ['edge_aware', 'baseline']:
                # 获取该场景的平均性能
                scenario_sar = []
                scenario_rewards = []
                
                for agent_type in self.agent_types:
                    agent_sar = [results[variant][agent_type]['sar'][i] for i in episode_indices 
                               if i < len(results[variant][agent_type]['sar'])]
                    agent_rewards = [results[variant][agent_type]['rewards'][i] for i in episode_indices 
                                   if i < len(results[variant][agent_type]['rewards'])]
                    
                    if agent_sar:  # 如果有数据
                        scenario_sar.extend(agent_sar)
                        scenario_rewards.extend(agent_rewards)
                
                if scenario_sar:
                    avg_sar = np.mean(scenario_sar)
                    avg_reward = np.mean(scenario_rewards)
                    
                    print(f"  {variant.upper():11} | SAR: {avg_sar:.3f} | Reward: {avg_reward:.1f}")
                    
                    summary_data.append({
                        'Scenario': scenario_info['name'],
                        'Variant': variant.upper(),
                        'SAR': avg_sar,
                        'Reward': avg_reward
                    })
        
        # 保存结果
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.results_dir, 'summary_results.csv'), index=False)
        
        print(f"\n💾 结果已保存: {os.path.join(self.results_dir, 'summary_results.csv')}")
        
        # Edge-aware vs Baseline 比较
        print(f"\n🆚 Edge-aware vs Baseline 比较:")
        for scenario_key, scenario_info in self.scenarios.items():
            edge_data = [d for d in summary_data if d['Scenario'] == scenario_info['name'] and d['Variant'] == 'EDGE_AWARE']
            baseline_data = [d for d in summary_data if d['Scenario'] == scenario_info['name'] and d['Variant'] == 'BASELINE']
            
            if edge_data and baseline_data:
                edge_sar = edge_data[0]['SAR']
                baseline_sar = baseline_data[0]['SAR']
                improvement = ((edge_sar - baseline_sar) / max(baseline_sar, 0.001)) * 100
                
                status = "🟢" if improvement > 5 else "🟡" if improvement > -5 else "🔴"
                print(f"  {scenario_info['name']:8} | {status} {improvement:+5.1f}% SAR提升")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VNF嵌入精简版训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    args = parser.parse_args()
    
    trainer = QuietTrainer(config_path=args.config)
    
    if args.episodes:
        trainer.episodes = args.episodes
    
    results = trainer.train()
    
    print(f"\n✅ 训练完成!")
    print(f"📁 结果保存在: {trainer.results_dir}")

if __name__ == "__main__":
    main()