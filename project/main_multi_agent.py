# main_multi_agent.py

import os
import yaml
import torch
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
import argparse

# 导入修复后的组件
from env.vnf_env_multi import MultiVNFEmbeddingEnv
from env.topology_loader import generate_topology
from agents.base_agent import create_agent
from utils.logger import Logger
from utils.metrics import calculate_sar, calculate_splat
from utils.visualization import plot_training_curves

class MultiAgentTrainer:
    """
    多智能体VNF嵌入训练器
    
    支持：
    1. 多种算法同时训练和对比（DDQN, DQN, PPO）
    2. Edge-aware vs Baseline对比
    3. 完整的实验记录和可视化
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")
        
        # 实验配置
        self.episodes = self.config['train']['episodes']
        self.save_interval = 50
        self.eval_interval = 25
        
        # 智能体类型
        self.agent_types = ['ddqn', 'dqn', 'ppo']
        
        # 创建结果目录
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化网络拓扑
        self._setup_network_topology()
        
        # 初始化环境和智能体
        self._setup_environments()
        self._setup_agents()
        
        # 初始化日志记录
        self._setup_logging()
        
        print(f"✅ 多智能体训练器初始化完成")
        print(f"   - 智能体类型: {self.agent_types}")
        print(f"   - 训练轮数: {self.episodes}")
        print(f"   - 网络节点: {len(self.graph.nodes())}")
    
    def _setup_network_topology(self):
        """设置网络拓扑"""
        topology_config = self.config.get('topology', {})
        
        if topology_config.get('type') == 'fat-tree':
            # Fat-tree拓扑
            k = topology_config.get('k', 4)
            self.graph = self._create_fat_tree(k)
        else:
            # 随机拓扑
            num_nodes = topology_config.get('num_nodes', 20)
            prob = topology_config.get('prob', 0.3)
            self.graph, self.node_features, self.edge_features = generate_topology(num_nodes, prob)
            return
        
        # 为Fat-tree生成特征
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        
        # 节点特征：[CPU, Memory, Storage, Available_Bandwidth]
        self.node_features = np.random.rand(num_nodes, 4)
        self.node_features[:, 0] = self.node_features[:, 0] * 0.8 + 0.2  # CPU: 0.2-1.0
        self.node_features[:, 1] = self.node_features[:, 1] * 0.8 + 0.2  # Memory: 0.2-1.0
        self.node_features[:, 2] = self.node_features[:, 2] * 0.6 + 0.4  # Storage: 0.4-1.0
        self.node_features[:, 3] = self.node_features[:, 3] * 50 + 50    # Bandwidth: 50-100
        
        # 边特征：[Available_Bandwidth, Delay, Jitter, Loss_Rate]
        self.edge_features = np.random.rand(num_edges, 4)
        self.edge_features[:, 0] = self.edge_features[:, 0] * 80 + 20    # Bandwidth: 20-100
        self.edge_features[:, 1] = self.edge_features[:, 1] * 5 + 1      # Delay: 1-6 ms
        self.edge_features[:, 2] = self.edge_features[:, 2] * 0.5        # Jitter: 0-0.5 ms
        self.edge_features[:, 3] = self.edge_features[:, 3] * 0.02       # Loss: 0-2%
        
        print(f"🌐 网络拓扑生成完成:")
        print(f"   - 节点数: {num_nodes}")
        print(f"   - 边数: {num_edges}")
        print(f"   - 连通性: {nx.is_connected(self.graph)}")
    
    def _create_fat_tree(self, k: int):
        """创建Fat-tree拓扑"""
        # 简化的Fat-tree实现
        G = nx.Graph()
        
        # 计算各层节点数
        core_switches = (k // 2) ** 2
        agg_switches = k * k // 2
        edge_switches = k * k // 2
        hosts = k ** 3 // 4
        
        total_nodes = core_switches + agg_switches + edge_switches + hosts
        
        # 添加节点
        for i in range(total_nodes):
            G.add_node(i)
        
        # 添加边（简化连接）
        # Core到Aggregation
        for core in range(core_switches):
            for agg in range(core_switches, core_switches + agg_switches):
                if np.random.random() < 0.5:  # 随机连接模拟Fat-tree
                    G.add_edge(core, agg)
        
        # Aggregation到Edge
        for agg in range(core_switches, core_switches + agg_switches):
            for edge in range(core_switches + agg_switches, core_switches + agg_switches + edge_switches):
                if np.random.random() < 0.6:
                    G.add_edge(agg, edge)
        
        # Edge到Host
        for edge in range(core_switches + agg_switches, core_switches + agg_switches + edge_switches):
            for host in range(core_switches + agg_switches + edge_switches, total_nodes):
                if np.random.random() < 0.3:
                    G.add_edge(edge, host)
        
        # 确保连通性
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                G.add_edge(node1, node2)
        
        return G
    
    def _setup_environments(self):
        """设置训练和测试环境"""
        reward_config = self.config['reward']
        
        # Edge-aware环境（使用完整的边特征）
        self.env_edge_aware = MultiVNFEmbeddingEnv(
            graph=self.graph,
            node_features=self.node_features,
            edge_features=self.edge_features,  # 完整的4维边特征
            reward_config=reward_config,
            chain_length_range=(2, 5)
        )
        
        # Baseline环境（简化的边特征，模拟传统方法）
        baseline_edge_features = self.edge_features[:, :2]  # 只使用带宽和延迟
        baseline_edge_features = np.hstack([
            baseline_edge_features,
            np.zeros((baseline_edge_features.shape[0], 2))  # 填充零值
        ])
        
        self.env_baseline = MultiVNFEmbeddingEnv(
            graph=self.graph,
            node_features=self.node_features,
            edge_features=baseline_edge_features,  # 简化的边特征
            reward_config=reward_config,
            chain_length_range=(2, 5)
        )
        
        print(f"🌍 环境设置完成:")
        print(f"   - Edge-aware环境: 4维边特征 (带宽, 延迟, 抖动, 丢包)")
        print(f"   - Baseline环境: 2维边特征 (仅带宽, 延迟)")
    
    def _setup_agents(self):
        """设置智能体"""
        # 计算实际的状态维度
        # 环境返回的节点特征 = 原始特征 + 状态信息(4维)
        actual_state_dim = self.node_features.shape[1] + 4  # +4 for [is_used, cpu_util, memory_util, vnf_count]
        action_dim = len(self.graph.nodes())
        edge_dim = self.edge_features.shape[1]
        
        print(f"🔧 智能体参数:")
        print(f"   - 原始节点特征维度: {self.node_features.shape[1]}")
        print(f"   - 实际状态维度: {actual_state_dim}")
        print(f"   - 动作维度: {action_dim}")
        print(f"   - 边特征维度: {edge_dim}")
        
        # Edge-aware智能体
        self.agents_edge_aware = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_edge_aware"
            self.agents_edge_aware[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=actual_state_dim,  # 使用实际状态维度
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
        
        # Baseline智能体
        self.agents_baseline = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_baseline"
            self.agents_baseline[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=actual_state_dim,  # 使用实际状态维度
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
        
        print(f"🤖 智能体设置完成:")
        print(f"   - Edge-aware智能体: {list(self.agents_edge_aware.keys())}")
        print(f"   - Baseline智能体: {list(self.agents_baseline.keys())}")
    
    def _setup_logging(self):
        """设置日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 为每个智能体创建日志记录器
        self.loggers = {}
        
        # Edge-aware智能体日志
        for agent_type in self.agent_types:
            logger_id = f"{agent_type}_edge_aware"
            self.loggers[logger_id] = Logger(
                log_dir=os.path.join(self.results_dir, f"{logger_id}_{timestamp}")
            )
        
        # Baseline智能体日志
        for agent_type in self.agent_types:
            logger_id = f"{agent_type}_baseline"
            self.loggers[logger_id] = Logger(
                log_dir=os.path.join(self.results_dir, f"{logger_id}_{timestamp}")
            )
        
        print(f"📊 日志记录设置完成")
    
    def train_single_episode(self, agent, env, agent_id: str) -> Dict[str, Any]:
        """
        训练单个episode
        
        Args:
            agent: 智能体
            env: 环境
            agent_id: 智能体ID
            
        Returns:
            episode_stats: Episode统计信息
        """
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        success = False
        
        # 重置episode统计
        agent.reset_episode_stats()
        
        while step_count < env.max_episode_steps:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            # 选择动作
            action = agent.select_action(state, valid_actions=valid_actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done, info=info)
            
            # 更新状态
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 学习（根据智能体类型）
            if hasattr(agent, 'should_update') and agent.should_update():
                # PPO智能体
                learning_info = agent.learn()
            elif len(getattr(agent, 'replay_buffer', [])) >= agent.batch_size:
                # DQN系列智能体
                learning_info = agent.learn()
            
            if done:
                success = info.get('success', False)
                break
        
        # PPO智能体的最终学习
        if hasattr(agent, 'should_update') and len(agent.experiences) > 0:
            learning_info = agent.learn()
        
        # 计算SAR和SPLat
        sar = 1.0 if success else 0.0
        splat = info.get('splat', step_count) if success else float('inf')
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': step_count,
            'success': success,
            'sar': sar,
            'splat': splat,
            'info': info
        }
        
        return episode_stats
    
    def train(self):
        """主训练循环"""
        print(f"\n🚀 开始多智能体训练...")
        print(f"目标episodes: {self.episodes}")
        
        # 存储所有结果
        all_results = {
            'edge_aware': {agent_type: {'rewards': [], 'sar': [], 'splat': [], 'success': []} 
                          for agent_type in self.agent_types},
            'baseline': {agent_type: {'rewards': [], 'sar': [], 'splat': [], 'success': []} 
                        for agent_type in self.agent_types}
        }
        
        for episode in range(1, self.episodes + 1):
            print(f"\n📍 Episode {episode}/{self.episodes}")
            
            # 训练Edge-aware智能体
            for agent_type in self.agent_types:
                agent = self.agents_edge_aware[agent_type]
                env = self.env_edge_aware
                
                episode_stats = self.train_single_episode(agent, env, f"{agent_type}_edge_aware")
                
                # 记录结果
                all_results['edge_aware'][agent_type]['rewards'].append(episode_stats['total_reward'])
                all_results['edge_aware'][agent_type]['sar'].append(episode_stats['sar'])
                all_results['edge_aware'][agent_type]['splat'].append(episode_stats['splat'])
                all_results['edge_aware'][agent_type]['success'].append(episode_stats['success'])
                
                # 记录到日志
                logger_id = f"{agent_type}_edge_aware"
                if logger_id in self.loggers:
                    self.loggers[logger_id].log_episode(episode, episode_stats)
            
            # 训练Baseline智能体
            for agent_type in self.agent_types:
                agent = self.agents_baseline[agent_type]
                env = self.env_baseline
                
                episode_stats = self.train_single_episode(agent, env, f"{agent_type}_baseline")
                
                # 记录结果
                all_results['baseline'][agent_type]['rewards'].append(episode_stats['total_reward'])
                all_results['baseline'][agent_type]['sar'].append(episode_stats['sar'])
                all_results['baseline'][agent_type]['splat'].append(episode_stats['splat'])
                all_results['baseline'][agent_type]['success'].append(episode_stats['success'])
                
                # 记录到日志
                logger_id = f"{agent_type}_baseline"
                if logger_id in self.loggers:
                    self.loggers[logger_id].log_episode(episode, episode_stats)
            
            # 定期打印进度
            if episode % 25 == 0:
                self._print_progress(episode, all_results)
            
            # 定期保存模型
            if episode % self.save_interval == 0:
                self._save_models(episode)
            
            # 定期生成可视化
            if episode % 50 == 0:
                self._generate_visualizations(episode, all_results)
        
        # 训练完成后的最终分析
        self._final_analysis(all_results)
        
        print(f"\n🎉 训练完成!")
        return all_results
    
    def _print_progress(self, episode: int, results: Dict):
        """打印训练进度"""
        print(f"\n📊 Episode {episode} 性能统计:")
        
        # 计算最近25个episode的平均性能
        window = 25
        start_idx = max(0, episode - window)
        
        for variant in ['edge_aware', 'baseline']:
            print(f"\n{variant.upper()}:")
            for agent_type in self.agent_types:
                recent_sar = np.mean(results[variant][agent_type]['sar'][start_idx:])
                recent_splat = np.mean([s for s in results[variant][agent_type]['splat'][start_idx:] 
                                      if s != float('inf')])
                recent_reward = np.mean(results[variant][agent_type]['rewards'][start_idx:])
                
                print(f"  {agent_type.upper()}: SAR={recent_sar:.3f}, SPLat={recent_splat:.2f}, Reward={recent_reward:.1f}")
    
    def _save_models(self, episode: int):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints", f"episode_{episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存Edge-aware智能体
        for agent_type, agent in self.agents_edge_aware.items():
            filepath = os.path.join(checkpoint_dir, f"{agent_type}_edge_aware.pth")
            agent.save_checkpoint(filepath)
        
        # 保存Baseline智能体
        for agent_type, agent in self.agents_baseline.items():
            filepath = os.path.join(checkpoint_dir, f"{agent_type}_baseline.pth")
            agent.save_checkpoint(filepath)
    
    def _generate_visualizations(self, episode: int, results: Dict):
        """生成训练可视化"""
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # SAR对比
        axes[0, 0].set_title('Service Acceptance Rate (SAR)')
        for agent_type in self.agent_types:
            # Edge-aware
            episodes = list(range(1, len(results['edge_aware'][agent_type]['sar']) + 1))
            axes[0, 0].plot(episodes, results['edge_aware'][agent_type]['sar'], 
                          label=f'{agent_type.upper()} (Edge-aware)', alpha=0.8)
            # Baseline
            axes[0, 0].plot(episodes, results['baseline'][agent_type]['sar'], 
                          label=f'{agent_type.upper()} (Baseline)', alpha=0.8, linestyle='--')
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('SAR')
        axes[0, 0].grid(True)
        
        # SPLat对比
        axes[0, 1].set_title('Service Path Latency (SPLat)')
        for agent_type in self.agent_types:
            # 过滤无穷大值
            edge_splat = [s for s in results['edge_aware'][agent_type]['splat'] if s != float('inf')]
            baseline_splat = [s for s in results['baseline'][agent_type]['splat'] if s != float('inf')]
            
            if edge_splat:
                axes[0, 1].plot(range(1, len(edge_splat) + 1), edge_splat, 
                              label=f'{agent_type.upper()} (Edge-aware)', alpha=0.8)
            if baseline_splat:
                axes[0, 1].plot(range(1, len(baseline_splat) + 1), baseline_splat, 
                              label=f'{agent_type.upper()} (Baseline)', alpha=0.8, linestyle='--')
        axes[0, 1].legend()
        axes[0, 1].set_ylabel('SPLat')
        axes[0, 1].grid(True)
        
        # 奖励对比
        axes[1, 0].set_title('Training Rewards')
        for agent_type in self.agent_types:
            episodes = list(range(1, len(results['edge_aware'][agent_type]['rewards']) + 1))
            axes[1, 0].plot(episodes, results['edge_aware'][agent_type]['rewards'], 
                          label=f'{agent_type.upper()} (Edge-aware)', alpha=0.8)
            axes[1, 0].plot(episodes, results['baseline'][agent_type]['rewards'], 
                          label=f'{agent_type.upper()} (Baseline)', alpha=0.8, linestyle='--')
        axes[1, 0].legend()
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
        
        # 成功率对比
        axes[1, 1].set_title('Success Rate')
        for agent_type in self.agent_types:
            # 计算滑动平均成功率
            window = 20
            edge_success = self._rolling_average(results['edge_aware'][agent_type]['success'], window)
            baseline_success = self._rolling_average(results['baseline'][agent_type]['success'], window)
            
            episodes = list(range(window, len(edge_success) + window))
            axes[1, 1].plot(episodes, edge_success, 
                          label=f'{agent_type.upper()} (Edge-aware)', alpha=0.8)
            axes[1, 1].plot(episodes, baseline_success, 
                          label=f'{agent_type.upper()} (Baseline)', alpha=0.8, linestyle='--')
        axes[1, 1].legend()
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'training_progress_episode_{episode}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _rolling_average(self, data: List[float], window: int) -> List[float]:
        """计算滑动平均"""
        if len(data) < window:
            return []
        
        return [np.mean(data[i:i+window]) for i in range(len(data) - window + 1)]
    
    def _final_analysis(self, results: Dict):
        """最终性能分析"""
        print(f"\n🎯 最终性能分析:")
        print(f"{'='*60}")
        
        # 计算最后50个episode的平均性能
        window = 50
        
        summary_data = []
        
        for variant in ['edge_aware', 'baseline']:
            print(f"\n{variant.upper()} 结果:")
            for agent_type in self.agent_types:
                recent_sar = np.mean(results[variant][agent_type]['sar'][-window:])
                recent_splat = np.mean([s for s in results[variant][agent_type]['splat'][-window:] 
                                      if s != float('inf')])
                recent_reward = np.mean(results[variant][agent_type]['rewards'][-window:])
                recent_success = np.mean(results[variant][agent_type]['success'][-window:])
                
                print(f"  {agent_type.upper()}:")
                print(f"    SAR: {recent_sar:.3f}")
                print(f"    SPLat: {recent_splat:.2f}")
                print(f"    Reward: {recent_reward:.1f}")
                print(f"    Success Rate: {recent_success:.3f}")
                
                summary_data.append({
                    'Variant': variant,
                    'Algorithm': agent_type.upper(),
                    'SAR': recent_sar,
                    'SPLat': recent_splat,
                    'Reward': recent_reward,
                    'Success_Rate': recent_success
                })
        
        # 保存CSV结果
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.results_dir, 'final_results_summary.csv'), index=False)
        
        # 计算改进幅度
        print(f"\n📈 Edge-aware vs Baseline 改进幅度:")
        for agent_type in self.agent_types:
            edge_sar = np.mean(results['edge_aware'][agent_type]['sar'][-window:])
            baseline_sar = np.mean(results['baseline'][agent_type]['sar'][-window:])
            sar_improvement = ((edge_sar - baseline_sar) / baseline_sar) * 100 if baseline_sar > 0 else 0
            
            edge_splat = np.mean([s for s in results['edge_aware'][agent_type]['splat'][-window:] 
                                if s != float('inf')])
            baseline_splat = np.mean([s for s in results['baseline'][agent_type]['splat'][-window:] 
                                    if s != float('inf')])
            splat_improvement = ((baseline_splat - edge_splat) / baseline_splat) * 100 if baseline_splat > 0 else 0
            
            print(f"  {agent_type.upper()}:")
            print(f"    SAR改进: {sar_improvement:+.1f}%")
            print(f"    SPLat改进: {splat_improvement:+.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VNF嵌入多智能体训练')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    args = parser.parse_args()
    
    # 创建训练器
    trainer = MultiAgentTrainer(config_path=args.config)
    
    # 覆盖配置中的episodes数（如果指定）
    if args.episodes:
        trainer.episodes = args.episodes
        trainer.config['train']['episodes'] = args.episodes
    
    # 开始训练
    results = trainer.train()
    
    print(f"\n✅ 训练任务完成!")
    print(f"📁 结果保存在: {trainer.results_dir}")


if __name__ == "__main__":
    main()