# main_multi_agent.py - 修复版：使用新的配置加载器

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
import random

from env.vnf_env_multi import MultiVNFEmbeddingEnv
from env.topology_loader import generate_topology
from agents.base_agent import create_agent
from utils.logger import Logger
from utils.metrics import calculate_sar, calculate_splat
# ✅ 关键修复：使用新的配置加载器
from config_loader import get_scenario_config, print_scenario_plan, validate_all_configs, load_config
# 在main_multi_agent.py中  
from rewards.enhanced_edge_aware_reward import compute_enhanced_edge_aware_reward
from project.enhanced_training_system import SafeEnhancedTrainer

class MultiAgentTrainer:
    """
    多智能体VNF嵌入训练器 - 配置加载器修复版本
    
    主要修复：
    1. ✅ 使用统一的config_loader替代旧的场景配置系统
    2. ✅ 正确的场景配置传递机制
    3. ✅ 避免重复资源修改造成的冲突
    4. ✅ 确保场景名称正确显示
    5. ✅ 场景间资源配置的平滑过渡
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # ✅ 使用新的配置加载器
        self.config = load_config(config_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")
        
        self.episodes = self.config['train']['episodes']
        self.save_interval = 50
        self.eval_interval = 25
        self.agent_types = ['ddqn', 'dqn', 'ppo']
        
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # ✅ 渐进式场景相关 - 使用新配置系统
        self.current_scenario = "normal_operation"
        self.scenario_start_episode = 1
        self.last_applied_scenario = None  # 避免重复应用
        
        # ✅ 场景信息映射
        self.scenario_info = {
            'normal_operation': {
                'name': '正常运营期',
                'episodes': [1, 25],
                'expected_sar': '80-95%',
                'realism': '⭐⭐⭐⭐⭐',
                'focus': '基础功能验证'
            },
            'peak_congestion': {
                'name': '高峰拥塞期',
                'episodes': [26, 50],
                'expected_sar': '65-80%',
                'realism': '⭐⭐⭐⭐',
                'focus': 'Edge-aware优势体现'
            },
            'failure_recovery': {
                'name': '故障恢复期',
                'episodes': [51, 75],
                'expected_sar': '50-65%',
                'realism': '⭐⭐⭐',
                'focus': '鲁棒性验证'
            },
            'extreme_pressure': {
                'name': '极限压力期',
                'episodes': [76, 100],
                'expected_sar': '35-50%',
                'realism': '⭐⭐',
                'focus': '算法边界研究'
            }
        }
        
        # 打印训练计划
        print_scenario_plan()
        
        # 验证配置
        print("\n🔍 验证配置文件...")
        validate_all_configs()
        
        self._setup_network_topology()
        self._setup_environments()
        self._setup_agents()
        self._setup_logging()
        
        print(f"✅ 多智能体训练器初始化完成 (配置加载器修复版)")
        print(f"   - 智能体类型: {self.agent_types}")
        print(f"   - 训练轮数: {self.episodes}")
        print(f"   - 网络节点: {len(self.graph.nodes())}")
    
    def _setup_network_topology(self):
        """设置网络拓扑"""
        # ✅ 使用完整的配置字典生成拓扑
        full_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'dimensions': self.config['dimensions']
        }
        
        self.graph, self.node_features, self.edge_features = generate_topology(config=full_config)
        
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        
        # 验证网络拓扑
        if self.graph.edges():
            sample_edge = list(self.graph.edges(data=True))[0]
            edge_attrs = list(sample_edge[2].keys())
            bandwidths = [self.graph.edges[u, v].get('bandwidth', 0) for u, v in self.graph.edges()]
            
            print(f"🌐 网络拓扑生成完成:")
            print(f"   - 节点数: {num_nodes}")
            print(f"   - 边数: {num_edges}")
            print(f"   - 连通性: {nx.is_connected(self.graph)}")
            print(f"   - 节点特征维度: {self.node_features.shape[1]} (预期4维)")
            print(f"   - 边特征维度: {self.edge_features.shape[1]} (预期4维)")
            print(f"   - 边属性: {edge_attrs}")
            print(f"   - 带宽范围: {min(bandwidths):.1f} - {max(bandwidths):.1f}")
            
            # ✅ 验证维度一致性
            assert self.node_features.shape[1] == 4, f"节点特征应为4维，实际{self.node_features.shape[1]}维"
            assert self.edge_features.shape[1] == 4, f"边特征应为4维，实际{self.edge_features.shape[1]}维"
            print(f"   ✅ 特征维度验证通过")
    
    def _setup_environments(self):
        """设置训练和测试环境"""
        reward_config = self.config['reward']
        chain_length_range = tuple(self.config['vnf_requirements']['chain_length_range'])
        
        # ✅ 创建完整的配置字典传递给环境
        env_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'reward': self.config['reward'],
            'train': self.config['train'],
            'dimensions': self.config['dimensions']
        }
        
        # Edge-aware环境（使用完整的4维边特征）
        self.env_edge_aware = MultiVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=reward_config,
            chain_length_range=chain_length_range,
            config=env_config.copy()
        )
        
        # Baseline环境
        self.env_baseline = MultiVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=reward_config,
            chain_length_range=chain_length_range,
            config=env_config.copy()
        )
        # 标记Baseline环境，让智能体只看到2维特征
        self.env_baseline.is_baseline_mode = True
        
        print(f"🌍 环境设置完成:")
        print(f"   - Edge-aware环境: 4维边特征 (带宽, 延迟, 抖动, 丢包)")
        print(f"   - Baseline环境: 4维环境特征，但智能体只感知2维 (带宽, 延迟)")
    
    def _setup_agents(self):
        """设置智能体"""
        # ✅ 根据配置文件确定维度
        expected_node_dim = self.config['dimensions']['node_feature_dim']  # 8维
        actual_state_dim = expected_node_dim  # 环境直接提供8维
        action_dim = len(self.graph.nodes())
        
        print(f"🔧 智能体参数:")
        print(f"   - 原始节点特征维度: {self.node_features.shape[1]} (4维基础)")
        print(f"   - 环境输出状态维度: {expected_node_dim} (8维扩展)")
        print(f"   - 智能体输入维度: {actual_state_dim}")
        print(f"   - 动作维度: {action_dim}")
        
        # Edge-aware智能体
        self.agents_edge_aware = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_edge_aware"
            edge_dim = self.config['gnn']['edge_aware']['edge_dim']
            self.agents_edge_aware[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=actual_state_dim,
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
            print(f"🤖 Agent {agent_id} 使用边特征维度: {edge_dim}")
        
        # Baseline智能体
        self.agents_baseline = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_baseline"
            edge_dim = self.config['gnn']['baseline']['edge_dim']
            self.agents_baseline[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=actual_state_dim,
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
            print(f"🤖 Agent {agent_id} 使用边特征维度: {edge_dim}")
        
        print(f"🤖 智能体设置完成:")
        print(f"   - Edge-aware智能体: {list(self.agents_edge_aware.keys())}")
        print(f"   - Baseline智能体: {list(self.agents_baseline.keys())}")
    
    def _setup_logging(self):
        """设置日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.loggers = {}
        
        for agent_type in self.agent_types:
            # Edge-aware日志器
            logger_id = f"{agent_type}_edge_aware"
            self.loggers[logger_id] = Logger(
                log_dir=os.path.join(self.results_dir, f"{logger_id}_{timestamp}")
            )
            # Baseline日志器
            logger_id = f"{agent_type}_baseline"
            self.loggers[logger_id] = Logger(
                log_dir=os.path.join(self.results_dir, f"{logger_id}_{timestamp}")
            )
        
        print(f"📊 日志记录设置完成")

    def _update_scenario(self, episode: int):
        """✅ 修复版：更新当前场景 - 使用新配置系统"""
        new_scenario = None
        
        # 基于episode数量确定当前应该是哪个场景
        if episode <= 25:
            new_scenario = "normal_operation"
        elif episode <= 50:
            new_scenario = "peak_congestion"
        elif episode <= 75:
            new_scenario = "failure_recovery"
        else:
            new_scenario = "extreme_pressure"
        
        # ✅ 关键修复：只在场景真正改变时才应用配置
        if new_scenario and new_scenario != self.current_scenario:
            print(f"\n🎯 场景切换: {self.current_scenario} → {new_scenario}")
            
            old_scenario = self.current_scenario
            self.current_scenario = new_scenario
            self.scenario_start_episode = episode
            
            # ✅ 使用新的配置加载器获取场景配置
            scenario_config = get_scenario_config(episode)
            
            # 显示场景信息
            current_scenario_info = self.scenario_info[new_scenario]
            print(f"📍 Episode {episode}: 进入 {current_scenario_info['name']}")
            print(f"   现实性等级: {current_scenario_info['realism']}")
            print(f"   预期SAR范围: {current_scenario_info['expected_sar']}")
            print(f"   研究焦点: {current_scenario_info['focus']}")
            
            # ✅ 直接应用场景配置到环境并验证
            print(f"🔧 应用场景配置到环境...")
            print(f"   配置详情: {scenario_config['topology']['node_resources']}")
            print(f"   VNF需求: {scenario_config['vnf_requirements']}")
            self.env_edge_aware.apply_scenario_config(scenario_config)
            self.env_baseline.apply_scenario_config(scenario_config)
            
            # 验证资源更新
            bandwidths = [self.env_edge_aware.graph.edges[u, v].get('bandwidth', 0) for u, v in self.env_edge_aware.graph.edges()]
            print(f"   更新后带宽范围: {min(bandwidths):.1f} - {max(bandwidths):.1f}")
            print(f"   {'-'*50}")
            self.last_applied_scenario = new_scenario
            return True
        return False
    
    def train_single_episode(self, agent, env, agent_id: str) -> Dict[str, Any]:
        """训练单个episode"""
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        success = False
        info = {}
        
        # 重置智能体episode统计
        if hasattr(agent, 'reset_episode_stats'):
            agent.reset_episode_stats()
        
        max_steps = getattr(env, 'max_episode_steps', 20)
        
        while step_count < max_steps:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                info = {'success': False, 'reason': 'no_valid_actions'}
                break
            
            # 选择动作
            action = agent.select_action(state, valid_actions=valid_actions)
            if action not in valid_actions:
                action = random.choice(valid_actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新状态和统计
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 学习更新
            try:
                if hasattr(agent, 'should_update') and agent.should_update():
                    learning_info = agent.learn()
                elif hasattr(agent, 'replay_buffer') and len(getattr(agent, 'replay_buffer', [])) >= getattr(agent, 'batch_size', 32):
                    learning_info = agent.learn()
            except Exception as e:
                pass  # 忽略学习错误
            
            if done:
                success = info.get('success', False)
                print(f"Episode结束: agent={agent_id}, success={success}, reason={info.get('reason', 'unknown')}")  # ✅ 添加调试信息
                break
        
        # 最后一次学习更新
        try:
            if hasattr(agent, 'experiences') and len(getattr(agent, 'experiences', [])) > 0:
                if hasattr(agent, 'should_update') and agent.should_update():
                    learning_info = agent.learn()
        except Exception as e:
            pass
        
        # ✅ 关键修复：正确获取场景名称用于统计
        current_scenario_name = getattr(env, 'current_scenario_name', self.current_scenario)
        
        # 计算episode统计
        sar = 1.0 if success else 0.0
        splat = info.get('splat', info.get('avg_delay', float('inf'))) if success else float('inf')
        jitter = info.get('avg_jitter', 0.0) if success else 0.0
        loss = info.get('avg_loss', 0.0) if success else 0.0
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': step_count,
            'success': success,
            'sar': sar,
            'splat': splat,
            'jitter': jitter,
            'loss': loss,
            'info': info,
            'scenario': current_scenario_name
        }
        
        return episode_stats
    
    def train(self):
        """主训练循环 - 配置加载器修复版本"""
        print(f"\n🚀 开始多智能体渐进式场景训练 (配置加载器修复版)...")
        print(f"目标episodes: {self.episodes}")
        
        all_results = {
            'edge_aware': {agent_type: {
                'rewards': [], 'sar': [], 'splat': [], 'success': [], 
                'jitter': [], 'loss': [], 'scenarios': []
            } for agent_type in self.agent_types},
            'baseline': {agent_type: {
                'rewards': [], 'sar': [], 'splat': [], 'success': [], 
                'jitter': [], 'loss': [], 'scenarios': []
            } for agent_type in self.agent_types}
        }
        
        # ✅ 初始化第一个场景 - 使用新配置系统
        print(f"🔧 初始化第一个场景...")
        initial_scenario_config = get_scenario_config(1)
        self.env_edge_aware.apply_scenario_config(initial_scenario_config)
        self.env_baseline.apply_scenario_config(initial_scenario_config)
        print(f"🎯 开始场景: {self.scenario_info[self.current_scenario]['name']}")
        
        for episode in range(1, self.episodes + 1):
            # 检查并更新场景
            scenario_changed = self._update_scenario(episode)
            
            if episode % 25 == 0 or scenario_changed:
                current_scenario_info = self.scenario_info.get(self.current_scenario, {})
                scenario_display_name = current_scenario_info.get('name', self.current_scenario)
                print(f"\n📍 Episode {episode}/{self.episodes} - 当前场景: {scenario_display_name}")
            
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
                all_results['edge_aware'][agent_type]['jitter'].append(episode_stats['jitter'])
                all_results['edge_aware'][agent_type]['loss'].append(episode_stats['loss'])
                all_results['edge_aware'][agent_type]['scenarios'].append(episode_stats['scenario'])
                
                # 记录日志
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
                all_results['baseline'][agent_type]['jitter'].append(episode_stats['jitter'])
                all_results['baseline'][agent_type]['loss'].append(episode_stats['loss'])
                all_results['baseline'][agent_type]['scenarios'].append(episode_stats['scenario'])
                
                # 记录日志
                logger_id = f"{agent_type}_baseline"
                if logger_id in self.loggers:
                    self.loggers[logger_id].log_episode(episode, episode_stats)
            
            # 定期打印进度
            if episode % 25 == 0:
                self._print_progress(episode, all_results)
            
            # 定期保存模型（确保只调用一次）
            if episode % self.save_interval == 0:
                print(f"保存检查点: episode {episode}")  # ✅ 添加调试信息
                self._save_models(episode)
        
        # 最终分析
        self._final_analysis(all_results)
        
        print(f"\n🎉 渐进式场景训练完成!")
        return all_results
    
    def _print_progress(self, episode: int, results: Dict):
        """打印训练进度 - 包含场景信息"""
        current_scenario_info = self.scenario_info.get(self.current_scenario, {})
        scenario_display_name = current_scenario_info.get('name', self.current_scenario)
        
        print(f"\n📊 Episode {episode} 性能统计 (场景: {scenario_display_name}):")
        window = 25
        start_idx = max(0, episode - window)
        
        # 获取当前场景的预期性能
        expected_sar = current_scenario_info.get('expected_sar', 'Unknown')
        
        for variant in ['edge_aware', 'baseline']:
            print(f"\n{variant.upper()}:")
            for agent_type in self.agent_types:
                recent_sar = np.mean(results[variant][agent_type]['sar'][start_idx:])
                recent_splat = np.mean([s for s in results[variant][agent_type]['splat'][start_idx:] 
                                      if s != float('inf')])
                recent_reward = np.mean(results[variant][agent_type]['rewards'][start_idx:])
                
                # ✅ 修复版：更准确的SAR评估
                if expected_sar == "80-95%":
                    sar_status = "✅" if 0.8 <= recent_sar <= 1.0 else "❌"
                elif expected_sar == "65-80%":
                    sar_status = "✅" if 0.65 <= recent_sar <= 0.8 else ("⚠️" if recent_sar > 0.8 else "❌")
                elif expected_sar == "50-65%":
                    sar_status = "✅" if 0.5 <= recent_sar <= 0.65 else ("⚠️" if recent_sar > 0.65 else "❌")
                elif expected_sar == "35-50%":
                    sar_status = "✅" if 0.35 <= recent_sar <= 0.5 else ("⚠️" if recent_sar > 0.5 else "❌")
                else:
                    sar_status = "?"
                
                print(f"  {agent_type.upper()}:")
                print(f"    SAR: {recent_sar:.3f} {sar_status} (预期: {expected_sar})")
                print(f"    SPLat: {recent_splat:.2f}")
                print(f"    Reward: {recent_reward:.1f}")
    
    def _save_models(self, episode: int):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints", f"episode_{episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存Edge-aware智能体
        for agent_type, agent in self.agents_edge_aware.items():
            filepath = os.path.join(checkpoint_dir, f"{agent_type}_edge_aware.pth")
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(filepath)
                print(f"💾 Agent {agent_type}_edge_aware 检查点已保存: {filepath}")
        
        # 保存Baseline智能体
        for agent_type, agent in self.agents_baseline.items():
            filepath = os.path.join(checkpoint_dir, f"{agent_type}_baseline.pth")
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(filepath)
                print(f"💾 Agent {agent_type}_baseline 检查点已保存: {filepath}")
    
    def _final_analysis(self, results: Dict):
        """最终性能分析 - 按场景分组"""
        print(f"\n🎯 渐进式场景最终性能分析:")
        print(f"{'='*70}")
        
        # 按场景分组分析
        all_scenario_data = []
        for scenario_name, scenario_config in self.scenario_info.items():
            print(f"\n📋 {scenario_config['name']}:")
            print(f"   现实性: {scenario_config['realism']} | 预期SAR: {scenario_config['expected_sar']}")
            
            for variant in ['edge_aware', 'baseline']:
                print(f"\n  {variant.upper()}:")
                for agent_type in self.agent_types:
                    # ✅ 修复版：更准确地获取场景数据
                    scenario_episodes = []
                    for i, ep_scenario in enumerate(results[variant][agent_type]['scenarios']):
                        if ep_scenario == scenario_name:
                            scenario_episodes.append(i)
                    
                    if scenario_episodes:
                        # 计算该场景的平均性能（使用最后10个episode以获得更稳定的结果）
                        recent_episodes = scenario_episodes[-10:] if len(scenario_episodes) >= 10 else scenario_episodes
                        
                        scenario_sar = np.mean([results[variant][agent_type]['sar'][i] for i in recent_episodes])
                        scenario_splat_values = [results[variant][agent_type]['splat'][i] for i in recent_episodes 
                                               if results[variant][agent_type]['splat'][i] != float('inf')]
                        scenario_splat = np.mean(scenario_splat_values) if scenario_splat_values else float('inf')
                        scenario_reward = np.mean([results[variant][agent_type]['rewards'][i] for i in recent_episodes])
                        
                        print(f"    {agent_type.upper()}:")
                        print(f"      SAR: {scenario_sar:.3f}")
                        print(f"      SPLat: {scenario_splat:.2f}")
                        print(f"      Reward: {scenario_reward:.1f}")
                        
                        all_scenario_data.append({
                            'Scenario': scenario_config['name'],
                            'Variant': variant,
                            'Algorithm': agent_type.upper(),
                            'SAR': scenario_sar,
                            'SPLat': scenario_splat,
                            'Reward': scenario_reward,
                            'Expected_SAR': scenario_config['expected_sar']
                        })
        
        # 保存结果
        df_scenarios = pd.DataFrame(all_scenario_data)
        df_scenarios.to_csv(os.path.join(self.results_dir, 'scenario_results.csv'), index=False)
        
        print(f"\n💾 结果已保存: scenario_results.csv")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VNF嵌入多智能体渐进式场景训练 (配置加载器修复版)')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    args = parser.parse_args()
    
    trainer = MultiAgentTrainer(config_path=args.config)
    
    if args.episodes:
        trainer.episodes = args.episodes
        trainer.config['train']['episodes'] = args.episodes
    
    results = trainer.train()
    
    print(f"\n✅ 渐进式场景训练任务完成!")
    print(f"📁 结果保存在: {trainer.results_dir}")
    print(f"🎯 训练经历了4个场景，从高现实性到研究导向")

if __name__ == "__main__":
    main()