# test_system.py

"""
快速系统测试脚本
验证所有修复的组件是否正常工作
"""

import os
import sys
import yaml
import torch
import numpy as np
import networkx as nx
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所有组件
try:
    from env.vnf_env_multi import MultiVNFEmbeddingEnv
    from env.topology_loader import generate_topology
    from agents.base_agent import create_agent
    from models.gnn_encoder import GNNEncoder
    from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    from utils.logger import Logger
    from utils.metrics import calculate_sar, calculate_splat
    from utils.visualization import plot_training_curves
    print("✅ 所有组件导入成功!")
except ImportError as e:
    print(f"❌ 组件导入失败: {e}")
    sys.exit(1)

def test_basic_components():
    """测试基础组件"""
    print("\n🧪 测试基础组件...")
    
    # 1. 测试GNN编码器
    print("   测试GNN编码器...")
    encoder = GNNEncoder(node_dim=8, edge_dim=4, hidden_dim=64, output_dim=128)
    
    # 创建测试图数据
    from torch_geometric.data import Data
    test_data = Data(
        x=torch.randn(10, 8),
        edge_index=torch.randint(0, 10, (2, 20)),
        edge_attr=torch.randn(20, 4)
    )
    
    with torch.no_grad():
        output = encoder(test_data)
    print(f"      ✅ GNN编码器输出形状: {output.shape}")
    
    # 2. 测试回放缓冲区
    print("   测试回放缓冲区...")
    buffer = ReplayBuffer(capacity=100)
    
    for i in range(10):
        buffer.add(
            state=torch.randn(8),
            action=i % 5,
            reward=np.random.random(),
            next_state=torch.randn(8),
            done=False
        )
    
    states, actions, rewards, next_states, dones = buffer.sample(5)
    print(f"      ✅ 回放缓冲区采样成功: {len(states)} 样本")
    
    # 3. 测试优先级回放缓冲区
    print("   测试优先级回放缓冲区...")
    priority_buffer = PrioritizedReplayBuffer(capacity=100)
    
    for i in range(10):
        priority_buffer.add(
            state=torch.randn(8),
            action=i % 5,
            reward=np.random.random(),
            next_state=torch.randn(8),
            done=False,
            priority=np.random.random()
        )
    
    # 确保有足够的数据再采样
    if len(priority_buffer) >= 5:
        batch_data = priority_buffer.sample(5)
        print(f"      ✅ 优先级回放缓冲区采样成功: {len(batch_data)} 个组件")
    else:
        print(f"      ⚠️ 优先级回放缓冲区数据不足，跳过采样测试")
    
    print("✅ 基础组件测试完成")

def test_environment():
    """测试VNF嵌入环境"""
    print("\n🌍 测试VNF嵌入环境...")
    
    # 创建测试网络
    G = nx.erdos_renyi_graph(n=10, p=0.4, seed=42)
    node_features = np.random.rand(10, 4) * 0.8 + 0.2
    edge_features = np.random.rand(len(G.edges()), 4)
    edge_features[:, 0] = edge_features[:, 0] * 80 + 20  # 带宽
    edge_features[:, 1] = edge_features[:, 1] * 5 + 1    # 延迟
    
    reward_config = {
        "alpha": 0.5, "beta": 0.2, "gamma": 0.2, "delta": 0.1, "penalty": 1.0
    }
    
    # 创建环境
    env = MultiVNFEmbeddingEnv(
        graph=G,
        node_features=node_features,
        edge_features=edge_features,
        reward_config=reward_config
    )
    
    print(f"   ✅ 环境创建成功")
    print(f"      网络节点: {len(G.nodes())}")
    print(f"      动作空间: {env.action_space}")
    
    # 测试环境交互
    state = env.reset()
    print(f"      初始状态类型: {type(state)}")
    print(f"      状态特征形状: {state.x.shape}")
    
    # 执行几步
    for step in range(3):
        valid_actions = env.get_valid_actions()
        if valid_actions:
            action = np.random.choice(valid_actions)
            next_state, reward, done, info = env.step(action)
            print(f"      步骤 {step+1}: 动作={action}, 奖励={reward:.2f}, 完成={done}")
            if done:
                break
        else:
            print(f"      步骤 {step+1}: 没有有效动作")
            break

def test_agents():
    """测试智能体"""
    print("\n🤖 测试智能体...")
    
    # 配置
    config = {
        "gnn": {"hidden_dim": 64, "output_dim": 128},
        "train": {
            "lr": 0.001, "gamma": 0.99, "batch_size": 16,
            "epsilon_start": 1.0, "epsilon_decay": 0.995, "epsilon_min": 0.01,
            "buffer_size": 1000, "target_update": 10,
            "eps_clip": 0.2, "entropy_coef": 0.01, "value_coef": 0.5,
            "ppo_epochs": 2, "mini_batch_size": 8, "rollout_length": 16
        },
        "network": {"hidden_dim": 256}
    }
    
    # 测试每种智能体
    agent_types = ['ddqn', 'dqn', 'ppo']
    agents = {}
    
    for agent_type in agent_types:
        print(f"   测试 {agent_type.upper()} 智能体...")
        
        try:
            agent = create_agent(
                agent_type=agent_type,
                agent_id=f"test_{agent_type}",
                state_dim=8,
                action_dim=10,
                edge_dim=4,
                config=config
            )
            agents[agent_type] = agent
            print(f"      ✅ {agent_type.upper()} 智能体创建成功")
            
            # 测试动作选择
            test_state = torch.randn(1, 128)
            action = agent.select_action(test_state)
            print(f"      动作选择测试: {action}")
            
            # 测试经验存储
            agent.store_transition(
                state=test_state,
                action=action,
                reward=1.0,
                next_state=torch.randn(1, 128),
                done=False
            )
            print(f"      ✅ 经验存储成功")
            
        except Exception as e:
            print(f"      ❌ {agent_type.upper()} 智能体测试失败: {e}")

def test_integration():
    """测试完整集成"""
    print("\n🔄 测试完整集成...")
    
    # 创建简单的配置
    config = {
        "gnn": {"hidden_dim": 32, "output_dim": 64},
        "train": {
            "lr": 0.001, "gamma": 0.99, "batch_size": 8,
            "epsilon_start": 1.0, "epsilon_decay": 0.995, "epsilon_min": 0.01,
            "buffer_size": 100, "target_update": 10,
            "episodes": 5  # 只运行5个episode测试
        },
        "network": {"hidden_dim": 128}
    }
    
    # 创建网络和环境
    G = nx.erdos_renyi_graph(n=8, p=0.5, seed=42)
    node_features = np.random.rand(8, 4) * 0.8 + 0.2
    edge_features = np.random.rand(len(G.edges()), 4)
    edge_features[:, 0] = edge_features[:, 0] * 80 + 20
    edge_features[:, 1] = edge_features[:, 1] * 5 + 1
    
    reward_config = {"alpha": 0.5, "beta": 0.2, "gamma": 0.2, "delta": 0.1, "penalty": 1.0}
    
    env = MultiVNFEmbeddingEnv(
        graph=G,
        node_features=node_features,
        edge_features=edge_features,
        reward_config=reward_config
    )
    
    # 创建DDQN智能体
    # 注意：环境的状态包含增强特征，所以state_dim应该是 node_features + status_features
    actual_state_dim = 4 + 4  # 原始节点特征4维 + 状态信息4维
    agent = create_agent(
        agent_type='ddqn',
        agent_id='integration_test',
        state_dim=actual_state_dim,  # 使用正确的状态维度
        action_dim=8,
        edge_dim=4,
        config=config
    )
    
    print(f"   运行 {config['train']['episodes']} 个episode...")
    print(f"   网络节点特征形状: {node_features.shape}")
    print(f"   环境实际状态维度: {env.actual_state_dim}")
    print(f"   智能体期望状态维度: {actual_state_dim}")
    
    # 运行训练循环
    episode_rewards = []
    episode_sars = []
    
    for episode in range(config['train']['episodes']):
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        max_steps = 10
        
        while step_count < max_steps:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            # 选择动作
            action = agent.select_action(state, valid_actions=valid_actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 学习
            if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) >= agent.batch_size:
                learning_info = agent.learn()
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        # 记录结果
        episode_rewards.append(total_reward)
        sar = 1.0 if info.get('success', False) else 0.0
        episode_sars.append(sar)
        
        print(f"      Episode {episode+1}: 奖励={total_reward:.2f}, SAR={sar:.2f}, 步数={step_count}")
    
    # 计算平均性能
    avg_reward = np.mean(episode_rewards)
    avg_sar = np.mean(episode_sars)
    
    print(f"   ✅ 集成测试完成!")
    print(f"      平均奖励: {avg_reward:.2f}")
    print(f"      平均SAR: {avg_sar:.2f}")
    
    return avg_reward, avg_sar

def test_logging():
    """测试日志功能"""
    print("\n📊 测试日志功能...")
    
    # 创建临时日志目录
    log_dir = f"test_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = Logger(log_dir)
    
    # 记录测试数据
    for episode in range(3):
        episode_stats = {
            'total_reward': np.random.uniform(10, 50),
            'steps': np.random.randint(5, 15),
            'success': np.random.choice([True, False]),
            'sar': np.random.uniform(0.5, 1.0),
            'splat': np.random.uniform(2, 8)
        }
        logger.log_episode(episode + 1, episode_stats)
    
    print(f"   ✅ 日志记录成功: {log_dir}")
    
    # 清理测试文件
    import shutil
    try:
        shutil.rmtree(log_dir)
        print(f"   🧹 清理测试文件: {log_dir}")
    except:
        pass

def main():
    """主测试函数"""
    print("🚀 开始系统完整性测试...")
    print("="*60)
    
    try:
        # 1. 基础组件测试
        test_basic_components()
        
        # 2. 环境测试
        test_environment()
        
        # 3. 智能体测试
        test_agents()
        
        # 4. 集成测试
        avg_reward, avg_sar = test_integration()
        
        # 5. 日志测试
        test_logging()
        
        print("\n" + "="*60)
        print("🎉 系统测试完成!")
        print(f"   集成测试结果: 平均奖励={avg_reward:.2f}, 平均SAR={avg_sar:.2f}")
        
        if avg_sar > 0.3:  # 如果SAR大于30%，认为系统基本正常
            print("✅ 系统运行正常，可以开始完整训练!")
            print("\n📋 下一步建议:")
            print("   1. 运行: python main_multi_agent.py --episodes 300")
            print("   2. 检查 results/ 目录中的训练结果")
            print("   3. 对比 edge-aware 和 baseline 的性能差异")
        else:
            print("⚠️  系统可能需要进一步调优")
            print("   建议检查奖励函数和网络参数设置")
        
    except Exception as e:
        print(f"\n❌ 系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)