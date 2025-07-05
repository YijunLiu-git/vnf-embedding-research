# test_agents.py

import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np
from typing import Dict, Any, List
from agents.multi_dqn_agent import MultiDQNAgent
from agents.multi_ddqn_agent import MultiDDQNAgent
from agents.multi_ppo_agent import MultiPPOAgent
from agents.base_agent import create_agent
from config_loader import get_scenario_config

def create_test_state(config: Dict[str, Any], num_nodes: int = 42) -> Data:
    """创建测试用的网络状态"""
    node_dim = config['dimensions']['node_feature_dim']
    edge_dim = config['dimensions']['edge_feature_dim_full']
    vnf_context_dim = config['dimensions']['vnf_context_dim']
    
    print(f"📏 配置维度: node_dim={node_dim}, edge_dim={edge_dim}, vnf_context_dim={vnf_context_dim}")
    
    G = nx.erdos_renyi_graph(num_nodes, 0.3, seed=42)
    edge_list = list(G.edges())
    num_edges = len(edge_list)
    
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_dim)
    
    if edge_attr.size(1) != 4:
        raise ValueError(f"边特征维度不匹配: 期望 4, 实际 {edge_attr.size(1)}")
    
    x = torch.randn(num_nodes, node_dim)
    vnf_context = torch.randn(vnf_context_dim)
    network_state = torch.randn(8)
    
    enhanced_info = {
        'path_quality_matrix': {(i, j): {
            'quality_score': np.random.rand(),
            'bandwidth': np.random.rand() * 100,
            'latency': np.random.rand() * 100,
            'jitter': np.random.rand() * 5,
            'packet_loss': np.random.rand() * 0.1
        } for i in range(num_nodes) for j in range(num_nodes) if i != j},
        'network_state_vector': network_state.numpy()
    }
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        vnf_context=vnf_context,
        network_state=network_state,
        enhanced_info=enhanced_info
    )
    
    print(f"生成测试状态: x.shape={x.shape}, edge_attr.shape={edge_attr.shape}, vnf_context.shape={vnf_context.shape}")
    return data

def test_agent_initialization():
    """测试代理初始化"""
    print("🧪 测试代理初始化...")
    config = get_scenario_config('normal_operation')
    agent_types = ['dqn', 'ddqn', 'ppo']
    
    for agent_type in agent_types:
        agent = create_agent(
            agent_type=agent_type,
            agent_id=f"test_{agent_type}_enhanced",
            state_dim=config['dimensions']['node_feature_dim'],
            action_dim=config['topology']['node_counts']['total'],
            edge_dim=config['dimensions']['edge_feature_dim_full'],
            config=config,
            use_enhanced_gnn=True
        )
        assert agent is not None, f"{agent_type.upper()} 初始化失败"
        assert hasattr(agent, 'gnn_encoder'), f"{agent_type.upper()} 缺少GNN编码器"
        assert agent.gnn_encoder.__class__.__name__ == 'EnhancedEdgeAwareGNN', f"{agent_type.upper()} 未使用增强GNN"
        print(f"✅ {agent_type.upper()} 初始化测试通过")

def test_action_selection():
    """测试动作选择"""
    print("\n🧪 测试动作选择...")
    config = get_scenario_config('normal_operation')
    agent_types = ['dqn', 'ddqn', 'ppo']
    num_nodes = config['topology']['node_counts']['total']
    
    for agent_type in agent_types:
        agent = create_agent(
            agent_type=agent_type,
            agent_id=f"test_{agent_type}_enhanced",
            state_dim=config['dimensions']['node_feature_dim'],
            action_dim=num_nodes,
            edge_dim=config['dimensions']['edge_feature_dim_full'],
            config=config,
            use_enhanced_gnn=True
        )
        test_state = create_test_state(config, num_nodes)
        
        assert test_state.edge_attr.size(1) == 4, \
            f"{agent_type.upper()} 测试状态边特征维度错误: {test_state.edge_attr.size(1)}"
        
        valid_actions = list(range(num_nodes))
        action = agent.select_action(test_state, valid_actions=valid_actions)
        assert action in valid_actions, f"{agent_type.upper()} 动作选择无效: {action}"
        print(f"✅ {agent_type.upper()} 动作选择测试通过: 动作={action}, 边特征维度={test_state.edge_attr.size(1)}")

def test_learning_process():
    """测试学习过程"""
    print("\n🧪 测试学习过程...")
    scenarios = ['normal_operation', 'peak_congestion', 'failure_recovery', 'extreme_pressure']
    agent_types = ['dqn', 'ddqn', 'ppo']
    
    for scenario in scenarios:
        print(f"\n📊 场景: {scenario}")
        config = get_scenario_config(scenario)
        num_nodes = config['topology']['node_counts']['total']
        
        for agent_type in agent_types:
            agent = create_agent(
                agent_type=agent_type,
                agent_id=f"test_{agent_type}_{scenario}",
                state_dim=config['dimensions']['node_feature_dim'],
                action_dim=num_nodes,
                edge_dim=config['dimensions']['edge_feature_dim_full'],
                config=config,
                use_enhanced_gnn=True
            )
            agent.is_training = True
            
            for _ in range(20):
                state = create_test_state(config, num_nodes)
                action = agent.select_action(state, valid_actions=list(range(num_nodes)))
                reward = np.random.uniform(-1.0, 1.0)
                next_state = create_test_state(config, num_nodes)
                done = np.random.random() < 0.2
                agent.store_transition(state, action, reward, next_state, done)
            
            if agent_type != 'ppo' or agent.should_update():
                learning_info = agent.learn()
                assert 'loss' in learning_info, f"{agent_type.upper()} 学习过程未返回损失"
                print(f"✅ {agent_type.upper()} 学习测试通过: Loss={learning_info['loss']:.4f}, 边特征维度={state.edge_attr.size(1)}")
            else:
                print(f"✅ {agent_type.upper()} 学习测试跳过（PPO未达更新条件）")

def test_config_compatibility():
    """测试配置兼容性"""
    print("\n🧪 测试配置兼容性...")
    scenarios = ['normal_operation', 'peak_congestion', 'failure_recovery', 'extreme_pressure']
    
    for scenario in scenarios:
        config = get_scenario_config(scenario)
        assert 'dimensions' in config, f"{scenario} 缺少维度配置"
        assert config['dimensions']['vnf_context_dim'] == 6, f"{scenario} VNF上下文维度错误"
        assert config['dimensions']['node_feature_dim'] == 8, f"{scenario} 节点特征维度错误"
        assert config['dimensions']['edge_feature_dim_full'] == 4, f"{scenario} 边特征维度错误"
        assert config['gnn']['edge_aware']['edge_dim'] == 4, f"{scenario} GNN边维度不匹配"
        print(f"✅ 场景 {scenario} 配置兼容性测试通过")

def main():
    print("🚀 增强版代理测试")
    print("=" * 50)
    
    test_agent_initialization()
    test_action_selection()
    test_learning_process()
    test_config_compatibility()
    
    print("\n🎉 所有测试通过!")

if __name__ == "__main__":
    main()