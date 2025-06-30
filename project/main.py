# main.py

import yaml
import os
from env.topology_loader import load_fat_tree_topology
from env.vnf_env import VNFEmbeddingEnv
from agents.ddqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.hrl_controller import HRLController
from utils.visualization import plot_curve
import torch
import numpy as np

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()

    # === 加载网络拓扑与特征 ===
    graph, node_features, edge_features = load_fat_tree_topology()
    env = VNFEmbeddingEnv(graph, node_features, edge_features)

    # === 初始化智能体 ===
    agent_name = config["train"]["agent"]
    state_dim = env.state_dim
    edge_dim = env.edge_dim
    action_dim = env.action_dim

    if agent_name == "ddqn":
        agent = DDQNAgent(state_dim, action_dim, config)
    elif agent_name == "dqn":
        agent = DQNAgent(state_dim, action_dim, config)
    elif agent_name == "ppo":
        agent = PPOAgent(state_dim, action_dim, config)
    elif agent_name == "hrl":
        # 注意：你需要在 config.yaml 中提供 high_level 和 low_level 配置
        agent = HRLController(config["hrl_high"], config["hrl_low"])
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    episodes = config["train"]["episodes"]
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        # 简单一轮部署（可扩展为多阶段部署）
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()
        total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    # === 保存结果图 ===
    os.makedirs("results", exist_ok=True)
    plot_curve(rewards, title="Training Reward Curve", save_path="results/reward_curve.png")

if __name__ == "__main__":
    main()