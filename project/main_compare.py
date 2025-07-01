# import os
# import yaml
# from datetime import datetime
# from env.vnf_env import VNFEmbeddingEnv
# from utils.topology import generate_topology
# from utils.visualization import plot_multi_curve, plot_success_rate

# from agents.ddqn_agent import DDQNAgent
# from agents.dqn_agent import DQNAgent
# from agents.ppo_agent import PPOAgent


# def load_config(path='config/config.yaml'):
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)


# def run_agent(agent_class, agent_name, config, graph, node_features, edge_features):
#     print(f"[INFO] Training agent: {agent_name.upper()}")

#     env = VNFEmbeddingEnv(graph, node_features, edge_features)
#     state_dim = env.state_dim
#     action_dim = env.action_dim
#     edge_dim = edge_features.shape[1]

#     agent = agent_class(state_dim, action_dim, edge_dim, config)
#     episodes = config['train']['episodes']

#     reward_list = []
#     success_list = []

#     for episode in range(1, episodes + 1):
#         state = env.reset()
#         episode_reward = 0
#         done = False

#         while not done:
#             output = agent.select_action(state)
#             action = output[0] if isinstance(output, tuple) else output
#             log_prob = output[1] if isinstance(output, tuple) else None

#             if not (0 <= action < action_dim):
#                 print(f"[ERROR] Invalid action {action}. Skipping this episode.")
#                 episode_reward = -10
#                 break

#             next_state, reward, done, success = env.step(action)
#             success_list.append(int(success))

#             if isinstance(agent, PPOAgent):
#                 agent.store_transition(state, action, reward, next_state, done, log_prob)
#             else:
#                 agent.store_transition(state, action, reward, next_state, done)

#             state = next_state
#             episode_reward += reward

#         if hasattr(agent, "learn"):
#             agent.learn()
#         if hasattr(agent, "update_epsilon"):
#             agent.update_epsilon()

#         reward_list.append(episode_reward)
#         print(f"[{agent_name.upper()}] Episode {episode}/{episodes} - Reward: {episode_reward:.2f}")

#     return reward_list, success_list


# def main():
#     config = load_config()

#     # 时间戳目录
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     result_dir = os.path.join("results", timestamp)
#     os.makedirs(result_dir, exist_ok=True)

#     graph, node_features, edge_features = generate_topology(config['topology'])

#     result_dict = {}
#     success_dict = {}

#     for agent_class, name in [(DDQNAgent, "DDQN"), (DQNAgent, "DQN"), (PPOAgent, "PPO")]:
#         rewards, successes = run_agent(agent_class, name, config, graph, node_features, edge_features)
#         result_dict[name] = rewards
#         success_dict[name] = successes

#     # 可视化并保存到 timestamp 子目录
#     plot_multi_curve(result_dict, save_path=os.path.join(result_dir, "reward_curve"))
#     plot_success_rate(success_dict, save_path=os.path.join(result_dir, "success_rate_curve"))


# if __name__ == "__main__":
#     main()

import os
import torch
import random
import numpy as np
import yaml

from agents.multi_ddqn_agent import MultiDDQNAgent
from agents.multi_dqn_agent import MultiDQNAgent
from agents.multi_ppo_agent import MultiPPOAgent
from env.vnf_env_multi import MultiVNFEmbeddingEnv
from utils.topology import generate_topology
from utils.visualization import plot_multi_curve, plot_success_rate, save_csv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_agent(agent_class, name, config, graph, node_features, edge_features):
    print(f"[INFO] Training agent: {name.upper()}")

    env = MultiVNFEmbeddingEnv(graph, node_features, edge_features)
    agent = agent_class(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        edge_dim=config["gnn"]["edge_dim"],
        config=config
    )

    episodes = config["train"]["episodes"]
    reward_list, success_list = [], []

    for ep in range(episodes):
        state = env.reset()
        done = False
        rewards = 0

        # For multi-step action selection (one shot multi-VNF)
        action = agent.select_action(state)
        next_state, reward, done, success = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        rewards += reward

        reward_list.append(rewards)
        success_list.append(1 if success else 0)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {rewards:.2f} | Success: {success}")

    return reward_list, success_list

def main():
    # Load config.yaml file
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Generate topology from config
    graph, node_features, edge_features = generate_topology(config["topology"])

    set_seed(42)

    agent_dict = {
        "ddqn": MultiDDQNAgent,
        "dqn": MultiDQNAgent,
        "ppo": MultiPPOAgent,
    }

    result_rewards = {}
    result_successes = {}

    for name, agent_cls in agent_dict.items():
        rewards, successes = run_agent(agent_cls, name, config, graph, node_features, edge_features)
        result_rewards[name] = rewards
        result_successes[name] = successes

    timestamp = "results/compare_multi"
    plot_multi_curve(result_rewards, save_path=timestamp + "_rewards")
    plot_success_rate(result_successes, save_path=timestamp + "_success")
    save_csv(result_rewards, timestamp + "_rewards.csv")
    save_csv(result_successes, timestamp + "_success.csv")

if __name__ == '__main__':
    main()