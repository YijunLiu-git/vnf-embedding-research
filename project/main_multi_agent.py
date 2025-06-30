import os
import torch
import random
import numpy as np
import yaml

from agents.multi_ddqn_agent import MultiDDQNAgent
from agents.multi_dqn_agent import MultiDQNAgent
from agents.multi_ppo_agent import MultiPPOAgent
from env.vnf_env_multi import MultiVNFEmbeddingEnv
from rewards.reward_v4_comprehensive_multi import compute_reward
from utils.topology import generate_topology
from utils.visualization import plot_multi_curve, plot_success_rate, save_csv
from utils.metrics import calculate_sar, calculate_splat
from utils.logger import Logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_agent(agent_class, name, config, graph, node_features, edge_features):
    print(f"[INFO] Training agent: {name.upper()}")

    env = MultiVNFEmbeddingEnv(
        graph, node_features, edge_features, reward_config=config["reward"]
    )

    agent = agent_class(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        edge_dim=config["gnn"]["edge_dim"],
        config=config
    )

    episodes = config["train"]["episodes"]
    reward_list, success_list, sar_list, splat_list = [], [], [], []

    os.makedirs("results", exist_ok=True)
    logger = Logger(log_dir="results", agent_name=name, filename=f"{name}_log.csv")

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        action = agent.select_action(state)
        next_state, env_reward, done, info = env.step(action)
        reward = compute_reward(info, config["reward"])

        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        total_reward += reward
        success = info.get("success", False)

        reward_list.append(total_reward)
        success_list.append(1 if success else 0)

        sar = calculate_sar(env)
        splat = calculate_splat(env)
        sar_list.append(sar)
        splat_list.append(splat)

        logger.log(ep + 1, total_reward, success, sar, splat)

        if (ep + 1) % 10 == 0:
            print(f"[{name}] Ep {ep+1} | R={total_reward:.2f} | Success={success} | SAR={sar:.3f} | SPLat={splat:.3f}")

    return reward_list, success_list, sar_list, splat_list

def main():
    # 加载配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 拓扑生成
    graph, node_features, edge_features = generate_topology(config["topology"])

    set_seed(42)

    agent_dict = {
        "ddqn": MultiDDQNAgent,
        "dqn": MultiDQNAgent,
        "ppo": MultiPPOAgent,
    }

    results = {
        "rewards": {},
        "successes": {},
        "sars": {},
        "splats": {}
    }

    for name, agent_cls in agent_dict.items():
        rewards, successes, sars, splats = run_agent(agent_cls, name, config, graph, node_features, edge_features)
        results["rewards"][name] = rewards
        results["successes"][name] = successes
        results["sars"][name] = sars
        results["splats"][name] = splats

    timestamp = "results/compare_multi"
    plot_multi_curve(results["rewards"], save_path=timestamp + "_rewards")
    plot_success_rate(results["successes"], save_path=timestamp + "_success")
    plot_multi_curve(results["sars"], save_path=timestamp + "_sar")
    plot_multi_curve(results["splats"], save_path=timestamp + "_splat")

    save_csv(results["rewards"], timestamp + "_rewards.csv")
    save_csv(results["successes"], timestamp + "_success.csv")
    save_csv(results["sars"], timestamp + "_sar.csv")
    save_csv(results["splats"], timestamp + "_splat.csv")

if __name__ == '__main__':
    main()