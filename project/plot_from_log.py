import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_logs(log_paths):
    data = defaultdict(lambda: {"episode": [], "reward": [], "sar": [], "splat": []})

    for path in log_paths:
        agent_name = os.path.basename(path).replace("_log.csv", "")
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[agent_name]["episode"].append(int(row["episode"]))
                data[agent_name]["reward"].append(float(row["reward"]))
                data[agent_name]["sar"].append(float(row["sar"]))
                data[agent_name]["splat"].append(float(row["splat"]))
    return data

def smooth(values, window=10):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def plot_metric(data, metric, ylabel, save_path):
    plt.figure(figsize=(8, 5))  # 更适合论文
    colors = {"ddqn": "tab:blue", "ppo": "tab:orange", "dqn": "tab:green"}

    for agent, values in data.items():
        if len(values[metric]) == 0:
            continue
        smoothed = smooth(values[metric], window=10)
        episodes = values["episode"][:len(smoothed)]
        color = colors.get(agent.lower(), None)
        plt.plot(episodes, smoothed, label=agent.upper(), linewidth=2, color=color)

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} over Episodes", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()
    
def main():
    log_dir = "results"
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith("_log.csv")]
    data = load_logs(log_files)

    plot_metric(data, "reward", "Total Reward", os.path.join(log_dir, "reward_comparison.png"))
    plot_metric(data, "sar", "Service Acceptance Rate (SAR)", os.path.join(log_dir, "sar_comparison.png"))
    plot_metric(data, "splat", "Avg Single-Path Latency (SPLat)", os.path.join(log_dir, "splat_comparison.png"))

if __name__ == "__main__":
    main()