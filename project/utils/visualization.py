import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

sns.set(style="whitegrid")

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_csv(data_dict, save_path):
    ensure_dir(save_path)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Episode'] + list(data_dict.keys())
        writer.writerow(header)

        max_len = max(len(v) for v in data_dict.values())
        for i in range(max_len):
            row = [i + 1]
            for v in data_dict.values():
                row.append(v[i] if i < len(v) else '')
            writer.writerow(row)

def timestamped_path(base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_path}_{timestamp}"

def plot_multi_curve(result_dict, save_path='results/reward_curve'):
    plt.figure(figsize=(10, 6))

    for label, rewards in result_dict.items():
        rewards = np.array(rewards)
        ma_rewards = moving_average(rewards)
        plt.plot(ma_rewards, label=f"{label} (avg)", linewidth=2)
        plt.scatter(len(rewards) - 1, rewards[-1], label=f"{label} final: {rewards[-1]:.2f}", s=30)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Agent Performance (Smoothed Reward)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ensure_dir(save_path)
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".svg")
    plt.close()

    save_csv(result_dict, save_path + ".csv")

def plot_success_rate(success_dict, save_path='results/success_rate_curve'):
    plt.figure(figsize=(10, 6))

    for label, successes in success_dict.items():
        successes = np.array(successes)
        avg_success = np.cumsum(successes) / (np.arange(len(successes)) + 1)
        plt.plot(avg_success, label=f"{label} (avg)", linewidth=2)
        plt.scatter(len(successes) - 1, avg_success[-1], label=f"{label} final: {avg_success[-1]*100:.1f}%", s=30)

    plt.xlabel("Episode")
    plt.ylabel("Average Success Rate")
    plt.title("Success Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ensure_dir(save_path)
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".svg")
    plt.close()

    save_csv(success_dict, save_path + ".csv")

# ✅ 新增：支持 main_compare.py 导入的保存函数
def save_results_to_csv(reward_dict, success_dict, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    for agent_name in reward_dict:
        rewards = reward_dict[agent_name]
        successes = success_dict.get(agent_name, [])

        df = pd.DataFrame({
            'Episode': list(range(1, len(rewards)+1)),
            'Reward': rewards,
            'Success': successes if len(successes) == len(rewards) else [None]*len(rewards)
        })

        df.to_csv(os.path.join(save_dir, f"{agent_name}_results.csv"), index=False)