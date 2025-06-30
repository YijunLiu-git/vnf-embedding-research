# rewards/reward_v2_hierarchical.py

def hierarchical_reward(success, delay, bandwidth, hops, jitter=0.0, loss=0.0):
    """
    综合奖励函数：融合多个网络参数
    """
    # 权重可以根据需求微调
    weights = {
        "success": 1.0,
        "bandwidth": 0.6,
        "delay": -0.5,
        "hops": -0.3,
        "jitter": -0.4,
        "loss": -0.7
    }

    reward = 0.0

    # 成功/失败
    reward += weights["success"] if success else -1.0 * weights["success"]

    # 带宽越高越好
    reward += weights["bandwidth"] * bandwidth

    # 延迟越低越好
    reward += weights["delay"] * delay

    # 跳数越少越好
    reward += weights["hops"] * hops

    # 抖动越低越好
    reward += weights["jitter"] * jitter

    # 丢包率越低越好
    reward += weights["loss"] * loss

    return reward