# rewards/reward_v3_composite.py

def composite_reward(success, delay, min_bandwidth, hops, max_delay=100, max_hops=10):
    """
    层次型复合奖励函数：
    - success: 是否成功
    - delay: 累积延迟（越小越好）
    - min_bandwidth: 路径中最小带宽（越大越好）
    - hops: 路径跳数（越少越好）
    """

    weights = {
        "success": 1.0,
        "delay": 0.3,
        "bandwidth": 0.4,
        "hops": 0.3,
        "penalty_failed": -5.0,
        "bonus_success": 2.0
    }

    if not success:
        return weights["penalty_failed"]  # 硬惩罚失败

    # 各维度 reward（归一化到 0-1 区间）
    delay_reward = 1.0 - min(delay / max_delay, 1.0)  # 越小越好
    bandwidth_reward = min_bandwidth / 10.0  # 假设最大带宽为 10
    hops_reward = 1.0 - min(hops / max_hops, 1.0)  # 越小越好

    total = (
        weights["success"] * 1.0 +
        weights["delay"] * delay_reward +
        weights["bandwidth"] * bandwidth_reward +
        weights["hops"] * hops_reward +
        weights["bonus_success"]
    )

    return total