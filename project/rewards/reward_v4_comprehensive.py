# rewards/reward_v4_comprehensive.py

def comprehensive_reward(
    success,
    delay,
    min_bandwidth,
    hops,
    jitter,
    packet_loss,
    link_util,
    reliability,
    max_delay=100,
    max_bandwidth=10,
    max_hops=10
):
    """
    综合奖励函数：考虑成功率、延迟、带宽、跳数、抖动、丢包、利用率、可靠性
    """

    weights = {
        "success": 1.0,
        "delay": 0.2,
        "bandwidth": 0.2,
        "hops": 0.1,
        "jitter": 0.1,
        "packet_loss": 0.2,
        "link_util": 0.1,
        "reliability": 0.1,
        "bonus_success": 2.0,
        "penalty_fail": -5.0
    }

    if not success:
        return weights["penalty_fail"]

    # 各指标归一化 reward（0~1之间）
    delay_reward = 1.0 - min(delay / max_delay, 1.0)
    bandwidth_reward = min(min_bandwidth / max_bandwidth, 1.0)
    hops_reward = 1.0 - min(hops / max_hops, 1.0)
    jitter_reward = 1.0 - min(jitter / 1.0, 1.0)
    packet_loss_reward = 1.0 - min(packet_loss / 0.1, 1.0)
    link_util_reward = 1.0 - min(link_util / 1.0, 1.0)
    reliability_reward = min(reliability, 1.0)

    total_reward = (
        weights["success"] * 1.0 +
        weights["delay"] * delay_reward +
        weights["bandwidth"] * bandwidth_reward +
        weights["hops"] * hops_reward +
        weights["jitter"] * jitter_reward +
        weights["packet_loss"] * packet_loss_reward +
        weights["link_util"] * link_util_reward +
        weights["reliability"] * reliability_reward +
        weights["bonus_success"]
    )

    return total_reward