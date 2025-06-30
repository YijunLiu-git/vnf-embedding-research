def multi_vnf_reward(successes, delays, min_bandwidths, hops_list, jitters, losses):
    # QoS Baseline Thresholds
    max_total_delay = 100.0
    min_required_bandwidth = 5.0
    max_avg_jitter = 1.0
    max_avg_loss = 0.1
    max_hops = 10.0

    # Aggregates
    total_delay = sum(delays)
    avg_min_bandwidth = sum(min_bandwidths) / len(min_bandwidths) if min_bandwidths else 0
    avg_hops = sum(hops_list) / len(hops_list) if hops_list else 0
    avg_jitter = sum(jitters) / len(jitters) if jitters else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    success_ratio = sum(successes) / len(successes) if successes else 0

    # Non-linear reward amplification
    delay_reward = (1.0 - min(total_delay / max_total_delay, 1.0)) ** 2
    bandwidth_reward = min(avg_min_bandwidth / min_required_bandwidth, 1.0) ** 1.5
    hop_reward = (1.0 - min(avg_hops / max_hops, 1.0)) ** 1.2
    jitter_penalty = (1.0 - min(avg_jitter / max_avg_jitter, 1.0)) ** 1.5
    loss_penalty = (1.0 - min(avg_loss / max_avg_loss, 1.0)) ** 1.5

    # Adjustable weights
    w_delay = 0.3
    w_bandwidth = 0.25
    w_hops = 0.15
    w_jitter = 0.15
    w_loss = 0.15

    total_reward = (
        w_delay * delay_reward +
        w_bandwidth * bandwidth_reward +
        w_hops * hop_reward +
        w_jitter * jitter_penalty +
        w_loss * loss_penalty
    )

    # Penalize failure softly
    if success_ratio == 1.0:
        reward_scale = 1.0
    elif success_ratio >= 0.5:
        reward_scale = 0.5
    else:
        reward_scale = 0.1  # severe penalty for most failed chains

    total_reward *= reward_scale

    return total_reward