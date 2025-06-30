
# rewards/reward_v4_comprehensive.py
def compute_reward(info, reward_config):
    alpha = reward_config.get("alpha", 0.5)   # bandwidth weight
    beta = reward_config.get("beta", 0.2)     # delay weight
    gamma = reward_config.get("gamma", 0.2)   # jitter weight
    delta = reward_config.get("delta", 0.1)   # loss weight
    penalty = reward_config.get("penalty", 0.5)  # failure penalty
    hop_weight = reward_config.get("hop_weight", 0.1)  # hop number penalty

    if not info["paths"]:
        return -penalty  # 失败时直接给予惩罚

    total_reward = 0.0
    for path in info["paths"]:
        bw = path["bandwidth"]
        delay = path["delay"]
        jitter = path["jitter"]
        loss = path["loss"]
        hops = path["hops"]

        # 奖励公式：越低越好 delay/jitter/loss，带宽越高越好
        path_reward = (
            alpha * bw -
            beta * delay -
            gamma * jitter -
            delta * loss -
            hop_weight * hops
        )
        total_reward += path_reward

    return total_reward / len(info["paths"])