def enhanced_reward(success, delay, bandwidth):
    if not success:
        return -10
    # delay 越小越好，bandwidth 越大越好
    reward = (10 - delay) + (bandwidth * 5)
    return reward