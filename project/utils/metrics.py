# utils/metrics.py
def compute_success_rate(results):
    successes = sum(1 for r in results if r['success'])
    return successes / len(results) if results else 0.0

def compute_avg_reward(results):
    rewards = [r['reward'] for r in results]
    return sum(rewards) / len(rewards) if rewards else 0.0

def compute_avg_latency(results):
    latencies = [r['latency'] for r in results if 'latency' in r]
    return sum(latencies) / len(latencies) if latencies else 0.0

def calculate_sar(env):
    """
    Service Acceptance Rate (SAR): Ratio of successfully embedded VNFs to the total in service chain
    """
    if hasattr(env, "embedding_map") and hasattr(env, "service_chain"):
        embedded = len(env.embedding_map)
        total = len(env.service_chain)
        return embedded / total if total > 0 else 0.0
    return 0.0

def calculate_splat(env):
    """
    Average Per-Link Latency (SPLat) for the embedded paths
    """
    if not hasattr(env, "path_records") or not env.path_records:
        return 0.0

    total_delay = 0.0
    total_hops = 0

    for path in env.path_records:
        delay = path.get("delay", 0.0)
        hops = path.get("hops", 1)
        total_delay += delay
        total_hops += hops

    return total_delay / total_hops if total_hops > 0 else 0.0