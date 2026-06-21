import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pufferlib.ocean.pommerman.pommerman import Pommerman, NUM_ACTIONS, NUM_AGENTS_PER_ENV

N_ENVS    = 100
TIMEOUT   = 10  # secondes


def benchmark_port(n_envs: int, timeout: int) -> float:
    env = Pommerman(num_envs=n_envs)
    env.reset()

    atn_cache = 512
    actions = np.random.randint(
        0, NUM_ACTIONS,
        (atn_cache, n_envs * NUM_AGENTS_PER_ENV),
    )

    tick = 0
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        env.step(actions[tick % atn_cache])
        tick += 1
    elapsed = time.perf_counter() - start

    env.close()
    total_agent_steps = n_envs * NUM_AGENTS_PER_ENV * tick
    return total_agent_steps / elapsed


if __name__ == "__main__":
    print("=" * 55)
    print("  BENCHMARK — Pommerman C (PufferMO)")
    print(f"  {N_ENVS} envs · {TIMEOUT}s · {NUM_AGENTS_PER_ENV} agents/env")
    print("=" * 55)

    print(f"\nRunning for {TIMEOUT} seconds...")
    sps = benchmark_port(N_ENVS, TIMEOUT)
    print(f"\n  SPS (agent steps/sec) : {sps:>14,.0f}")
    print(f"  SPS (env steps/sec)   : {sps/NUM_AGENTS_PER_ENV:>14,.0f}")
    print("=" * 55)