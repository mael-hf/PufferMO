"""
Benchmark de performance : OvercookedEnv (C) vs overcooked_ai_py (référence Python)
=====================================================================================
"""

import time
import numpy as np
import sys
import os

# --- Import référence Python ---
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as RefEnv
from overcooked_ai_py.mdp.actions import Action

# --- Import notre port C ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pufferlib.ocean.overcooked.overcooked import OvercookedEnv as PortEnv

# --- Config ---
LAYOUT    = "cramped_room"
MAX_TICKS = 400
N_STEPS_C   = 100_000
N_STEPS_REF = 2_000

# Mapping actions C -> référence
C_TO_REF = {
    0: Action.ALL_ACTIONS[4],  # noop
    1: Action.ALL_ACTIONS[0],  # haut
    2: Action.ALL_ACTIONS[1],  # bas
    3: Action.ALL_ACTIONS[3],  # gauche
    4: Action.ALL_ACTIONS[2],  # droite
    5: Action.ALL_ACTIONS[5],  # interact
}


def benchmark_port(n_steps: int) -> float:
    env = PortEnv(layout=0, num_agents=2, grid_size=32, max_ticks=MAX_TICKS)
    env.reset()
    rng = np.random.default_rng(42)

    start = time.perf_counter()
    for _ in range(n_steps):
        actions = rng.integers(0, 5, size=2).tolist()
        obs, rewards, terminals, truncations, _ = env.step(actions)
        if bool(terminals[0]) or bool(truncations[0]):
            env.reset()
    elapsed = time.perf_counter() - start

    env.close()
    return n_steps / elapsed


def benchmark_reference(n_steps: int) -> float:
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = RefEnv.from_mdp(mdp, horizon=MAX_TICKS)
    env.reset()
    rng = np.random.default_rng(42)

    start = time.perf_counter()
    for _ in range(n_steps):
        actions = [C_TO_REF[a] for a in rng.integers(0, 5, size=2).tolist()]
        _, _, done, _ = env.step(actions)
        if done:
            env.reset()
    elapsed = time.perf_counter() - start

    # pas de env.close() sur la référence Python
    return n_steps / elapsed


if __name__ == "__main__":
    print("=" * 55)
    print("  BENCHMARK — Overcooked C vs référence Python")
    print(f"  C: {N_STEPS_C} steps | Ref: {N_STEPS_REF} steps · layout: {LAYOUT}")
    print("=" * 55)

    print("\n[1/2] Testing C environment performance...")
    sps_port = benchmark_port(N_STEPS_C)
    print(f"      C port SPS       : {sps_port:>12,.0f}")

    print("\n[2/2] Testing Python reference performance...")
    sps_ref = benchmark_reference(N_STEPS_REF)
    print(f"      Python ref SPS   : {sps_ref:>12,.0f}")

    speedup = sps_port / sps_ref
    print("\n" + "=" * 55)
    print(f"  Speedup C / Python  : {speedup:.1f}x")
    print("=" * 55)