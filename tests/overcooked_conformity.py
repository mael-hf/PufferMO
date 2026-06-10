"""
Conformity Test : OvercookedEnv (C) vs overcooked_ai_py (reference)
====================================================================
Validates game logic by comparing:
1. Wall blocking  — agents cannot move through walls
2. Noop           — no-op action yields zero reward
3. Episode end    — terminates at max_ticks
4. Trajectory     — agent positions match reference state-by-state
"""

import numpy as np
import sys
import os

# --- Reference import ---
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as RefEnv
from overcooked_ai_py.mdp.actions import Action

# --- C port import ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pufferlib.ocean.overcooked.overcooked import OvercookedEnv as PortEnv

# --- Action mapping: C -> reference ---
# C  : 0=noop, 1=up, 2=down, 3=left, 4=right, 5=interact
# Ref: (0,-1)=up, (0,1)=down, (1,0)=right, (-1,0)=left, (0,0)=noop, 'interact'
C_TO_REF = {
    0: Action.ALL_ACTIONS[4],   # noop    -> (0,0)
    1: Action.ALL_ACTIONS[0],   # up      -> (0,-1)
    2: Action.ALL_ACTIONS[1],   # down    -> (0,1)
    3: Action.ALL_ACTIONS[3],   # left    -> (-1,0)
    4: Action.ALL_ACTIONS[2],   # right   -> (1,0)
    5: Action.ALL_ACTIONS[5],   # interact
}

LAYOUT    = "cramped_room"
LAYOUT_W  = 5   # cramped_room is 5x5
LAYOUT_H  = 5
MAX_TICKS = 400
OBS_X_IDX = 40  # index of x/width  in observation vector
OBS_Y_IDX = 41  # index of y/height in observation vector

PASSED = []
FAILED = []


def make_ref():
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    return RefEnv.from_mdp(mdp, horizon=MAX_TICKS)


def make_port():
    return PortEnv(layout=0, num_agents=2, grid_size=32, max_ticks=MAX_TICKS)


def ref_step(env, actions_c):
    ref_actions = [C_TO_REF[a] for a in actions_c]
    state, reward, done, info = env.step(ref_actions)
    return state, reward, done


def port_step(env, actions_c):
    obs, rewards, terminals, truncations, info = env.step(actions_c)
    done  = bool(terminals[0])
    reward = float(rewards[0]) + float(rewards[1])
    return obs, reward, done


def get_port_positions(obs):
    """Extract (x, y) integer positions from the C env observation vector."""
    positions = []
    for i in range(2):
        x = round(obs[i][OBS_X_IDX] * LAYOUT_W)
        y = round(obs[i][OBS_Y_IDX] * LAYOUT_H)
        positions.append((y, x))  # swap x and y to match reference convention
    return positions


def get_ref_positions(state):
    """Extract (x, y) integer positions from the reference state."""
    return [state.players[i].position for i in range(2)]


def run_test(name, actions_sequence):
    """Run a sequence of actions on both envs and compare rewards + done."""
    ref  = make_ref()
    port = make_port()
    ref.reset()
    port.reset()

    mismatches     = []
    cumulative_ref  = 0.0
    cumulative_port = 0.0

    for t, actions in enumerate(actions_sequence):
        r_state, r_rew, r_done = ref_step(ref, actions)
        p_obs,   p_rew, p_done = port_step(port, actions)

        cumulative_ref  += r_rew
        cumulative_port += p_rew

        if r_rew != 0 or p_rew != 0:
            if abs(r_rew - p_rew) > 1e-3:
                mismatches.append((t, "reward", r_rew, p_rew))

        if r_done != p_done:
            mismatches.append((t, "done", r_done, p_done))

        if r_done:
            ref.reset()
        if p_done:
            port.reset()

    port.close()

    if mismatches:
        print(f"  [FAIL] {name} — {len(mismatches)} mismatch(es):")
        for t, kind, rv, pv in mismatches[:5]:
            print(f"         step {t:4d}  {kind:8s}  ref={rv}  port={pv}")
        FAILED.append(name)
        return False
    else:
        print(f"  [PASS] {name} — cumulative reward  ref={cumulative_ref:.1f}  port={cumulative_port:.1f}")
        PASSED.append(name)
        return True


# ============================================================
# TEST 1 — Wall blocking
# Both agents move up (into a wall) for 10 steps → no movement
# ============================================================
def test_wall_blocking():
    print("\n[TEST 1] Wall blocking")
    actions = [[1, 1]] * 10
    return run_test("wall_blocking", actions)


# ============================================================
# TEST 2 — Noop
# No-op for 20 steps → reward must stay 0
# ============================================================
def test_noop():
    print("\n[TEST 2] Noop — zero reward")
    actions = [[0, 0]] * 20
    return run_test("noop_no_reward", actions)


# ============================================================
# TEST 3 — Episode termination
# Episode must end at max_ticks (tolerance ±1)
# ============================================================
def test_episode_end():
    print("\n[TEST 3] Episode termination at max_ticks")
    ref  = make_ref()
    port = make_port()
    ref.reset()
    port.reset()

    ref_done_at  = None
    port_done_at = None

    for t in range(MAX_TICKS + 10):
        _, _, r_done = ref_step(ref, [0, 0])
        _, _, p_done = port_step(port, [0, 0])

        if r_done and ref_done_at is None:
            ref_done_at = t
            ref.reset()
        if p_done and port_done_at is None:
            port_done_at = t
            port.reset()

        if ref_done_at and port_done_at:
            break

    port.close()

    if ref_done_at is None:
        print("  [FAIL] episode_end — reference never terminates")
        FAILED.append("episode_end")
        return False
    if port_done_at is None:
        print("  [FAIL] episode_end — C port never terminates")
        FAILED.append("episode_end")
        return False

    if abs(ref_done_at - port_done_at) <= 1:
        print(f"  [PASS] episode_end — ref ends at step {ref_done_at}, port at step {port_done_at}")
        PASSED.append("episode_end")
        return True
    else:
        print(f"  [FAIL] episode_end — ref={ref_done_at}  port={port_done_at}")
        FAILED.append("episode_end")
        return False


# ============================================================
# TEST 4 — Trajectory conformity
# For the same sequence of actions, agent positions must match
# the reference state step by step.
# ============================================================
def test_trajectory():
    print("\n[TEST 4] Trajectory conformity (agent positions, 100 steps)")
    rng     = np.random.default_rng(42)
    actions = rng.integers(0, 5, size=(100, 2)).tolist()

    ref  = make_ref()
    port = make_port()
    ref.reset()
    port.reset()

    mismatches = []

    for t, act in enumerate(actions):
        r_state, _, r_done = ref_step(ref, act)
        p_obs,   _, p_done = port_step(port, act)

        ref_pos  = get_ref_positions(r_state)
        port_pos = get_port_positions(p_obs)

        for agent_i in range(2):
            if ref_pos[agent_i] != port_pos[agent_i]:
                mismatches.append((t, agent_i, ref_pos[agent_i], port_pos[agent_i]))

        if r_done:
            ref.reset()
        if p_done:
            port.reset()

    port.close()

    if mismatches:
        print(f"  [FAIL] trajectory — {len(mismatches)} position mismatch(es):")
        for t, agent_i, rv, pv in mismatches[:5]:
            print(f"         step {t:3d}  agent {agent_i}  ref={rv}  port={pv}")
        FAILED.append("trajectory")
        return False
    else:
        print(f"  [PASS] trajectory — all {len(actions)} steps matched")
        PASSED.append("trajectory")
        return True


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  CONFORMITY TEST — Overcooked C vs reference")
    print("=" * 55)

    test_wall_blocking()
    test_noop()
    test_episode_end()
    test_trajectory()

    print("\n" + "=" * 55)
    total = len(PASSED) + len(FAILED)
    if not FAILED:
        print(f"  RESULT : {len(PASSED)}/{total} tests PASSED ✓")
    else:
        print(f"  RESULT : {len(PASSED)}/{total} tests passed")
        print(f"  Failed : {', '.join(FAILED)}")
    print("=" * 55)