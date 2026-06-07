"""
test_ll_conformity.py
=====================
Conformity tests for the PufferLib LunarLander port.

Compares the PufferLib C implementation step-by-step against the
reference Gymnasium LunarLander-v3 on a set of deterministic action
sequences and random rollouts.

Reward objectives (REWARD_DIM = 2):
    0  REW_LANDING  — shaping delta + landing/crash bonus
    1  REW_FUEL     — negative fuel cost

Actions:
    0  nothing
    1  left thruster
    2  main engine
    3  right thruster

Observations (8-dim):
    0  x position (normalised)
    1  y position (normalised)
    2  vx (normalised)
    3  vy (normalised)
    4  angle [rad]
    5  angular velocity (normalised)
    6  left leg contact {0,1}
    7  right leg contact {0,1}
"""

import numpy as np
import gymnasium as gym

from pufferlib.ocean.lunar_lander.lunar_lander import LunarLander

# Absolute tolerance for reward comparison
# (float32 vs float64 arithmetic can introduce small differences)
ATOL = 1e-3

# Maximum number of steps per episode in Gymnasium LunarLander-v3
MAX_STEPS = 1000


# ─────────────────────────────────────────────────────────────────────
# Environment factories
# ─────────────────────────────────────────────────────────────────────

def make_reference(seed: int = 0):
    """Return a fresh Gymnasium LunarLander-v3 (no wind by default)."""
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    return env


def make_port(seed: int = 0, enable_wind: bool = False,
              wind_power: float = 15.0, turbulence_power: float = 1.5):
    """Return a fresh PufferLib LunarLander (num_envs=1)."""
    env = LunarLander(
        num_envs=1,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
    )
    env.reset(seed=seed)
    return env


# ─────────────────────────────────────────────────────────────────────
# Single-step wrappers — normalise both APIs to (obs, reward_vec, ended)
# ─────────────────────────────────────────────────────────────────────

def reference_step(env, action: int):
    """
    Step the Gymnasium reference env.

    Returns
    -------
    obs      : np.ndarray, float64, shape (8,)
    reward   : float64   — scalar landing reward only (Gymnasium is SO)
    ended    : bool
    """
    obs, reward, terminated, truncated, _ = env.step(action)
    ended = bool(terminated or truncated)
    return (
        np.asarray(obs, dtype=np.float64),
        float(reward),
        ended,
    )


def port_step(env, action: int):
    """
    Step the PufferLib port (num_envs=1).

    Returns
    -------
    obs         : np.ndarray, float64, shape (8,)
    reward_vec  : np.ndarray, float64, shape (2,)  [landing, fuel]
    ended       : bool
    """
    import pufferlib.ocean.lunar_lander.binding as binding
    env.actions[:] = np.array([action], dtype=env.actions.dtype)
    binding.vec_step(env.c_envs)
    obs        = env.observations[0].astype(np.float64).copy()
    reward_vec = env.rewards[0].astype(np.float64).copy()   # shape (2,)
    ended      = bool(env.terminals[0])
    return obs, reward_vec, ended


# ─────────────────────────────────────────────────────────────────────
# Core comparison engine
# ─────────────────────────────────────────────────────────────────────

def compare_sequence(actions: list, label: str,
                     seed: int = 0,
                     check_obs: bool = True,
                     check_fuel_sign: bool = True) -> bool:
    """
    Replay *actions* on both environments and collect mismatches.

    Parameters
    ----------
    actions          : list of int  (0-3)
    label            : name shown in output
    seed             : reset seed for both envs
    check_obs        : compare observations at every step
    check_fuel_sign  : verify REW_FUEL <= 0 at every step

    Notes
    -----
    * The reference Gymnasium env returns a *scalar* landing reward.
      We therefore compare only rewards[REW_LANDING] from the port.
    * Observations after a terminal step may differ (the port
      auto-resets); we skip that comparison when both agree on ended=True.
    """
    ref  = make_reference(seed=seed)
    port = make_port(seed=seed)

    # ── fix scalarisation weights so random sampling doesn't interfere ──
    port.set_weights(np.array([1.0, 0.0], dtype=np.float32))

    # ── initial observation check ──
    ref_obs0, _ = ref.reset(seed=seed)
    port.reset(seed=seed)

    ref_obs0  = np.asarray(ref_obs0, dtype=np.float64)
    port_obs0 = port.observations[0].astype(np.float64)

    mismatches = []

    if check_obs and not np.allclose(ref_obs0, port_obs0, atol=ATOL):
        mismatches.append((-1, "init_obs", ref_obs0.tolist(), port_obs0.tolist()))

    # ── step loop ──
    for t, a in enumerate(actions):
        r_obs, r_rew_scalar, r_end = reference_step(ref, a)
        p_obs, p_rew_vec,    p_end = port_step(port, a)

        p_rew_landing = p_rew_vec[0]   # REW_LANDING
        p_rew_fuel    = p_rew_vec[1]   # REW_FUEL

        # Observation comparison (skip post-terminal since envs auto-reset)
        both_ended = r_end and p_end
        if check_obs and not both_ended:
            if not np.allclose(r_obs, p_obs, atol=ATOL):
                mismatches.append((t, "obs", r_obs.tolist(), p_obs.tolist()))

        # Landing reward comparison
        if not np.isclose(r_rew_scalar, p_rew_landing, atol=ATOL):
            mismatches.append((t, "rew_landing",
                               round(r_rew_scalar, 6),
                               round(p_rew_landing, 6)))

        # Fuel reward must be <= 0 (cost, never positive)
        if check_fuel_sign and p_rew_fuel > 1e-6:
            mismatches.append((t, "rew_fuel_sign", "<=0", round(p_rew_fuel, 6)))

        # Terminal flag comparison
        if r_end != p_end:
            mismatches.append((t, "ended", r_end, p_end))

        # Reset reference if episode ended (port auto-resets)
        if r_end:
            ref.reset(seed=seed)
        if p_end:
            port.reset(seed=seed)

    ref.close()
    port.close()

    if mismatches:
        print(f"[{label}] FAILED — {len(mismatches)} mismatch(es):")
        for t, kind, rv, pv in mismatches[:20]:
            print(f"    step {t:4d}  {kind:20s}  ref={rv}  port={pv}")
        return False

    print(f"[{label}] PASSED ({len(actions)} steps)")
    return True


# ─────────────────────────────────────────────────────────────────────
# Individual test scenarios
# ─────────────────────────────────────────────────────────────────────

def test_do_nothing():
    """
    Test 1 — full episode with action=0 (no thrusters).
    The lander falls under gravity alone; no fuel cost incurred.
    Verifies basic physics integration and gravity.
    """
    actions = [0] * 200
    return compare_sequence(actions, "do_nothing_200")


def test_main_engine_only():
    """
    Test 2 — fire main engine every step.
    Checks that:
    * REW_FUEL <= 0 every step (main engine costs 0.30 units)
    * Landing reward reflects upward thrust shaping
    """
    actions = [2] * 150
    return compare_sequence(actions, "main_engine_only_150")


def test_side_thrusters():
    """
    Test 3 — alternate left/right thrusters.
    Verifies torque application and angular velocity accumulation.
    Side engine cost = 0.03 units/step.
    """
    actions = ([1, 3] * 75)
    return compare_sequence(actions, "alternate_side_thrusters_150")


def test_landing_attempt():
    """
    Test 4 — simple heuristic landing sequence.
    Fire main engine to brake descent, then do nothing.
    This frequently triggers the +100 landing bonus or the -100 crash
    penalty, exercising the terminal reward paths.
    """
    # Brake with main engine, then coast
    actions = [2] * 40 + [0] * 60 + [2] * 20 + [0] * 80
    return compare_sequence(actions, "landing_attempt")


def test_out_of_bounds():
    """
    Test 5 — push the lander sideways until |obs[0]| >= 1 (out-of-bounds).
    The env should terminate with rew_landing = -100.
    """
    # Side thrusters on one side to drift left
    actions = [1] * 500
    return compare_sequence(actions, "out_of_bounds_left")


def test_wall_boundary_right():
    """
    Test 6 — same as test 5 but drifting right.
    """
    actions = [3] * 500
    return compare_sequence(actions, "out_of_bounds_right")


def test_random_actions_short():
    """
    Test 7 — 1 000 random actions (seed 42).
    General robustness test covering all action types.
    """
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 4, size=1000).tolist()
    return compare_sequence(actions, "random_1000_seed42")


def test_random_actions_long():
    """
    Test 8 — 3 000 random actions (seed 2024) with a bias toward
    main-engine (action 2) to produce more landing events.
    Also exercises the 1 000-step truncation boundary.
    """
    rng = np.random.default_rng(2024)
    actions = rng.choice([0, 1, 2, 3], size=3000,
                         p=[0.20, 0.20, 0.40, 0.20]).tolist()
    return compare_sequence(actions, "random_3000_bias_main_engine")


def test_fuel_reward_no_action():
    """
    Test 9 — verify fuel reward is exactly 0.0 when action=0.
    With no engine firing: rew_fuel = -(0*0.30 + 0*0.03) = 0.
    """
    port = make_port(seed=7)
    port.set_weights(np.array([0.0, 1.0], dtype=np.float32))
    port.reset(seed=7)

    import pufferlib.ocean.lunar_lander.binding as binding

    failures = []
    for t in range(100):
        port.actions[:] = np.array([0], dtype=port.actions.dtype)
        binding.vec_step(port.c_envs)
        fuel_rew = float(port.rewards[0, 1])
        if abs(fuel_rew) > 1e-6:
            failures.append((t, fuel_rew))
        if port.terminals[0]:
            port.reset(seed=7)

    port.close()

    label = "fuel_reward_zero_on_no_action"
    if failures:
        print(f"[{label}] FAILED — {len(failures)} steps with non-zero fuel reward:")
        for t, v in failures[:10]:
            print(f"    step {t:3d}  fuel_rew={v:.6f}")
        return False
    print(f"[{label}] PASSED (100 steps, fuel reward = 0 throughout)")
    return True


def test_set_weights_respected():
    """
    Test 10 — verify that set_weights() prevents the Dirichlet
    re-sampling at episode reset.
    After calling set_weights([0.8, 0.2]), the weights must remain
    unchanged across multiple episode boundaries.
    """
    port = make_port(seed=99)
    target = np.array([0.8, 0.2], dtype=np.float32)
    port.set_weights(target)
    port.reset(seed=99)

    import pufferlib.ocean.lunar_lander.binding as binding

    failures = []
    for t in range(500):
        port.actions[:] = np.array([0], dtype=port.actions.dtype)
        binding.vec_step(port.c_envs)
        w = port.weights[0].copy()
        if not np.allclose(w, target, atol=1e-6):
            failures.append((t, w.tolist()))
        if port.terminals[0]:
            port.reset(seed=99)

    port.close()

    label = "set_weights_respected_across_resets"
    if failures:
        print(f"[{label}] FAILED — weights changed at {len(failures)} step(s):")
        for t, w in failures[:5]:
            print(f"    step {t:3d}  weights={w}")
        return False
    print(f"[{label}] PASSED (weights stable for 500 steps, {sum(1 for _ in failures)} drift events)")
    return True


def test_observation_bounds():
    """
    Test 11 — run 2 000 random steps and verify the observation
    vector is always finite (no NaN / Inf).
    This catches integration blow-ups.
    """
    port = make_port(seed=13)

    import pufferlib.ocean.lunar_lander.binding as binding

    rng = np.random.default_rng(13)
    actions = rng.integers(0, 4, size=2000).tolist()
    failures = []

    for t, a in enumerate(actions):
        port.actions[:] = np.array([a], dtype=port.actions.dtype)
        binding.vec_step(port.c_envs)
        obs = port.observations[0].copy()
        if not np.all(np.isfinite(obs)):
            failures.append((t, obs.tolist()))
        if port.terminals[0]:
            port.reset(seed=13)

    port.close()

    label = "observation_always_finite"
    if failures:
        print(f"[{label}] FAILED — non-finite obs at {len(failures)} step(s):")
        for t, o in failures[:5]:
            print(f"    step {t:3d}  obs={o}")
        return False
    print(f"[{label}] PASSED (2 000 steps, all observations finite)")
    return True


def test_wind_mode_no_crash():
    """
    Test 12 — enable wind and run 1 000 random steps.
    Verifies the wind path does not cause NaN/Inf or assertion errors.
    (Exact numerical match with reference not expected — wind uses
    continuous sinusoidal indices that diverge from Gymnasium's numpy RNG.)
    """
    port = LunarLander(
        num_envs=1,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
    )
    port.set_weights(np.array([0.5, 0.5], dtype=np.float32))
    port.reset(seed=0)

    import pufferlib.ocean.lunar_lander.binding as binding

    rng = np.random.default_rng(0)
    actions = rng.integers(0, 4, size=1000).tolist()
    failures = []

    for t, a in enumerate(actions):
        port.actions[:] = np.array([a], dtype=port.actions.dtype)
        binding.vec_step(port.c_envs)
        obs = port.observations[0].copy()
        rew = port.rewards[0].copy()
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(rew)):
            failures.append((t, obs.tolist(), rew.tolist()))
        if port.terminals[0]:
            port.reset(seed=0)

    port.close()

    label = "wind_mode_stable"
    if failures:
        print(f"[{label}] FAILED — non-finite values at {len(failures)} step(s):")
        for t, o, r in failures[:5]:
            print(f"    step {t:3d}  obs={o}  rew={r}")
        return False
    print(f"[{label}] PASSED (1 000 steps with wind enabled)")
    return True


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_do_nothing,
        test_main_engine_only,
        test_side_thrusters,
        test_landing_attempt,
        test_out_of_bounds,
        test_wall_boundary_right,
        test_random_actions_short,
        test_random_actions_long,
        test_fuel_reward_no_action,
        test_set_weights_respected,
        test_observation_bounds,
        test_wind_mode_no_crash,
    ]

    results = [t() for t in tests]

    print()
    if all(results):
        print(f"ALL {len(results)} TESTS PASSED")
    else:
        n_fail = results.count(False)
        print(f"{n_fail}/{len(results)} TESTS FAILED")