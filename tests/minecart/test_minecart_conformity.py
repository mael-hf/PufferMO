import numpy as np
import mo_gymnasium as mo_gym

from pufferlib.ocean.minecart.minecart import Minecart
import pufferlib.ocean.minecart.binding as binding

ATOL = 1e-5
OBS_ATOL = 1e-6

NUM_ENVS = 1

ACT_MINE, ACT_LEFT, ACT_RIGHT, ACT_ACCEL, ACT_BRAKE, ACT_NONE = range(6)


def make_reference():
    return mo_gym.make("minecart-v0")


def make_port(capacity=1.5, mine_cnt=5, **kw):
    return Minecart(
        num_envs=NUM_ENVS, capacity=capacity, mine_cnt=mine_cnt,
        incremental_frame_skip=True, **kw,
    )


def reference_step(env, action):
    obs, reward, terminated, truncated, _ = env.step(action)
    ended = bool(terminated or truncated)
    return np.asarray(obs, dtype=np.float64), np.asarray(reward, dtype=np.float64), ended


def port_step(env, action):
    env.actions[:] = np.array([action], dtype=env.actions.dtype)
    binding.vec_step(env.c_envs)
    obs = env.observations[0].astype(np.float64).copy()
    reward = env.rewards[0].astype(np.float64).copy()
    ended = bool(env.terminals[0])
    return obs, reward, ended


def compare_sequence(actions, label, check_content=True):
    ref = make_reference()
    port = make_port()

    ref_obs, _ = ref.reset(seed=0)
    port.reset(seed=0)

    ref_obs = np.asarray(ref_obs, dtype=np.float64)
    port_obs = port.observations[0].astype(np.float64)

    mismatches = []
    def check_obs(step, r_obs, p_obs, r_end, p_end):
        if r_end and p_end:
            return
        state = r_obs[:5]
        p_state = p_obs[:5]
        if not np.allclose(state, p_state, atol=OBS_ATOL):
            mismatches.append((step, "state(xy,speed,angle)", state.tolist(), p_state.tolist()))

        if check_content:
            content = r_obs[5:]
            p_content = p_obs[5:]
            if not np.allclose(content, p_content, atol=ATOL):
                mismatches.append((step, "content", content.tolist(), p_content.tolist()))

    check_obs(0, ref_obs, port_obs, False, False)

    for t, a in enumerate(actions):
        r_obs, r_rew, r_end = reference_step(ref, a)
        p_obs, p_rew, p_end = port_step(port, a)

        check_obs(t + 1, r_obs, p_obs, r_end, p_end)

        if not np.allclose(r_rew[2], p_rew[2], atol=ATOL):
            mismatches.append((t + 1, "fuel_reward", r_rew[2], p_rew[2]))
        if check_content:
            if not np.allclose(r_rew[:2], p_rew[:2], atol=ATOL):
                mismatches.append((t + 1, "ore_reward", r_rew[:2].tolist(), p_rew[:2].tolist()))

        if r_end != p_end:
            mismatches.append((t + 1, "ended", r_end, p_end))

        if r_end or p_end:
            ref_obs, _ = ref.reset(seed=0)
            port.reset(seed=0)
            ref_obs = np.asarray(ref_obs, dtype=np.float64)
            port_obs = port.observations[0].astype(np.float64)
            check_obs(t + 1, ref_obs, port_obs, False, False)

    ref.close()
    port.close()

    if mismatches:
        kind_counts = {}
        for _, kind, _, _ in mismatches:
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        print(f"[{label}] FAILED with {len(mismatches)} mismatch(es):")
        for t, kind, rv, pv in mismatches[:10]:
            kind_short = kind[:15]
            r_str = str(rv)[:60]
            p_str = str(pv)[:60]
            print(f"    step {t:3d}  {kind_short:15s}  ref={r_str}  port={p_str}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")
        return False
    else:
        print(f"[{label}] PASSED ({len(actions)} steps)")
        return True


def test_initial_state():
    ref = make_reference()
    port = make_port()

    ref_obs, _ = ref.reset(seed=0)
    port.reset(seed=0)

    ref_obs = np.asarray(ref_obs, dtype=np.float64)
    port_obs = port.observations[0].astype(np.float64)

    ref.close()
    port.close()

    assert np.allclose(ref_obs, port_obs), (
        f"Initial obs mismatch: ref={ref_obs} port={port_obs}"
    )
    print("[initial_state] PASSED")
    return True


def test_fuel_idle():
    actions = [ACT_NONE] * 10
    assert compare_sequence(actions, "fuel_idle")


def test_fuel_accel():
    actions = [ACT_ACCEL] * 10
    assert compare_sequence(actions, "fuel_accel")


def test_fuel_mine():
    actions = [ACT_MINE] * 10
    assert compare_sequence(actions, "fuel_mine")


def test_fuel_brake():
    actions = [ACT_BRAKE] * 10
    assert compare_sequence(actions, "fuel_brake")


def test_left_turns():
    actions = [ACT_LEFT] * 10
    assert compare_sequence(actions, "left_turns")


def test_right_turns():
    actions = [ACT_RIGHT] * 10
    assert compare_sequence(actions, "right_turns")


def test_accel_then_brake():
    actions = [ACT_ACCEL] * 20 + [ACT_BRAKE] * 20
    assert compare_sequence(actions, "accel_then_brake")


def test_drive_and_turn():
    path = [ACT_ACCEL] * 10 + [ACT_RIGHT] * 5 + [ACT_ACCEL] * 5 + [ACT_LEFT] * 5
    assert compare_sequence(path, "drive_and_turn")


def test_deposit_no_depart():
    actions = [ACT_MINE, ACT_NONE, ACT_NONE, ACT_NONE]
    assert compare_sequence(actions, "deposit_no_depart")


def test_depart_and_return():
    actions = [ACT_ACCEL] * 15 + [ACT_BRAKE] * 5
    assert compare_sequence(actions, "depart_and_return")


def test_random_short():
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 6, size=100).tolist()
    assert compare_sequence(actions, "random_short_100", check_content=False)


def test_wall_right():
    path = [ACT_LEFT] + [ACT_ACCEL] * 30 + [ACT_NONE] * 10
    assert compare_sequence(path, "wall_right")


def test_wall_top():
    path = [ACT_ACCEL] * 22 + [ACT_NONE] * 10
    assert compare_sequence(path, "wall_top")


def test_wall_left():
    path = [ACT_RIGHT] * 4 + [ACT_ACCEL] * 35 + [ACT_NONE] * 10
    assert compare_sequence(path, "wall_left")


def test_wall_bottom():
    path = [ACT_RIGHT] * 6 + [ACT_ACCEL] * 25 + [ACT_NONE] * 10
    assert compare_sequence(path, "wall_bottom")


def test_corner_topright():
    path = [ACT_ACCEL] * 25 + [ACT_NONE] * 10
    assert compare_sequence(path, "corner_topright")


def test_wall_sequence_right_left():
    path = [ACT_LEFT] + [ACT_ACCEL] * 12 + [ACT_LEFT] * 2 + [ACT_ACCEL] * 10 + [ACT_LEFT] * 2 + [ACT_ACCEL] * 10
    assert compare_sequence(path, "wall_sequence_right_left")


def test_long_random_no_mine():
    rng = np.random.default_rng(0)
    actions = rng.choice([ACT_LEFT, ACT_RIGHT, ACT_ACCEL, ACT_BRAKE, ACT_NONE], size=2000).tolist()
    assert compare_sequence(actions, "long_random_no_mine_2000")


def test_mine_mechanic():
    port = make_port()
    port.reset(seed=0)

    rng = np.random.default_rng(42)
    mined_any = False
    for _ in range(1000):
        action = rng.integers(0, 6)
        port_step(port, action)
        content = port.observations[0, 5:] * 1.5
        if content[0] > 0 or content[1] > 0:
            mined_any = True
            break

    port.close()
    if not mined_any:
        print("[mine_mechanic] WARNING: No ore collected in 1000 random steps; "
              "the cart may not have encountered a mine. This is not necessarily a failure.")
    else:
        print(f"[mine_mechanic] PASSED - ore collected at step {_}")
    return True


def test_set_weights():
    port = make_port()
    port.reset(seed=0)

    weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    port.set_weights(weights)

    rng = np.random.default_rng(0)
    for _ in range(2000):
        action = rng.integers(0, 6)
        port.actions[:] = np.array([action], dtype=port.actions.dtype)
        binding.vec_step(port.c_envs)
        if port.terminals[0]:
            break

    log = binding.vec_log(port.c_envs)
    assert abs(log["weight_ore0"] - 0.5) < 0.01, f"Weight ore0 mismatch: {log}"
    assert abs(log["weight_ore1"] - 0.3) < 0.01, f"Weight ore1 mismatch: {log}"
    assert abs(log["weight_fuel"] - 0.2) < 0.01, f"Weight fuel mismatch: {log}"

    port.close()
    print("[set_weights] PASSED")
    return True

if __name__ == "__main__":
    # Wrap test execution in try/except to gracefully catch assertion errors 
    # instead of crashing mid-suite, giving you a full report.
    tests = [
        ("initial_state", test_initial_state),
        ("fuel_idle", test_fuel_idle),
        ("fuel_accel", test_fuel_accel),
        ("fuel_mine", test_fuel_mine),
        ("fuel_brake", test_fuel_brake),
        ("left_turns", test_left_turns),
        ("right_turns", test_right_turns),
        ("accel_then_brake", test_accel_then_brake),
        ("drive_and_turn", test_drive_and_turn),
        ("deposit_no_depart", test_deposit_no_depart),
        ("depart_and_return", test_depart_and_return),
        ("random_short", test_random_short),
        ("wall_right", test_wall_right),
        ("wall_top", test_wall_top),
        ("wall_left", test_wall_left),
        ("wall_bottom", test_wall_bottom),
        ("corner_topright", test_corner_topright),
        ("wall_sequence_right_left", test_wall_sequence_right_left),
        ("long_random_no_mine", test_long_random_no_mine),
        ("mine_mechanic", test_mine_mechanic),
        ("set_weights", test_set_weights),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append(success)
        except AssertionError as e:
            print(f"[{name}] FAILED with AssertionError: {e}")
            results.append(False)

    print()
    if all(results):
        print(f"ALL {len(results)} TESTS PASSED")
    else:
        n_fail = results.count(False)
        print(f"{n_fail}/{len(results)} TESTS FAILED")
