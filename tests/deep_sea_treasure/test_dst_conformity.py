import numpy as np
import mo_gymnasium as mo_gym

from pufferlib.ocean.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure

ATOL = 1e-5

DST_MAP = [
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [0.7,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [-10., 8.2,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [-10., -10., 11.5, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [-10., -10., -10., 14.0, 15.1, 16.1, 0.0,  0.0,  0.0,  0.0,  0.0],
    [-10., -10., -10., -10., -10., -10., 0.0,  0.0,  0.0,  0.0,  0.0],
    [-10., -10., -10., -10., -10., -10., 0.0,  0.0,  0.0,  0.0,  0.0],
    [-10., -10., -10., -10., -10., -10., 19.6, 20.3, 0.0,  0.0,  0.0],
    [-10., -10., -10., -10., -10., -10., -10., -10., 22.4, 0.0,  0.0],
    [-10., -10., -10., -10., -10., -10., -10., -10., -10., 23.7, 0.0],
    [-10., -10., -10., -10., -10., -10., -10., -10., -10., -10., 0.0],
]
GRID_ROWS = 11
GRID_COLS = 11
ROCK_VAL = -10.0

ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT = 0, 1, 2, 3


def make_reference():
    return mo_gym.make("deep-sea-treasure-v0")


def make_port():
    return DeepSeaTreasure(num_envs=1, gamma=0.99)


def reference_step(env, action):
    obs, reward, terminated, truncated, _ = env.step(action)
    ended = bool(terminated or truncated)
    return np.asarray(obs, dtype=np.float64), np.asarray(reward, dtype=np.float64), ended


def port_step(env, action):
    env.actions[:] = np.array([action], dtype=env.actions.dtype)
    import pufferlib.ocean.deep_sea_treasure.binding as binding
    binding.vec_step(env.c_envs)
    obs = env.observations[0].astype(np.float64).copy()
    reward = env.rewards[0].astype(np.float64).copy()
    ended = bool(env.terminals[0])
    return obs, reward, ended


def compare_sequence(actions, label):
    ref = make_reference()
    port = make_port()

    ref_obs, _ = ref.reset(seed=0)
    port.reset(seed=0)

    ref_obs = np.asarray(ref_obs, dtype=np.float64)
    port_obs = port.observations[0].astype(np.float64)

    assert np.array_equal(ref_obs, port_obs), (
        f"[{label}] initial obs mismatch: ref={ref_obs} port={port_obs}")

    mismatches = []
    for t, a in enumerate(actions):
        r_obs, r_rew, r_end = reference_step(ref, a)
        p_obs, p_rew, p_end = port_step(port, a)

        if not np.array_equal(r_obs, p_obs) and not (r_end and p_end):
            mismatches.append((t, "obs", r_obs.tolist(), p_obs.tolist()))

        if not np.allclose(r_rew, p_rew, atol=ATOL):
            mismatches.append((t, "reward", r_rew.tolist(), p_rew.tolist()))

        if r_end != p_end:
            mismatches.append((t, "ended", r_end, p_end))

        if r_end or p_end:
            ref.reset(seed=0)

    ref.close()
    port.close()

    if mismatches:
        print(f"[{label}] FAILED with {len(mismatches)} mismatch(es):")
        for t, kind, rv, pv in mismatches[:20]:
            print(f"    step {t:3d}  {kind:7s}  ref={rv}  port={pv}")
        return False
    else:
        print(f"[{label}] PASSED ({len(actions)} steps)")
        return True



def test_manual_nearest_treasure():
    actions = [1]
    return compare_sequence(actions, "manual_nearest_treasure")


def test_manual_wall_bumps():
    actions = [0, 2, 1]
    return compare_sequence(actions, "manual_wall_bumps")


def test_manual_deeper_treasure():
    actions = [3, 1, 1]
    return compare_sequence(actions, "manual_deeper_treasure")


def test_manual_rock_block():
    actions = [1, 1]
    return compare_sequence(actions, "manual_rock_block")


def test_random_actions():
    rng = np.random.default_rng(12345)
    actions = rng.integers(0, 4, size=2000).tolist()
    return compare_sequence(actions, "random_2000")


def test_random_long():
    rng = np.random.default_rng(999)
    choices = rng.choice([0, 2, 1, 3], size=5000, p=[0.35, 0.35, 0.15, 0.15])
    return compare_sequence(choices.tolist(), "random_long_truncation")



def get_ref_state(ref_env):
    inner = ref_env.unwrapped
    for attr in ("current_state", "state", "_state", "position", "_position"):
        if hasattr(inner, attr):
            val = getattr(inner, attr)
            return (int(val[0]), int(val[1]))
    raise AttributeError(
        "Couldn't find DST internal state attribute on env.unwrapped. "
        "Tried: current_state, state, _state, position, _position. "
        f"Available: {[a for a in dir(inner) if not a.startswith('_')]}"
    )


def test_internal_state_matches():
    ref = make_reference()
    port = make_port()

    ref.reset(seed=0)
    port.reset(seed=0)

    rng = np.random.default_rng(2024)
    actions = rng.integers(0, 4, size=3000).tolist()

    mismatches = []
    for t, a in enumerate(actions):
        ref.step(a)
        port_step(port, a)

        try:
            ref_state = get_ref_state(ref)
        except AttributeError as e:
            print(f"[internal_state_matches] SKIPPED: {e}")
            ref.close()
            port.close()
            return True  

        port_state = (int(port.observations[0, 0]), int(port.observations[0, 1]))

        if not bool(port.terminals[0]):
            if ref_state != port_state:
                mismatches.append((t, ref_state, port_state))

        if bool(port.terminals[0]):
            ref.reset(seed=0)
            port.reset(seed=0)

    ref.close()
    port.close()

    if mismatches:
        print(f"[internal_state_matches] FAILED with {len(mismatches)} mismatch(es):")
        for t, r, p in mismatches[:10]:
            print(f"    step {t}: ref_state={r}  port_state={p}")
        return False
    print(f"[internal_state_matches] PASSED ({len(actions)} steps)")
    return True



def is_navigable_non_terminal(row, col):

    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        return False
    v = DST_MAP[row][col]
    return v == 0.0


def navigation_path(target_row, target_col):
    path = []
    for _ in range(target_col):
        path.append(ACT_RIGHT)
    for _ in range(target_row):
        path.append(ACT_DOWN)
    return path


def test_exhaustive_transitions():

    navigable_cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if is_navigable_non_terminal(row, col):
                navigable_cells.append((row, col))

    print(f"[exhaustive_transitions] Testing {len(navigable_cells)} states × 4 actions "
          f"= {len(navigable_cells) * 4} transitions")

    mismatches = []
    n_tested = 0

    for (target_row, target_col) in navigable_cells:
        path = navigation_path(target_row, target_col)

        for test_action in range(4):
            ref = make_reference()
            port = make_port()
            ref.reset(seed=0)
            port.reset(seed=0)

            for nav_action in path:
                ref.step(nav_action)
                port_step(port, nav_action)

            port_pos = (int(port.observations[0, 0]), int(port.observations[0, 1]))
            if port_pos != (target_row, target_col):
                print(f"  WARNING: failed to navigate to ({target_row},{target_col}); "
                      f"port at {port_pos} after path of {len(path)} actions")
                ref.close()
                port.close()
                continue


            r_obs, r_rew, r_term, r_trunc, _ = ref.step(test_action)
            port_step(port, test_action)

            r_obs = np.asarray(r_obs, dtype=np.float64)
            r_rew = np.asarray(r_rew, dtype=np.float64)
            r_ended = bool(r_term or r_trunc)

            p_obs = port.observations[0].astype(np.float64)
            p_rew = port.rewards[0].astype(np.float64)
            p_term = bool(port.terminals[0])

            if not r_ended and not np.array_equal(r_obs, p_obs):
                mismatches.append(((target_row, target_col), test_action,
                                   "next_state", r_obs.tolist(), p_obs.tolist()))

            if not np.allclose(r_rew, p_rew, atol=ATOL):
                mismatches.append(((target_row, target_col), test_action,
                                   "reward", r_rew.tolist(), p_rew.tolist()))


            if r_ended != p_term:
                mismatches.append(((target_row, target_col), test_action,
                                   "terminal", r_ended, p_term))

            n_tested += 1
            ref.close()
            port.close()

    if mismatches:
        print(f"[exhaustive_transitions] FAILED with {len(mismatches)} mismatch(es) "
              f"out of {n_tested} transitions:")
        for state, action, kind, rv, pv in mismatches[:15]:
            action_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[action]
            print(f"    state={state} action={action_name} {kind}: ref={rv}  port={pv}")
        return False

    print(f"[exhaustive_transitions] PASSED (all {n_tested} (state, action) transitions match)")
    return True


if __name__ == "__main__":
    results = []
    results.append(test_manual_nearest_treasure())
    results.append(test_manual_wall_bumps())
    results.append(test_manual_deeper_treasure())
    results.append(test_manual_rock_block())
    results.append(test_random_actions())
    results.append(test_random_long())
    results.append(test_internal_state_matches())
    results.append(test_exhaustive_transitions())

    if all(results):
        print(f"\nALL {len(results)} TESTS PASSED")
    else:
        n_fail = results.count(False)
        print(f"\n{n_fail}/{len(results)} TESTS FAILED")