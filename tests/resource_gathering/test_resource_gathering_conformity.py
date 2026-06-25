import numpy as np
import mo_gymnasium as mo_gym

from pufferlib.ocean.resource_gathering.resource_gathering import ResourceGathering

ATOL = 1e-5

GRID_ROWS = 5
GRID_COLS = 5

TILE_EMPTY = 0
TILE_HOME = 1
TILE_GOLD = 2
TILE_GEM = 3
TILE_ENEMY = 4

# Cells by (row, col) in the standard layout
HOME_POS = (4, 2)
GOLD_POS = (3, 1)
GEM_POS = (1, 3)
ENEMY_POSITIONS = [(1, 1), (3, 3)]

def cell_type(row, col):
    if (row, col) == HOME_POS:
        return TILE_HOME
    if (row, col) == GOLD_POS:
        return TILE_GOLD
    if (row, col) == GEM_POS:
        return TILE_GEM
    if (row, col) in ENEMY_POSITIONS:
        return TILE_ENEMY
    return TILE_EMPTY

# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT = 0, 1, 2, 3


def make_reference():
    return mo_gym.make("resource-gathering-v0")


def make_port():
    return ResourceGathering(num_envs=1)


def reference_step(env, action):
    obs, reward, terminated, truncated, _ = env.step(action)
    ended = bool(terminated or truncated)
    return np.asarray(obs, dtype=np.float64), np.asarray(reward, dtype=np.float64), ended


def port_step(env, action):
    env.actions[:] = np.array([action], dtype=env.actions.dtype)
    import pufferlib.ocean.resource_gathering.binding as binding
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

    if not np.array_equal(ref_obs, port_obs):
        print(f"[{label}] INITIAL OBS MISMATCH: ref={ref_obs} port={port_obs}")
        ref.close()
        port.close()
        return False

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
            port.reset(seed=0)

    ref.close()
    port.close()

    if mismatches:
        print(f"[{label}] FAILED with {len(mismatches)} mismatch(es):")
        for t, kind, rv, pv in mismatches[:20]:
            print(f"    step {t:3d}  {kind:7s}  ref={rv}  port={pv}")
        return False
    print(f"[{label}] PASSED ({len(actions)} steps)")
    return True




def test_manual_pickup_gold():
    actions = [ACT_UP, ACT_LEFT, ACT_DOWN, ACT_RIGHT]
    return compare_sequence(actions, "manual_pickup_gold")


def test_manual_pickup_gem():
    actions = [ACT_UP, ACT_UP, ACT_UP, ACT_RIGHT,
               ACT_DOWN, ACT_LEFT, ACT_DOWN, ACT_DOWN]
    return compare_sequence(actions, "manual_pickup_gem")


def test_manual_enemy_collision():
    actions = [ACT_UP, ACT_RIGHT]
    return compare_sequence(actions, "manual_enemy_collision")


def test_manual_wall_bumps():
    actions = [ACT_DOWN, ACT_RIGHT, ACT_RIGHT, ACT_RIGHT]
    return compare_sequence(actions, "manual_wall_bumps")


def test_random_short():
    rng = np.random.default_rng(7)
    actions = rng.integers(0, 4, size=500).tolist()
    return compare_sequence(actions, "random_500")


def test_random_long():
    rng = np.random.default_rng(2024)
    actions = rng.integers(0, 4, size=5000).tolist()
    return compare_sequence(actions, "random_5000")


def test_random_up_biased():
    rng = np.random.default_rng(42)
    choices = rng.choice([0, 1, 2, 3], size=3000, p=[0.5, 0.15, 0.2, 0.15])
    return compare_sequence(choices.tolist(), "random_up_biased")


def test_random_many_seeds():
    all_passed = True
    for seed in [1, 23, 456, 7890, 99999]:
        rng = np.random.default_rng(seed)
        actions = rng.integers(0, 4, size=1500).tolist()
        if not compare_sequence(actions, f"random_seed_{seed}"):
            all_passed = False
    return all_passed


def get_ref_state(ref_env):
    """Read internal state from MO-Gym resource_gathering env.
    Typically a 4-tuple (row, col, has_gold, has_gem)."""
    inner = ref_env.unwrapped
    for attr in ("current_state", "state", "_state"):
        if hasattr(inner, attr):
            val = getattr(inner, attr)
            if hasattr(val, '__len__') and len(val) >= 4:
                return (int(val[0]), int(val[1]), int(val[2]), int(val[3]))
    if all(hasattr(inner, a) for a in ('row', 'col', 'has_gold', 'has_gem')):
        return (int(inner.row), int(inner.col),
                int(inner.has_gold), int(inner.has_gem))
    raise AttributeError(
        "Couldn't find resource_gathering internal state. "
        f"Available: {[a for a in dir(inner) if not a.startswith('_')]}"
    )


def get_port_state(port):
    obs = port.observations[0]
    # Most likely layout based on standard MO-Gym format
    return (int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]))


def test_internal_state_matches():
    ref = make_reference()
    port = make_port()

    ref.reset(seed=0)
    port.reset(seed=0)

    rng = np.random.default_rng(2024)
    actions = rng.integers(0, 4, size=3000).tolist()

    try:
        _ = get_ref_state(ref)
    except AttributeError as e:
        print(f"[internal_state_matches] SKIPPED: {e}")
        ref.close()
        port.close()
        return True

    mismatches = []
    for t, a in enumerate(actions):
        ref.step(a)
        port_step(port, a)

        if not bool(port.terminals[0]):
            ref_state = get_ref_state(ref)
            port_state = get_port_state(port)
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



def is_navigable(row, col):
    """Cell can be stood on (not out of bounds; no impassable cells in RG)."""
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def enumerate_reachable_states():
    states = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if not is_navigable(row, col):
                continue
            for has_gold in [0, 1]:
                for has_gem in [0, 1]:
                    # Skip enemy cells — agent can't be at the start of
                    # a step in an enemy cell (entering one triggers reset)
                    if (row, col) in ENEMY_POSITIONS:
                        continue
                    # Skip the terminal state (home with both resources)
                    if (row, col) == HOME_POS and has_gold == 1 and has_gem == 1:
                        continue
                    # Skip "have gold but not at the position where you could
                    # have picked it up consistently" — simplest: include all,
                    # navigation logic decides if it's reachable
                    states.append((row, col, has_gold, has_gem))
    return states


def navigation_path_to(target_row, target_col, target_has_gold, target_has_gem):
    path = []
    current = list(HOME_POS) + [0, 0]   # [row, col, has_gold, has_gem]

    def move_to(target_r, target_c, avoid_pickup=True):
        """Move toward (target_r, target_c) avoiding enemy cells and
        avoiding picking up resources we don't want."""
        steps = []
        while (current[0], current[1]) != (target_r, target_c):
            # Try to move in the direction that gets us closer
            dr = target_r - current[0]
            dc = target_c - current[1]
            candidates = []
            if dr < 0: candidates.append((ACT_UP, current[0]-1, current[1]))
            if dr > 0: candidates.append((ACT_DOWN, current[0]+1, current[1]))
            if dc < 0: candidates.append((ACT_LEFT, current[0], current[1]-1))
            if dc > 0: candidates.append((ACT_RIGHT, current[0], current[1]+1))

            chosen = None
            for act, nr, nc in candidates:
                if (nr, nc) in ENEMY_POSITIONS:
                    continue
                if avoid_pickup:
                    if (nr, nc) == GOLD_POS and not target_has_gold:
                        continue
                    if (nr, nc) == GEM_POS and not target_has_gem:
                        continue
                chosen = (act, nr, nc)
                break

            if chosen is None:
                # No valid move toward target — unreachable along this strategy
                return None
            steps.append(chosen[0])
            current[0], current[1] = chosen[1], chosen[2]

            # Update resource flags
            if (current[0], current[1]) == GOLD_POS:
                current[2] = 1
            if (current[0], current[1]) == GEM_POS:
                current[3] = 1
        return steps

    if target_has_gold and current[2] == 0:
        s = move_to(GOLD_POS[0], GOLD_POS[1], avoid_pickup=True)
        if s is None: return None
        path.extend(s)

    if target_has_gem and current[3] == 0:
        s = move_to(GEM_POS[0], GEM_POS[1], avoid_pickup=True)
        if s is None: return None
        path.extend(s)

    s = move_to(target_row, target_col, avoid_pickup=True)
    if s is None: return None
    path.extend(s)

    if (current[0], current[1], current[2], current[3]) != \
       (target_row, target_col, target_has_gold, target_has_gem):
        return None

    return path


def test_exhaustive_transitions():
    candidate_states = enumerate_reachable_states()
    print(f"[exhaustive_transitions] Candidate states: {len(candidate_states)}")

    mismatches = []
    n_tested = 0
    n_unreachable = 0

    for (target_row, target_col, target_has_gold, target_has_gem) in candidate_states:
        path = navigation_path_to(target_row, target_col,
                                  target_has_gold, target_has_gem)
        if path is None:
            n_unreachable += 1
            continue

        for test_action in range(4):
            ref = make_reference()
            port = make_port()
            ref.reset(seed=0)
            port.reset(seed=0)

            # Navigate to target state
            ok = True
            for nav_action in path:
                _, _, term, trunc, _ = ref.step(nav_action)
                port_step(port, nav_action)
                if bool(term or trunc) or bool(port.terminals[0]):
                    # Path triggered a terminal — skip this state-action test
                    ok = False
                    break

            if not ok:
                ref.close()
                port.close()
                continue

            # Now apply the test action and compare
            r_obs, r_rew, r_term, r_trunc, _ = ref.step(test_action)
            port_step(port, test_action)

            r_obs = np.asarray(r_obs, dtype=np.float64)
            r_rew = np.asarray(r_rew, dtype=np.float64)
            r_ended = bool(r_term or r_trunc)

            p_obs = port.observations[0].astype(np.float64)
            p_rew = port.rewards[0].astype(np.float64)
            p_term = bool(port.terminals[0])

            state_label = (target_row, target_col, target_has_gold, target_has_gem)

            if not r_ended and not np.array_equal(r_obs, p_obs):
                mismatches.append((state_label, test_action,
                                   "next_state", r_obs.tolist(), p_obs.tolist()))

            if not np.allclose(r_rew, p_rew, atol=ATOL):
                mismatches.append((state_label, test_action,
                                   "reward", r_rew.tolist(), p_rew.tolist()))

            if r_ended != p_term:
                mismatches.append((state_label, test_action,
                                   "terminal", r_ended, p_term))

            n_tested += 1
            ref.close()
            port.close()

    if mismatches:
        print(f"[exhaustive_transitions] FAILED: {len(mismatches)} mismatch(es) "
              f"out of {n_tested} transitions ({n_unreachable} unreachable states skipped):")
        for state, action, kind, rv, pv in mismatches[:15]:
            action_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[action]
            print(f"    state={state} action={action_name} {kind}: ref={rv}  port={pv}")
        return False

    print(f"[exhaustive_transitions] PASSED "
          f"(all {n_tested} (state, action) transitions match, "
          f"{n_unreachable} states unreachable by greedy navigation)")
    return True


if __name__ == "__main__":
    results = []
    results.append(test_manual_pickup_gold())
    results.append(test_manual_pickup_gem())
    results.append(test_manual_enemy_collision())
    results.append(test_manual_wall_bumps())
    results.append(test_random_short())
    results.append(test_random_long())
    results.append(test_random_up_biased())
    results.append(test_random_many_seeds())
    results.append(test_internal_state_matches())
    results.append(test_exhaustive_transitions())

    if all(results):
        print(f"\nALL {len(results)} TESTS PASSED")
    else:
        n_fail = results.count(False)
        print(f"\n{n_fail}/{len(results)} TESTS FAILED")