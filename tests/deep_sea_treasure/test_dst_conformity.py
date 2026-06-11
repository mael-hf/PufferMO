import numpy as np
import mo_gymnasium as mo_gym

from pufferlib.ocean.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure

ATOL = 1e-5   #


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
    # vec_step writes obs/rewards/terminals into the shared buffers.
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



# Test 1: sequence reaching the nearest treasure
def test_manual_nearest_treasure():
    # From (0,0): one DOWN reaches (1,0)
    actions = [1]  
    return compare_sequence(actions, "manual_nearest_treasure")



# Test 2: bump into the boundary, then reach a treasure

def test_manual_wall_bumps():
    # From (0,0): try UP , LEFT to hit the boundary (stay at (0,0)),
    # then DOWN to (1,0)=0.7 terminal.
    actions = [0, 2, 1]  # up, left, down
    return compare_sequence(actions, "manual_wall_bumps")



# Test 3: navigate to a deeper treasure 

def test_manual_deeper_treasure():
    # From (0,0): RIGHT to (0,1), DOWN to (1,1), DOWN to (2,1)=8.2 terminal.
    actions = [3, 1, 1]  # right, down, down
    return compare_sequence(actions, "manual_deeper_treasure")



# Test 4: exercise the rock-cell blocking

def test_manual_rock_block():
    actions = [1, 1]
    return compare_sequence(actions, "manual_rock_block")



# Test 5: random actions 

def test_random_actions():
    rng = np.random.default_rng(12345)
    actions = rng.integers(0, 4, size=2000).tolist()
    return compare_sequence(actions, "random_2000")



# Test 6: random with a different seed, longer run to hit the 100-step truncation

def test_random_long():
    rng = np.random.default_rng(999)
    choices = rng.choice([0, 2, 1, 3], size=5000, p=[0.35, 0.35, 0.15, 0.15])
    return compare_sequence(choices.tolist(), "random_long_truncation")




if __name__ == "__main__":
    results = []
    results.append(test_manual_nearest_treasure())
    results.append(test_manual_wall_bumps())
    results.append(test_manual_deeper_treasure())
    results.append(test_manual_rock_block())
    results.append(test_random_actions())
    results.append(test_random_long())

    if all(results):
        print(f"ALL {len(results)} TESTS PASSED")
    else:
        n_fail = results.count(False)
        print(f"{n_fail}/{len(results)} TESTS FAILED")