"""
test_conformity_LL.py
======================
Conformity tests for the PufferLib LunarLander port, **synchronisés sur
la même tape RNG** que la référence Gymnasium (via record_reference_tape /
set_rng_tape) -- voir rng_tape_recorder.py.

Pourquoi cette version et pas l'ancienne ?
-------------------------------------------
Sans synchronisation RNG, le port C (rand()/PCG côté C) et la référence
Gymnasium (numpy Generator) tirent des nombres aléatoires totalement
indépendants. Dès le premier tirage (terrain, force initiale), les deux
trajectoires divergent -- et on voit des mismatches sur obs/rew_landing
qui ne disent RIEN sur la justesse de la physique portée, seulement sur
le fait que le hasard n'est pas le même des deux côtés.

Le principe ici :
    1. On enregistre la tape RNG RÉELLEMENT consommée par la référence
       sur une séquence d'actions donnée (record_reference_tape).
    2. On injecte cette même tape dans le port via set_rng_tape().
    3. On compare port_step(...) aux obs/rewards déjà capturés à
       l'étape 1 (on ne re-step PAS la référence une 2e fois).

À RNG identique, tout écart restant (au-delà de ATOL, qui n'absorbe que
le bruit float32 vs float64) provient bien d'une différence de physique
réelle entre le port et la référence -- et c'est ce qu'on veut détecter.

Limitation connue : à un pas terminal, le port effectue son auto-reset
*avant* de retourner les observations à Python (cf. c_step -> c_reset en
fin de fonction), donc l'obs renvoyée par le port à ce pas est déjà celle
du nouvel épisode, alors que la référence renvoie encore l'obs terminale.
On exclut donc la comparaison d'obs (pas celle du reward) sur les pas où
les deux environnements terminent en même temps.
"""

import numpy as np
import gymnasium as gym

from pufferlib.ocean.lunar_lander.Lunar_Lander import LunarLander, record_reference_tape
import pufferlib.ocean.lunar_lander.binding as binding

ATOL = 1e-3


# ─────────────────────────────────────────────────────────────────────
# Construction des environnements
# ─────────────────────────────────────────────────────────────────────

def make_reference(enable_wind: bool = False):
    return gym.make("LunarLander-v3", enable_wind=enable_wind)


def make_port(enable_wind: bool = False):
    port = LunarLander(num_envs=1, enable_wind=enable_wind)
    # Poids fixes : seul REW_LANDING nous intéresse pour la comparaison
    # avec la référence (qui est mono-objectif). manual_weights=1 évite
    # aussi le tirage Dirichlet, qui ne doit PAS consommer de la tape.
    port.set_weights(np.array([1.0, 0.0], dtype=np.float32))
    return port


def port_step(port, action: int):
    port.actions[:] = np.array([action], dtype=port.actions.dtype)
    binding.vec_step(port.c_envs)
    obs    = port.observations[0].astype(np.float64).copy()
    rewvec = port.rewards[0].astype(np.float64).copy()
    ended  = bool(port.terminals[0])
    return obs, rewvec, ended


# ─────────────────────────────────────────────────────────────────────
# Comparaison pas-à-pas, RNG synchronisée
# ─────────────────────────────────────────────────────────────────────

def compare_sequence(actions: list, label: str, seed: int = 0,
                      enable_wind: bool = False, verbose: bool = True) -> bool:
    """
    1. Enregistre la tape RNG réelle de la référence sur `actions`.
    2. Injecte cette tape dans le port et rejoue la même séquence.
    3. Compare, étape par étape, obs / rew_landing / ended.

    Retourne True si tout est dans ATOL, False sinon. Affiche dans tous
    les cas un tableau de l'écart par étape (utile même quand ça passe,
    pour voir l'ordre de grandeur du bruit float32 résiduel).
    """
    ref = make_reference(enable_wind=enable_wind)
    tape, ref_obs, ref_rews, ref_dones = record_reference_tape(ref, seed, actions)
    ref.close()

    port = make_port(enable_wind=enable_wind)
    port.set_rng_tape(tape)
    port.reset(seed=seed)

    port_obs0 = port.observations[0].astype(np.float64)
    rows = []          # (step, max_obs_diff or None, rew_landing_diff, ok)
    mismatches = []

    # ── observation initiale (après le step(0) interne de reset()) ──
    init_diff = float(np.max(np.abs(ref_obs[0] - port_obs0)))
    if init_diff > ATOL:
        mismatches.append((-1, "init_obs", ref_obs[0].tolist(), port_obs0.tolist()))
    rows.append((-1, init_diff, None, init_diff <= ATOL))

    for t, a in enumerate(actions):
        p_obs, p_rewvec, p_end = port_step(port, a)
        r_obs, r_rew, r_end = ref_obs[t + 1], ref_rews[t], ref_dones[t]
        p_rew_landing = p_rewvec[0]

        both_ended = r_end and p_end
        obs_diff = None if both_ended else float(np.max(np.abs(r_obs - p_obs)))
        rew_diff = float(abs(r_rew - p_rew_landing))

        ok = (rew_diff <= ATOL) and (obs_diff is None or obs_diff <= ATOL) \
             and (r_end == p_end)
        rows.append((t, obs_diff, rew_diff, ok))

        if not ok:
            if obs_diff is not None and obs_diff > ATOL:
                mismatches.append((t, "obs", r_obs.tolist(), p_obs.tolist()))
            if rew_diff > ATOL:
                mismatches.append((t, "rew_landing", round(r_rew, 6), round(p_rew_landing, 6)))
            if r_end != p_end:
                mismatches.append((t, "ended", r_end, p_end))

    port.close()

    if verbose:
        _print_step_table(label, rows)

    if mismatches:
        print(f"[{label}] FAILED -- {len(mismatches)} mismatch(es) (ATOL={ATOL}):")
        for t, kind, rv, pv in mismatches[:20]:
            print(f"    step {t:4d}  {kind:12s}  ref={rv}  port={pv}")
        return False

    print(f"[{label}] PASSED ({len(actions)} steps, RNG-synchronised)")
    return True


def _print_step_table(label: str, rows, max_rows: int = 30):
    """Affiche l'écart par étape : utile pour repérer À QUEL PAS la
    divergence apparaît (souvent plus parlant qu'une liste de mismatches
    en vrac)."""
    print(f"\n--- Écart pas-à-pas [{label}] ---")
    print(f"{'step':>6} {'max|Δobs|':>12} {'Δrew_landing':>14} {'status':>8}")
    n = len(rows)
    shown = rows if n <= max_rows else rows[:max_rows // 2] + rows[-max_rows // 2:]
    truncated = n > max_rows
    for i, (t, obs_diff, rew_diff, ok) in enumerate(shown):
        if truncated and i == max_rows // 2:
            print("   ...   (lignes intermédiaires omises)")
        obs_str = f"{obs_diff:.3e}" if obs_diff is not None else "  (skip)"
        rew_str = f"{rew_diff:.3e}" if rew_diff is not None else "    --"
        status = "OK" if ok else "MISMATCH"
        print(f"{t:6d} {obs_str:>12} {rew_str:>14} {status:>8}")
    print()


# ─────────────────────────────────────────────────────────────────────
# Scénarios (mêmes intentions que l'ancienne version, RNG en plus)
# ─────────────────────────────────────────────────────────────────────

def test_do_nothing():
    return compare_sequence([0] * 200, "do_nothing_200")


def test_main_engine_only():
    return compare_sequence([2] * 150, "main_engine_only_150")


def test_side_thrusters():
    return compare_sequence([1, 3] * 75, "alternate_side_thrusters_150")


def test_landing_attempt():
    actions = [2] * 40 + [0] * 60 + [2] * 20 + [0] * 80
    return compare_sequence(actions, "landing_attempt")


def test_out_of_bounds():
    return compare_sequence([1] * 500, "out_of_bounds_left")


def test_wall_boundary_right():
    return compare_sequence([3] * 500, "out_of_bounds_right")


def test_random_actions_short():
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 4, size=1000).tolist()
    return compare_sequence(actions, "random_1000_seed42", verbose=False)


def test_random_actions_long():
    rng = np.random.default_rng(2024)
    actions = rng.choice([0, 1, 2, 3], size=3000,
                          p=[0.20, 0.20, 0.40, 0.20]).tolist()
    return compare_sequence(actions, "random_3000_bias_main_engine", verbose=False)


def test_wind_mode_sync():
    """Avec vent activé : la tape couvre aussi wind_idx/torque_idx
    (integers(-9999,9999)) -- voir record_reference_tape / set_rng_tape."""
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 4, size=300).tolist()
    return compare_sequence(actions, "wind_mode_300_seed0", enable_wind=True)


def test_fuel_reward_no_action():
    """Pas besoin de RNG synchronisée ici : on vérifie juste un invariant
    structurel (action=0 => coût fuel = 0), indépendant du hasard."""
    port = make_port()
    port.reset(seed=7)
    failures = []
    for t in range(100):
        _, rewvec, ended = port_step(port, 0)
        if abs(rewvec[1]) > 1e-6:
            failures.append((t, rewvec[1]))
        if ended:
            port.reset(seed=7)
    port.close()
    label = "fuel_reward_zero_on_no_action"
    if failures:
        print(f"[{label}] FAILED -- {len(failures)} step(s) avec fuel reward non nul")
        return False
    print(f"[{label}] PASSED (100 steps)")
    return True


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
        test_wind_mode_sync,
        test_fuel_reward_no_action,
    ]

    results = [t() for t in tests]

    print()
    if all(results):
        print(f"ALL {len(results)} TESTS PASSED (RNG-synchronised)")
    else:
        n_fail = results.count(False)
        print(f"{n_fail}/{len(results)} TESTS FAILED")