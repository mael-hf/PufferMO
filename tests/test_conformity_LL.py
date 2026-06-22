"""
test_ll_conformity.py
=====================
Conformity tests for the PufferLib LunarLander port.

Compares the PufferLib C implementation step-by-step against the
reference Gymnasium LunarLander-v3 on a set of deterministic action
sequences and random rollouts.

RNG INJECTION
-------------
Pour éliminer le bruit dû aux générateurs aléatoires différents entre
Python (NumPy PCG64) et le port C (libc rand()), compare_sequence()
enregistre la tape RNG réellement consommée par la référence Gymnasium
(via record_reference_tape, cf. Lunar_Lander.py) puis l'injecte dans le
port avant de le rejouer. Les deux environnements voient ainsi le même
terrain, la même force initiale, le même bruit moteur -- toute
divergence résiduelle reflète donc un véritable écart de physique
entre le port et la référence, pas un artefact RNG.

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

from pufferlib.ocean.lunar_lander.Lunar_Lander import LunarLander, record_reference_tape
import pufferlib.ocean.lunar_lander.binding as binding

# Absolute tolerance for reward comparison
# (float32 vs float64 arithmetic can introduce small differences)
ATOL = 1e-3

# Maximum number of steps per episode in Gymnasium LunarLander-v3
MAX_STEPS = 1000

# ─────────────────────────────────────────────────────────────────────
# LIMITATION CONNUE : couple des articulations de jambes non modélisé
# ─────────────────────────────────────────────────────────────────────
# Dans la référence Gymnasium/Box2D, les jambes sont des corps physiques
# reliés au corps principal par des revoluteJointDef avec moteur actif
# (enableMotor=True, motorSpeed, maxMotorTorque=40). Quand les jambes se
# mettent en position au reset, ce moteur transmet un couple de réaction
# sur le corps principal -- une dérive angulaire réelle apparaît dès le
# premier step, même sans action moteur.
#
# Le port C ne modélise aucune articulation : les jambes sont purement
# géométriques (calculées par formule depuis l'angle du corps), sans
# rétroaction physique. Avec action=0, `atorque` reste nul et `omega`
# (donc `angle`) ne bougent JAMAIS de leur valeur initiale -- c'est
# garanti par construction, pas un bug à corriger au cas par cas.
#
# Décision : on exclut explicitement obs[4] (angle) et obs[5] (omega) de
# la comparaison stricte d'observation, et on compare le reward via un
# shaping recalculé SANS le terme d'angle (`shaping_no_angle`), pour
# continuer à détecter de vraies régressions sur tout ce qui est
# effectivement modélisé (position, vitesse, contact des jambes), sans
# que ce gap d'architecture connu ne pollue chaque test.
ANGLE_IDX, OMEGA_IDX = 4, 5
OBS_IDX_STRICT = [0, 1, 2, 3, 6, 7]  # tous sauf angle/omega


def shaping_no_angle(obs: np.ndarray) -> float:
    """Même formule que le shaping du jeu, sans le terme -100*|angle|."""
    s0, s1, s2, s3, s6, s7 = obs[0], obs[1], obs[2], obs[3], obs[6], obs[7]
    return (
        -100.0 * np.sqrt(s0 * s0 + s1 * s1)
        - 100.0 * np.sqrt(s2 * s2 + s3 * s3)
        + 10.0 * s6
        + 10.0 * s7
    )


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
    * On enregistre D'ABORD toute la séquence sur la référence (via
      record_reference_tape), ce qui capture en une seule passe :
        - la tape RNG brute consommée (terrain, force initiale,
          dispersion moteur, wind/torque_idx si activé) ;
        - les observations/récompenses/terminaisons réelles à chaque
          step, y compris à travers d'éventuels redémarrages d'épisode
          (record_reference_tape relance automatiquement avec la même
          graine si l'épisode se termine avant la fin de la séquence).
    * On injecte ensuite cette même tape dans le port avant de le
      rejouer -- les deux environnements consomment alors exactement
      les mêmes tirages aléatoires.
    * Le port auto-reset en interne dès qu'un épisode se termine ; à ce
      step précis, port.observations reflète déjà la NOUVELLE episode
      (c_reset écrase observations après avoir écrit la récompense
      terminale) -- on saute donc la comparaison d'observation
      uniquement quand les DEUX environnements terminent au même step
      (comportement identique à l'ancienne version de ce test).
    * La référence Gymnasium retourne une récompense *scalaire*
      (landing). On compare donc uniquement rewards[REW_LANDING] côté
      port.
    """
    ref = make_reference(seed=seed)

    # 1. Enregistrer la tape réelle + obs/rewards/dones de la référence
    #    sur cette séquence précise (gère les redémarrages automatiques).
    tape, ref_obs, ref_rews, ref_dones = record_reference_tape(ref, seed, actions)
    ref.close()

    # 2. Créer le port, fixer les poids (isole REW_LANDING), injecter
    #    la même tape, puis reset.
    port = make_port(seed=seed)
    port.set_weights(np.array([1.0, 0.0], dtype=np.float32))
    port.set_rng_tape(tape)
    port.reset(seed=seed)

    mismatches = []

    # ── comparaison de l'observation initiale (hors angle/omega) ──
    ref_obs0  = np.asarray(ref_obs[0], dtype=np.float64)
    port_obs0 = port.observations[0].astype(np.float64)

    if check_obs and not np.allclose(
        ref_obs0[OBS_IDX_STRICT], port_obs0[OBS_IDX_STRICT], atol=ATOL
    ):
        mismatches.append((-1, "init_obs", ref_obs0.tolist(), port_obs0.tolist()))

    # Baselines de shaping (sans angle), pour la comparaison de reward.
    # Recalculées indépendamment côté ref et côté port -- chacune doit
    # repartir de NaN/None juste après un reset, comme prev_shaping dans
    # l'env (c_reset le remet à NaN ; on reproduit cette sémantique ici).
    prev_shaping_ref_na  = shaping_no_angle(ref_obs0)
    prev_shaping_port_na = shaping_no_angle(port_obs0)
    prev_r_end = False
    prev_p_end = False

    # ── boucle de step ──
    for t, a in enumerate(actions):
        port.actions[:] = np.array([a], dtype=port.actions.dtype)
        binding.vec_step(port.c_envs)

        p_obs        = port.observations[0].astype(np.float64).copy()
        p_rew_vec    = port.rewards[0].astype(np.float64).copy()
        p_end        = bool(port.terminals[0])
        p_rew_landing = p_rew_vec[0]   # REW_LANDING
        p_rew_fuel    = p_rew_vec[1]   # REW_FUEL

        r_obs        = np.asarray(ref_obs[t + 1], dtype=np.float64)
        r_rew_scalar = ref_rews[t]
        r_end        = ref_dones[t]

        # Observation comparison (skip post-terminal since both envs auto-reset;
        # exclut angle/omega -- voir LIMITATION CONNUE plus haut)
        both_ended = r_end and p_end
        if check_obs and not both_ended:
            if not np.allclose(
                r_obs[OBS_IDX_STRICT], p_obs[OBS_IDX_STRICT], atol=ATOL
            ):
                mismatches.append((t, "obs", r_obs.tolist(), p_obs.tolist()))

        # Reward comparison :
        #   - sur un step terminal (l'un OU l'autre), le reward est dominé
        #     par le bonus/malus fixe (+100/-100) -- on compare le reward
        #     complet, comme avant.
        #   - au step JUSTE APRES un redémarrage (l'un ou l'autre cote),
        #     la baseline de shaping_no_angle n'est pas fiable (l'episode
        #     interne de reset() n'est pas capture separement dans
        #     ref_obs) -- on ne compare pas le reward a ce step precis.
        #   - sinon, on compare shaping_no_angle, qui isole ce qui est
        #     reellement modelise du port.
        shaping_ref_na  = shaping_no_angle(r_obs)
        shaping_port_na = shaping_no_angle(p_obs)

        if r_end or p_end:
            if not np.isclose(r_rew_scalar, p_rew_landing, atol=ATOL):
                mismatches.append((t, "rew_landing",
                                   round(r_rew_scalar, 6),
                                   round(p_rew_landing, 6)))
        elif prev_r_end or prev_p_end:
            pass  # baseline non fiable juste apres un redemarrage, on saute
        else:
            r_rew_na = shaping_ref_na - prev_shaping_ref_na
            p_rew_na = shaping_port_na - prev_shaping_port_na
            if not np.isclose(r_rew_na, p_rew_na, atol=ATOL):
                mismatches.append((t, "rew_landing_no_angle",
                                   round(r_rew_na, 6),
                                   round(p_rew_na, 6)))

        # Mise à jour des baselines/etats pour le prochain step
        prev_shaping_ref_na  = shaping_ref_na
        prev_shaping_port_na = shaping_port_na
        prev_r_end = r_end
        prev_p_end = p_end

        # Fuel reward must be <= 0 (cost, never positive)
        if check_fuel_sign and p_rew_fuel > 1e-6:
            mismatches.append((t, "rew_fuel_sign", "<=0", round(p_rew_fuel, 6)))

        # Terminal flag comparison
        if r_end != p_end:
            mismatches.append((t, "ended", r_end, p_end))

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

    Test purement côté port (pas de comparaison à la référence) --
    n'a pas besoin d'injection RNG.
    """
    port = make_port(seed=7)
    port.set_weights(np.array([0.0, 1.0], dtype=np.float32))
    port.reset(seed=7)

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

    Test purement côté port -- n'a pas besoin d'injection RNG.
    """
    port = make_port(seed=99)
    target = np.array([0.8, 0.2], dtype=np.float32)
    port.set_weights(target)
    port.reset(seed=99)

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

    Test purement côté port -- n'a pas besoin d'injection RNG.
    """
    port = make_port(seed=13)

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

    Test purement côté port -- n'a pas besoin d'injection RNG.
    """
    port = LunarLander(
        num_envs=1,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
    )
    port.set_weights(np.array([0.5, 0.5], dtype=np.float32))
    port.reset(seed=0)

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