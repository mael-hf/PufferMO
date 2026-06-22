"""
verify_rng_tape.py
==================
Vérifie la cohérence de la tape RNG produite par rng_tape_recorder.py,
SANS dépendre du port C compilé. Deux types de vérifications :

  A. Longueur attendue : on calcule analytiquement combien de tirages
     la référence Gymnasium doit consommer pour une séquence d'actions
     donnée, et on compare à len(tape) réellement enregistrée.

  B. Cohérence de l'inversion raw <-> valeur : on reconstruit la valeur
     finale à partir du raw enregistré et on vérifie qu'elle retombe
     exactement sur la valeur réellement tirée par la référence (test
     de non-régression du recorder lui-même, indépendant du C).

Usage :
    python verify_rng_tape.py
"""

import numpy as np
import gymnasium as gym

from rng_tape_recorder import RecordingGenerator


# ─────────────────────────────────────────────────────────────────────
# A. Longueur attendue de la tape
# ─────────────────────────────────────────────────────────────────────

def expected_tape_length(n_actions: int, enable_wind: bool) -> int:
    """
    Nombre de tirages RNG bruts que reset() + n_actions x step() doivent
    consommer côté référence Gymnasium.

        reset() :
            12  terrain          (uniform, size=12)
             2  force initiale   (uniform x, uniform y)
             2  wind/torque_idx  (integers, SEULEMENT si enable_wind)
             2  dispersion       (step(0) interne)

        step() x n_actions :
             2  dispersion par appel
    """
    n = 12 + 2 + (2 if enable_wind else 0) + 2
    n += 2 * n_actions
    return n


def check_length(seed: int, n_actions: int, enable_wind: bool) -> bool:
    env = gym.make("LunarLander-v3", enable_wind=enable_wind)
    env.reset(seed=seed)
    recorder = RecordingGenerator(env.unwrapped.np_random)
    env.unwrapped.np_random = recorder

    env.reset(seed=seed)
    actions = np.random.default_rng(123).integers(0, 4, size=n_actions).tolist()

    n_played = 0
    for a in actions:
        _, _, term, trunc, _ = env.step(a)
        n_played += 1
        if term or trunc:
            break
    env.close()

    expected = expected_tape_length(n_played, enable_wind)
    actual = len(recorder.tape)
    ok = (actual == expected)
    status = "OK" if ok else "MISMATCH"
    suffix = " (episode termine avant la fin)" if n_played < n_actions else ""
    print(f"[{status}] seed={seed} n_actions_demandes={n_actions} "
          f"n_actions_jouees={n_played}{suffix} enable_wind={enable_wind}  "
          f"attendu={expected}  obtenu={actual}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# B. Cohérence de l'inversion raw <-> valeur
# ─────────────────────────────────────────────────────────────────────

def check_inversion_roundtrip(seed: int = 0, n_actions: int = 10) -> bool:
    """
    Ré-enregistre avec un recorder "verbeux" qui garde aussi la valeur
    réelle tirée (pas seulement le raw), puis vérifie que
        low + raw * (high - low) == valeur_reelle
    pour chaque tirage. Ça ne teste pas le C, mais ça garantit que la
    tape elle-même n'a pas de bug d'inversion avant de l'injecter.
    """
    class VerboseRecorder(RecordingGenerator):
        def __init__(self, gen):
            super().__init__(gen)
            self.checks = []  # (raw, low, high, valeur_reelle)

        def uniform(self, low=0.0, high=1.0, size=None):
            val = self._gen.uniform(low, high, size)
            arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
            span = float(high) - float(low)
            raw = (arr - float(low)) / span
            self.tape.extend(raw.tolist())
            for r, v in zip(raw, arr):
                self.checks.append((r, float(low), float(high), float(v)))
            return val

        def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
            val = self._gen.integers(low, high, size=size, dtype=dtype, endpoint=endpoint)
            arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
            span = float(high) - float(low) + (1.0 if endpoint else 0.0)
            raw = (arr - float(low)) / span
            self.tape.extend(raw.tolist())
            for r, v in zip(raw, arr):
                self.checks.append((r, float(low), float(high), float(v)))
            return val

    env = gym.make("LunarLander-v3", enable_wind=True)
    env.reset(seed=seed)
    rec = VerboseRecorder(env.unwrapped.np_random)
    env.unwrapped.np_random = rec
    env.reset(seed=seed)

    rng = np.random.default_rng(7)
    for a in rng.integers(0, 4, size=n_actions):
        _, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            break
    env.close()

    max_err = 0.0
    n_bad = 0
    for raw, low, high, real_val in rec.checks:
        reconstructed = low + raw * (high - low)
        err = abs(reconstructed - real_val)
        max_err = max(max_err, err)
        if err > 1e-9:
            n_bad += 1

    ok = (n_bad == 0)
    status = "OK" if ok else "MISMATCH"
    print(f"[{status}] round-trip inversion sur {len(rec.checks)} tirages "
          f"-- erreur max={max_err:.2e}, tirages incoherents={n_bad}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# C. Aperçu lisible des premières valeurs (pour inspection manuelle)
# ─────────────────────────────────────────────────────────────────────

def print_tape_preview(seed: int = 0, n_actions: int = 3, enable_wind: bool = False):
    from rng_tape_recorder import record_reference_tape

    env = gym.make("LunarLander-v3", enable_wind=enable_wind)
    actions = [0] * n_actions
    tape, obs, rews, dones = record_reference_tape(env, seed, actions)

    labels = (
        [f"terrain[{i}]" for i in range(12)]
        + ["force_x", "force_y"]
        + (["wind_idx", "torque_idx"] if enable_wind else [])
        + ["disp0_reset_step0", "disp1_reset_step0"]
    )
    for i in range(n_actions):
        labels += [f"disp0_step{i}", f"disp1_step{i}"]

    print(f"\n--- Apercu tape (seed={seed}, enable_wind={enable_wind}) ---")
    for i, label in enumerate(labels):
        if i < len(tape):
            print(f"  [{i:3d}] {label:20s} raw={tape[i]:.6f}")

    if len(tape) != len(labels):
        print(f"  ATTENTION: {len(tape)} valeurs dans la tape mais "
              f"{len(labels)} labels attendus -- decalage probable.")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== A. Verification des longueurs attendues ===")
    results_a = [
        check_length(seed=0, n_actions=0,  enable_wind=False),
        check_length(seed=0, n_actions=5,  enable_wind=False),
        check_length(seed=1, n_actions=50, enable_wind=False),
        check_length(seed=2, n_actions=20, enable_wind=True),
    ]

    print("\n=== B. Verification de l'inversion raw <-> valeur ===")
    results_b = [
        check_inversion_roundtrip(seed=0, n_actions=20),
        check_inversion_roundtrip(seed=42, n_actions=50),
    ]

    print_tape_preview(seed=0, n_actions=3, enable_wind=False)
    print_tape_preview(seed=0, n_actions=3, enable_wind=True)

    print()
    if all(results_a) and all(results_b):
        print("TOUTES LES VERIFICATIONS (cote Python) SONT PASSEES.")
        print("Prochaine etape : injecter la tape dans le port C et verifier")
        print("l'absence du message stderr 'tape RNG epuisee' (voir cote C).")
    else:
        print("DES INCOHERENCES ONT ETE DETECTEES -- a corriger avant")
        print("d'injecter la tape dans le port C.")