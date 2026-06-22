"""
test_rng_injection.py
======================
Teste l'injection de la tape RNG dans le port C compilé.

Deux cas :
  1. POSITIF : tape complète -> aucun avertissement stderr attendu.
  2. NEGATIF  : tape tronquée d'une valeur -> l'avertissement
                "tape RNG epuisee" DOIT apparaître. Si ce test négatif
                échoue (pas d'avertissement), c'est que la détection
                elle-même est cassée, et donc que le silence du cas
                positif ne veut rien dire.

Usage :
    python test_rng_injection.py
"""

import sys
import numpy as np
import gymnasium as gym

from pufferlib.ocean.lunar_lander.Lunar_Lander import LunarLander, record_reference_tape


def run_case(tape: np.ndarray, actions: list, seed: int, label: str):
    print(f"\n--- Cas: {label} ---")
    port = LunarLander(num_envs=1, enable_wind=False)
    port.set_weights(np.array([1.0, 0.0], dtype=np.float32))
    port.set_rng_tape(tape)
    port.reset(seed=seed)

    import pufferlib.ocean.lunar_lander.binding as binding
    for a in actions:
        port.actions[:] = np.array([a], dtype=port.actions.dtype)
        binding.vec_step(port.c_envs)
        if port.terminals[0]:
            break
    port.close()


if __name__ == "__main__":
    SEED = 0
    ACTIONS = [0, 2, 1, 3, 0, 2, 2, 1]

    ref = gym.make("LunarLander-v3")
    tape, ref_obs, ref_rews, ref_dones = record_reference_tape(ref, SEED, ACTIONS)
    ref.close()

    print(f"Tape enregistree : {len(tape)} valeurs pour {len(ACTIONS)} actions.")

    print("\n" + "=" * 70)
    print("CAS POSITIF -- tape complete. Surveillez l'absence du message")
    print("'[lunar_lander] ATTENTION: tape RNG epuisee' ci-dessous.")
    print("=" * 70)
    run_case(tape, ACTIONS, SEED, "tape complete (positif)")

    print("\n" + "=" * 70)
    print("CAS NEGATIF -- tape tronquee d'une valeur. Le message")
    print("'[lunar_lander] ATTENTION: tape RNG epuisee' DOIT apparaitre")
    print("ci-dessous. S'il n'apparait pas, la detection est cassee.")
    print("=" * 70)
    run_case(tape[:-1], ACTIONS, SEED, "tape tronquee (negatif -- doit avertir)")

    print("\n" + "=" * 70)
    print("Lisez les deux blocs stderr ci-dessus pour conclure :")
    print("  - cas positif sans avertissement  -> bon signe")
    print("  - cas negatif AVEC avertissement  -> la detection fonctionne")
    print("Si les deux sont vrais, l'injection est correctement synchronisee.")
    print("=" * 70)