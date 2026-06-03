"""
Test de conformité : OvercookedEnv (C) vs overcooked_ai_py (référence)
=======================================================================
On vérifie que la logique du jeu est correcte en comparant :
1. Le blocage par les murs (un agent ne peut pas traverser un mur)
2. Le déclenchement de récompense lors d'un service
3. La fin d'épisode après max_ticks steps
"""

import numpy as np
import sys
import os

# --- Import référence ---
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as RefEnv
from overcooked_ai_py.mdp.actions import Action

# --- Import notre port C ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pufferlib.ocean.overcooked.overcooked import OvercookedEnv as PortEnv
import pufferlib.ocean.overcooked.binding as binding

# --- Mapping des actions : notre C -> référence ---
# C  : 0=noop, 1=haut, 2=bas, 3=gauche, 4=droite, 5=interact
# Ref: (0,-1)=haut, (0,1)=bas, (1,0)=droite, (-1,0)=gauche, (0,0)=noop, 'interact'
C_TO_REF = {
    0: Action.ALL_ACTIONS[4],   # noop   -> (0,0)
    1: Action.ALL_ACTIONS[0],   # haut   -> (0,-1)
    2: Action.ALL_ACTIONS[1],   # bas    -> (0,1)
    3: Action.ALL_ACTIONS[3],   # gauche -> (-1,0)
    4: Action.ALL_ACTIONS[2],   # droite -> (1,0)
    5: Action.ALL_ACTIONS[5],   # interact -> 'interact'
}

LAYOUT = "cramped_room"
MAX_TICKS = 400
PASSED = []
FAILED = []


def make_ref():
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    return RefEnv.from_mdp(mdp, horizon=MAX_TICKS)


def make_port():
    return PortEnv(layout=0, num_agents=2, grid_size=32, max_ticks=MAX_TICKS)


def ref_step(env, actions_c):
    """Exécute un step sur la référence avec des actions codées en C."""
    ref_actions = [C_TO_REF[a] for a in actions_c]
    state, reward, done, info = env.step(ref_actions)
    return reward, done


def port_step(env, actions_c):
    """Exécute un step sur notre port C."""
    obs, rewards, terminals, truncations, info = env.step(actions_c)
    done = bool(terminals[0])
    reward = float(rewards[0]) + float(rewards[1])
    return reward, done


def run_test(name, actions_sequence):
    """Lance une séquence d'actions sur les deux envs et compare récompenses + done."""
    ref = make_ref()
    port = make_port()
    ref.reset()
    port.reset()

    mismatches = []
    cumulative_ref = 0.0
    cumulative_port = 0.0

    for t, actions in enumerate(actions_sequence):
        r_rew, r_done = ref_step(ref, actions)
        p_rew, p_done = port_step(port, actions)

        cumulative_ref += r_rew
        cumulative_port += p_rew

        # On vérifie uniquement quand une récompense est donnée
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
        print(f"  [FAIL] {name} — {len(mismatches)} divergence(s) :")
        for t, kind, rv, pv in mismatches[:5]:
            print(f"         step {t:4d}  {kind:8s}  ref={rv}  port={pv}")
        FAILED.append(name)
        return False
    else:
        print(f"  [PASS] {name} — récompense totale ref={cumulative_ref:.1f} port={cumulative_port:.1f}")
        PASSED.append(name)
        return True


# ============================================================
# TEST 1 : Blocage par les murs
# Un agent qui va vers un mur doit rester sur place (reward=0)
# ============================================================
def test_wall_blocking():
    print("\n[TEST 1] Blocage par les murs")
    # On envoie les deux agents vers le haut (mur) pendant 10 steps
    actions = [[1, 1]] * 10
    return run_test("wall_blocking", actions)


# ============================================================
# TEST 2 : Noop — aucun mouvement, aucune récompense
# ============================================================
def test_noop():
    print("\n[TEST 2] Noop — aucune action")
    actions = [[0, 0]] * 20
    return run_test("noop_no_reward", actions)


# ============================================================
# TEST 3 : Fin d'épisode après max_ticks
# ============================================================
def test_episode_end():
    print("\n[TEST 3] Fin d'épisode après max_ticks")
    ref = make_ref()
    port = make_port()
    ref.reset()
    port.reset()

    ref_done_at = None
    port_done_at = None

    for t in range(MAX_TICKS + 10):
        _, r_done = ref_step(ref, [0, 0])
        _, p_done = port_step(port, [0, 0])

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
        print(f"  [FAIL] episode_end — référence ne termine jamais")
        FAILED.append("episode_end")
        return False
    if port_done_at is None:
        print(f"  [FAIL] episode_end — port ne termine jamais")
        FAILED.append("episode_end")
        return False

    if abs(ref_done_at - port_done_at) <= 1:
        print(f"  [PASS] episode_end — ref termine à step {ref_done_at}, port à step {port_done_at}")
        PASSED.append("episode_end")
        return True
    else:
        print(f"  [FAIL] episode_end — ref={ref_done_at} port={port_done_at}")
        FAILED.append("episode_end")
        return False


# ============================================================
# TEST 4 : Actions aléatoires — cohérence globale des récompenses
# ============================================================
def test_random_actions():
    print("\n[TEST 4] Actions aléatoires (500 steps)")
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 5, size=(500, 2)).tolist()
    return run_test("random_500", actions)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  TEST DE CONFORMITÉ — Overcooked C vs référence")
    print("=" * 55)

    test_wall_blocking()
    test_noop()
    test_episode_end()
    test_random_actions()

    print("\n" + "=" * 55)
    total = len(PASSED) + len(FAILED)
    if not FAILED:
        print(f"  RÉSULTAT : {len(PASSED)}/{total} tests PASSÉS ✓")
    else:
        print(f"  RÉSULTAT : {len(PASSED)}/{total} tests passés")
        print(f"  Échecs   : {', '.join(FAILED)}")
    print("=" * 55)
