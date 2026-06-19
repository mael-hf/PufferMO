import numpy as np
import sys
import os

# Import de l'environnement 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pufferlib.ocean.pommerman.pommerman import Pommerman

# Constantes (Doivent correspondre à pommerman.h) 
BOARD_SIZE = 11
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
OBS_PER_AGENT = NUM_CELLS * 3 + 8

ACTION_STOP, ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_BOMB = 0, 1, 2, 3, 4, 5
CELL_AGENT_0 = 10

PASSED = []
FAILED = []

def get_board(obs, agent_idx=0):
    """Extrait la grille de jeu depuis les observations (Channel 0, normalisé par 13)"""
    board_flat = np.round(obs[agent_idx][:NUM_CELLS] * 13).astype(int)
    return board_flat.reshape((BOARD_SIZE, BOARD_SIZE))

def get_ammo(obs, agent_idx=0):
    """Extrait les munitions de l'agent depuis les scalaires (normalisé par 10)"""
    ammo_idx = NUM_CELLS * 3 + 2
    return round(obs[agent_idx][ammo_idx] * 10)

def run_test(name, test_func):
    print(f"\n[TEST] {name}")
    try:
        env = Pommerman(num_envs=1)
        env.reset()
        success, msg = test_func(env)
        env.close()
        
        if success:
            print(f"  [PASS] {msg}")
            PASSED.append(name)
        else:
            print(f"  [FAIL] {msg}")
            FAILED.append(name)
    except Exception as e:
        print(f"  [CRASH] Erreur fatale : {e}")
        FAILED.append(name)


# TEST 1 : Blocage par les frontières (Wall Blocking)

def test_wall_blocking(env):
    actions = np.zeros(4, dtype=int)
    
    # L'Agent 0 spawn en (0,0). On essaie de monter (UP) en dehors de la map.
    actions[0] = ACTION_UP
    obs, _, _, _, _ = env.step(actions)
    board = get_board(obs)
    if board[0, 0] != CELL_AGENT_0:
        return False, "L'agent a traversé le mur du haut !"

    # On essaie d'aller à gauche (LEFT)
    actions[0] = ACTION_LEFT
    obs, _, _, _, _ = env.step(actions)
    board = get_board(obs)
    if board[0, 0] != CELL_AGENT_0:
        return False, "L'agent a traversé le mur de gauche !"

    return True, "Blocage aux frontières parfaitement respecté."


# TEST 2 : Mouvement dans la Clear Zone

def test_basic_movement(env):
    actions = np.zeros(4, dtype=int)
    
    # Mouvement vers le bas (0,0) -> (1,0) (Garanti vide par pmm_is_clear_zone)
    actions[0] = ACTION_DOWN
    obs, _, _, _, _ = env.step(actions)
    board = get_board(obs)
    if board[1, 0] != CELL_AGENT_0:
        return False, "L'agent n'a pas réussi à descendre en (1,0)."

    # Mouvement vers la droite (1,0) -> (1,1)
    actions[0] = ACTION_RIGHT
    obs, _, _, _, _ = env.step(actions)
    board = get_board(obs)
    if board[1, 1] != CELL_AGENT_0:
        return False, "L'agent n'a pas réussi à aller à droite en (1,1)."

    return True, "Les mouvements libres fonctionnent correctement."


# TEST 3 : Chronomètre de la bombe et dégâts

def test_bomb_timer(env):
    actions = np.zeros(4, dtype=int)
    
    # Tick 0 : Pose la bombe
    actions[0] = ACTION_BOMB
    obs, _, _, _, _ = env.step(actions)

    # Ticks 1 à 9 : Reste sur place
    actions[0] = ACTION_STOP
    for step in range(1, 10):
        obs, rewards, terminals, _, _ = env.step(actions)
        if rewards[0] < 0:
            return False, f"La bombe a explosé trop tôt ! (Au tick {step})"

    # Tick 10 : L'explosion
    obs, rewards, terminals, _, _ = env.step(actions)
    if rewards[0] != -1.0:
        return False, "L'agent n'a pas reçu sa pénalité de mort (-1) après l'explosion."
    
    return True, "La bombe explose exactement après 10 ticks et inflige les dégâts."


# TEST 4 : Rechargement des munitions (Ammo Management)

def test_ammo_reload(env):
    # Vérification initiale
    obs, _ = env.reset()
    if get_ammo(obs) != 1:
        return False, "L'agent ne commence pas avec 1 munition."

    # Pose une bombe
    actions = np.zeros(4, dtype=int)
    actions[0] = ACTION_BOMB
    obs, _, _, _, _ = env.step(actions)
    
    if get_ammo(obs) != 0:
        return False, "La munition n'a pas été décrémentée après la pose."

    # Attend l'explosion
    actions[0] = ACTION_STOP
    for _ in range(10):
        obs, _, _, _, _ = env.step(actions)
        
    if get_ammo(obs) != 1:
        return False, "La munition n'a pas été rendue après l'explosion."

    return True, "Le cycle des munitions est géré à la perfection."


# MAIN

if __name__ == "__main__":
    print("=" * 60)
    print("  TESTS DE CONFORMITÉ — Pommerman C vs Règles Officielles")
    print("=" * 60)

    run_test("wall_blocking", test_wall_blocking)
    run_test("basic_movement", test_basic_movement)
    run_test("bomb_timer", test_bomb_timer)
    run_test("ammo_reload", test_ammo_reload)

    print("\n" + "=" * 60)
    total = len(PASSED) + len(FAILED)
    if not FAILED:
        print(f"  RÉSULTAT : {len(PASSED)}/{total} tests PASSÉS")
    else:
        print(f"  RÉSULTAT : {len(PASSED)}/{total} tests passés")
        print(f"  Échecs   : {', '.join(FAILED)}")
    print("=" * 60)