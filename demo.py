import numpy as np
import time

# Import de votre environnement PufferLib
from pufferlib.ocean.overcooked.overcooked import OvercookedEnv
# Import du moteur de rendu Raylib officiel de PufferLib
from pufferlib.ocean.render import GridRender

# --- 1. Couleurs (RGB) ---
OVERCOOKED_COLORS = np.array([
    [220, 220, 220], # 0: EMPTY (Sol gris clair)
    [210, 180, 140], # 1: COUNTER (Plan de travail bois)
    [200, 50, 50],   # 2: STOVE (Cuisinière rouge)
    [139, 69, 19],   # 3: CUTTING_BOARD (Planche marron)
    [50, 200, 50],   # 4: INGREDIENT_BOX (Boîte d'oignons verte)
    [255, 215, 0],   # 5: SERVING_AREA (Zone de service dorée)
    [100, 100, 100], # 6: WALL (Mur gris foncé)
    [255, 255, 255], # 7: PLATE_BOX (Assiettes blanches)
    [0, 100, 255],   # 8: AGENT 1 (Bleu)
    [255, 100, 0],   # 9: AGENT 2 (Orange)
], dtype=np.uint8)

# --- 2. La carte de base (CRAMPED_ROOM) ---
LAYOUT = np.array([
    [6, 1, 2, 1, 6],
    [4, 0, 0, 0, 4],
    [1, 0, 0, 0, 1],
    [6, 7, 1, 5, 6]
], dtype=np.uint8)

def main():
    print("🚀 Lancement de PufferMO - Overcooked (Multi-Agent, Raylib Colors)")
    
    # Initialisation de l'environnement C
    env = OvercookedEnv(num_envs=1, layout=0)
    env.reset(seed=42)
    
    height, width = LAYOUT.shape
    
    # Instanciation du moteur Raylib
    renderer = GridRender(
        width, height, 
        screen_width=800, screen_height=640, 
        colors=OVERCOOKED_COLORS, 
        name='PufferMO - Fast MARL Demo'
    )

    running = True
    while running:
        # Actions aléatoires (en attendant votre modèle IA)
        actions = np.array([[np.random.randint(0, 6), np.random.randint(0, 6)]])
        env.step(actions)
        obs = env.get_agent_observations()
        
        # --- Rafraîchissement de la grille ---
        frame = LAYOUT.copy()
        
        # Extraction des coordonnées X, Y depuis l'observation générée par votre C
        agent_1_obs = obs[0][0]
        agent_2_obs = obs[0][1]
        
        a1_y, a1_x = int(agent_1_obs[-3] * height), int(agent_1_obs[-2] * width)
        a2_y, a2_x = int(agent_2_obs[-3] * height), int(agent_2_obs[-2] * width)
        
        # Placement des agents sur la grille
        if 0 <= a1_y < height and 0 <= a1_x < width:
            frame[a1_y, a1_x] = 8
        if 0 <= a2_y < height and 0 <= a2_x < width:
            frame[a2_y, a2_x] = 9

        # --- Envoi au moteur graphique C ---
        output = renderer.render(frame, end_drawing=True)
        
        # Ralentissement artificiel à ~10 FPS pour que ce soit regardable à l'œil nu
        time.sleep(0.1) 

if __name__ == "__main__":
    main()