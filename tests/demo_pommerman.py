import numpy as np
import time
from pufferlib.ocean.pommerman.pommerman import Pommerman

def run_demo():
    print("Lancement de la démo Pommerman...")
    env = Pommerman(num_envs=1)
    env.reset()
    
    try:
        while True:
            # Format correct pour PufferLib : un vecteur plat (4,)
            actions = np.random.randint(0, 6, size=(4,))
            
            obs, rewards, terminals, truncations, info = env.step(actions)
            env.render()
            
            # terminals est un tableau 1D. On vérifie si l'agent 0 a fini.
            if terminals[0]:
                print("Fin de l'épisode, redémarrage...")
                env.reset()
                
    except KeyboardInterrupt:
        print("\nFermeture demandée par l'utilisateur.")
    except Exception as e:
        print(f"\nCRASH PYTHON : {e}")
    finally:
        env.close()

if __name__ == "__main__":
    run_demo()
    