"""
MO-LunarLander-v2 — affichage humain (render_mode="human")
Dépendances :
    pip install mo-gymnasium pygame box2d-py
"""

import mo_gymnasium as mo_gym
import numpy as np


def run(n_episodes: int = 3, max_steps: int = 1_000, seed: int = 42) -> None:
    """Lance l'environnement MO-LunarLander-v2 avec rendu graphique."""

    env = mo_gym.make("mo-lunar-lander-v3", render_mode="human")

    print(f"Espace d'observation : {env.observation_space}")
    print(f"Espace d'action      : {env.action_space}")
    print(f"Espace de récompense : {env.unwrapped.reward_space}")   # vecteur multi-objectif
    print()

    rng = np.random.default_rng(seed)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        total_reward = np.zeros(env.unwrapped.reward_space.shape)
        done = False
        step = 0

        print(f"=== Épisode {ep}/{n_episodes} ===")

        while not done and step < max_steps:
            # Politique aléatoire — remplacez par votre propre agent ici
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # `reward` est un vecteur numpy (ex. [récompense_atterrissage, carburant])
            total_reward += reward
            step += 1

        print(f"  Steps : {step}")
        print(f"  Récompense totale (vecteur) : {total_reward}")
        print()

    env.close()
    print("Simulation terminée.")


if __name__ == "__main__":
    run(n_episodes=10)
