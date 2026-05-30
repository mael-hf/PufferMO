import pufferlib
import gymnasium as gym
import numpy as np

class OvercookedEnv(pufferlib.PufferEnv):
    def __init__(self, buf, num_envs=1, **kwargs):
        self.num_agents = num_envs * 2 
        super().__init__(buf, multiobjective_reward=False)

        # Taille officielle extraite de overcooked_obs.h
        self.obs_size = 43 
        
        self.single_observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        
        # L'officiel gère 6 actions (0: NOOP, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT, 5: INTERACT)
        self.single_action_space = gym.spaces.Discrete(6)

    def get_agent_observations(self):
        agent_obs = []
        for i in range(self.num_envs):
            obs_slice = self.observations[i * 2 : (i + 1) * 2]
            agent_obs.append(obs_slice)
        return np.array(agent_obs)