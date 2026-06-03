import pufferlib
import gymnasium as gym
import numpy as np
from . import binding

class OvercookedEnv(pufferlib.PufferEnv):
    def __init__(self, buf=None, **kwargs):
        self.obs_size = 43
        self.num_agents = 2

        self.single_observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Discrete(6)

        super().__init__(buf=buf)
        
        self.c_bindings = binding 

        self.c_env = binding.env_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            0,  # seed
            layout=int(kwargs.get('layout', 0)),
            num_agents=int(kwargs.get('num_agents', self.num_agents)),
            grid_size=int(kwargs.get('grid_size', 32)),
            max_ticks=int(kwargs.get('max_ticks', 400))
        )

    def get_agent_observations(self):
        return self.observations.reshape(-1, self.num_agents, self.obs_size)
    
    def reset(self, seed=None, options=None):
        binding.env_reset(self.c_env, seed or 0)
        return self.observations, {}
    
    def step(self, actions):
        self.actions[:] = actions
        self.c_bindings.env_step(self.c_env)
        return self.observations, self.rewards, self.terminals, self.truncations, {}
    
    def close(self):
        self.c_bindings.env_close(self.c_env)

def env(**kwargs):
    return OvercookedEnv(**kwargs)

