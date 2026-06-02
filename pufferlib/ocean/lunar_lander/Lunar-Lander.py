"""
LunarLander — Multi-Objective PufferLib environment.

Two reward objectives:
    0 (REW_LANDING) : shaping-based guidance + landing bonus / crash penalty
    1 (REW_FUEL)    : fuel efficiency  (negative engine usage cost)

Weights are sampled from Dirichlet(1,1) at each episode reset unless
set_weights() has been called with a manual preference vector.
"""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.lunar_lander import binding

REWARD_DIM = 2


class LunarLander(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs: int = 1,
        render_mode=None,
        log_interval: int = 128,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        buf=None,
        seed: int = 0,
        **kwargs,           # absorb unused config keys
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Discrete(4)

        # Multi-objective reward space
        self.reward_dim = REWARD_DIM
        self.single_reward_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(REWARD_DIM,),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.num_agents  = num_envs

        super().__init__(buf, multiobjective_reward=True)

        # Initialise one C env per agent, each with its own obs/act/rew slices
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.weights,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            enable_wind=int(enable_wind),
            wind_power=float(wind_power),
            turbulence_power=float(turbulence_power),
        )

    # ------------------------------------------------------------------
    def reset(self, seed: int = 0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return (
            self.observations,
            self.rewards,       # shape (num_envs, REWARD_DIM)
            self.terminals,
            self.truncations,
            info,
        )

    # ------------------------------------------------------------------
    def set_weights(self, weights: np.ndarray):
        """
        Set scalarisation weights for all envs.

        Parameters
        ----------
        weights : np.ndarray, shape (REWARD_DIM,) or (num_envs, REWARD_DIM)
            Preference vector(s). Should sum to 1 per env (not enforced here).
        """
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 1:
            weights = np.broadcast_to(weights, (self.num_agents, REWARD_DIM)).copy()
        self.weights[:] = weights
        binding.vec_put(self.c_envs, weights)

    # ------------------------------------------------------------------
    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    N    = 4096
    env  = LunarLander(num_envs=N)
    env.reset()
    steps = 0

    CACHE   = 1024
    actions = np.random.randint(0, 4, (CACHE, N))

    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    sps = int(env.num_agents * steps / (time.time() - start))
    print(f"Squared SPS: {sps}")
    env.close()