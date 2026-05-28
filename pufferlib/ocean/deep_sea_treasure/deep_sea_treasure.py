'''MO-Gymnasium deep-sea-treasure ported to PufferMO.

Single-agent submarine on an 11x11 grid. Two objectives: treasure value
(collected on reaching a chest, which is terminal) and time penalty (-1
every step). Convex DEFAULT_MAP from Yang et al. (2019). Episodes truncate
at 100 steps to match MO-Gym's TimeLimit wrapper.
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.deep_sea_treasure import binding


class DeepSeaTreasure(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        gamma=0.99,
        report_interval=128,
        render_mode='human',
        buf=None,
        seed=0,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.tick = 0

        # One agent per env
        self.num_agents = num_envs

        # Observation: [row, col], each in [0, 10]
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=10, shape=(2,), dtype=np.uint8
        )

        # Action: 0=up, 1=down, 2=left, 3=right
        self.single_action_space = gymnasium.spaces.Discrete(4)

        # Two objectives: treasure value, time penalty
        self.reward_dim = 2
        self.single_reward_space = gymnasium.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([23.7, -1.0], dtype=np.float32),
            shape=(self.reward_dim,),
            dtype=np.float32,
        )

        super().__init__(buf, multiobjective_reward=True)

        c_envs = []
        for i in range(num_envs):
            obs_slice     = self.observations[i:i+1]
            act_slice     = self.actions[i:i+1]
            rew_slice     = self.rewards[i:i+1, :]
            weights_slice = self.weights[i:i+1, :]
            term_slice    = self.terminals[i:i+1]
            trunc_slice   = self.truncations[i:i+1]

            env_seed = i + seed * num_envs
            env_id = binding.env_init(
                obs_slice,
                act_slice,
                rew_slice,
                weights_slice,
                term_slice,
                trunc_slice,
                env_seed,
                gamma=gamma,
            )
            c_envs.append(env_id)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        infos = []
        if self.tick % self.report_interval == 0:
            infos.append(binding.vec_log(self.c_envs))

        return (
            self.observations,
            self.rewards,
            self.weights,
            self.terminals,
            self.truncations,
            infos,
        )

    def set_weights(self, weights):
        '''Inject fixed weights for evaluation. weights is a 1D float32 array
        of length 2 (treasure, time).'''
        binding.vec_put(self.c_envs, weights=weights)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    env = DeepSeaTreasure()
    env.reset()
    tick = 0

    actions = np.random.randint(0, 4, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start):.2f}')


if __name__ == '__main__':
    test_performance()