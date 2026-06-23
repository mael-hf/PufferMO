'''MO-Gymnasium Minecart-v0 ported to PufferMO.

Single-agent continuous cart with mines. The cart drives around a square map,
mines ore from two mine types, and returns to base to deposit.
Rewards: [ore1, ore2, -fuel]
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.minecart import binding


class Minecart(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        frame_skip=4,
        incremental_frame_skip=True,
        capacity=1.0,
        mine_cnt=2,
        gamma=0.99,
        report_interval=128,
        render_mode='human',
        buf=None,
        seed=0,
        max_ticks=0,
        max_ticks_offset_mod=1,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.tick = 0

        self.num_agents = num_envs

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.single_action_space = gymnasium.spaces.Discrete(6)

        self.reward_dim = 3
        self.single_reward_space = gymnasium.spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([capacity, capacity, 0.0], dtype=np.float32),
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
                frame_skip=frame_skip,
                incremental_frame_skip=incremental_frame_skip,
                capacity=capacity,
                mine_cnt=mine_cnt,
                gamma=gamma,
                max_ticks=max_ticks,
                max_ticks_offset_mod=max_ticks_offset_mod,
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
        binding.vec_put(self.c_envs, weights=weights)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    env = Minecart()
    env.reset()
    tick = 0

    actions = np.random.randint(0, 6, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start):.2f}')


if __name__ == '__main__':
    test_performance()
