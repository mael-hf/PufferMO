'''MO-Gymnasium resource-gathering ported to PufferMO.

Single-agent, 5x5 grid. Agent starts at home (4,2), navigates to collect
gold at (0,2) and a gem at (1,4), avoiding enemies at (1,2) and (0,3).
Stepping onto an enemy gives a 10% chance of death (-1 on objective 0).
Returning home pays +1 per resource carried (objectives 1 and 2).
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.resource_gathering import binding


class ResourceGathering(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        kill_probability=0.1,
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

        # One agent per env in this environment
        self.num_agents = num_envs

        # Observation: [row, col, has_gold, has_gem], each in [0, 4]
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=4, shape=(4,), dtype=np.uint8
        )

        # Action: 0=up, 1=down, 2=left, 3=right
        self.single_action_space = gymnasium.spaces.Discrete(4)

        # Three objectives: kill penalty, gold reward, gem reward
        self.reward_dim = 3
        self.single_reward_space = gymnasium.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.0, 1.0, 1.0], dtype=np.float32),
            shape=(self.reward_dim,),
            dtype=np.float32,
        )

        # Allocate batched buffers (multiobjective_reward=True widens rewards
        # and allocates the weights buffer)
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
                kill_probability=kill_probability,
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
        of length 3 (kill, gold, gem). Overrides the per-episode Dirichlet
        sampling done in c_reset.'''
        binding.vec_put(self.c_envs, weights=weights)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    env = ResourceGathering()
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