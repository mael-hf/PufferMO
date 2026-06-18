import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.pommerman import binding

NUM_AGENTS_PER_ENV = 4
OBS_PER_AGENT      = 371   # NUM_CELLS*3 + 8  (must match pommerman.h)
NUM_ACTIONS        = 6     # must match pommerman.h

MODE_FFA  = 0
MODE_TEAM = 1


class Pommerman(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs       = 1,
        render_mode    = None,
        report_interval= 1,
        game_mode      = MODE_FFA,
        max_steps      = 800,
        buf            = None,
        seed           = 0,
    ):
        self.num_agents     = num_envs * NUM_AGENTS_PER_ENV
        self.render_mode    = render_mode
        self.report_interval= report_interval

        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_PER_AGENT,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)

        super().__init__(buf=buf)

        c_envs = []
        for i in range(num_envs):
            env_id = binding.env_init(
                self.observations [i * NUM_AGENTS_PER_ENV : (i + 1) * NUM_AGENTS_PER_ENV],
                self.actions      [i * NUM_AGENTS_PER_ENV : (i + 1) * NUM_AGENTS_PER_ENV],
                self.rewards      [i * NUM_AGENTS_PER_ENV : (i + 1) * NUM_AGENTS_PER_ENV],
                self.terminals    [i * NUM_AGENTS_PER_ENV : (i + 1) * NUM_AGENTS_PER_ENV],
                self.truncations  [i * NUM_AGENTS_PER_ENV : (i + 1) * NUM_AGENTS_PER_ENV],
                i + seed * num_envs,
                game_mode  = game_mode,
                max_steps  = max_steps,
            )
            c_envs.append(env_id)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=512):
    num_envs = 500
    env = Pommerman(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(
        0, NUM_ACTIONS,
        (atn_cache, num_envs * NUM_AGENTS_PER_ENV),
    )

    import time
    start = time.time()
    while time.time() - start < timeout:
        env.step(actions[tick % atn_cache])
        tick += 1

    sps = num_envs * NUM_AGENTS_PER_ENV * tick / (time.time() - start)
    print(f'SPS: {sps:,.0f}')


if __name__ == '__main__':
    test_performance()
