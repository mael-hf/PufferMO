"""
LunarLander — Multi-Objective PufferLib environment.

Two reward objectives:
    0 (REW_LANDING) : shaping-based guidance + landing bonus / crash penalty
    1 (REW_FUEL)    : fuel efficiency  (negative engine usage cost)

Weights are sampled from Dirichlet(1,1) at each episode reset unless
set_weights() has been called with a manual preference vector.

Pour les tests de conformité (test_conformity_LL.py), set_rng_tape()
permet d'injecter une tape de tirages aléatoires bruts enregistrée
depuis l'environnement Gymnasium de référence — voir rng_tape_recorder.py.
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

    def set_rng_tape(self, tape: np.ndarray):
        """
        Injecte une tape de tirages aléatoires bruts (valeurs dans [0,1)),
        enregistrée depuis l'environnement Gymnasium de référence, pour
        les tests de conformité.

        Une fois la tape épuisée, l'env C affiche un avertissement sur
        stderr et retombe sur une valeur neutre (0.5) — signe qu'il y a
        eu une désynchronisation entre le nombre de tirages attendus et
        ceux réellement consommés (cf. rng_tape_recorder.py).

        ATTENTION : ce mécanisme est pensé pour num_envs=1 (le cas
        d'usage de test_conformity_LL.py). Avec num_envs>1, la tape
        n'est appliquée qu'à un seul environnement sous-jacent — ne
        pas l'utiliser pour de l'entraînement multi-env.

        Parameters
        ----------
        tape : np.ndarray, dtype float32, longueur quelconque > REWARD_DIM
            Séquence de tirages bruts dans [0,1), dans l'ordre exact de
            consommation côté C (terrain, force initiale, wind/torque_idx
            si activé, puis 2 valeurs de dispersion par reset/step).
        """
        tape = np.ascontiguousarray(tape, dtype=np.float32)
        if tape.ndim != 1:
            raise ValueError("set_rng_tape: expected a 1-D array")
        binding.vec_put(self.c_envs, tape)

    # ------------------------------------------------------------------
    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


# ──────────────────────────────────────────────────────────────────────


def test_performance(timeout=10, atn_cache=1024):
    N    = 64
    env  = LunarLander(num_envs=N)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 4, (atn_cache, N))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start):.2f}')
    env.close()


if __name__ == "__main__":
    test_performance()
