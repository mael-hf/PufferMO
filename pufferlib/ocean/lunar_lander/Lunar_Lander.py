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
from gymnasium.utils import seeding

import pufferlib
from pufferlib.ocean.lunar_lander import binding

REWARD_DIM = 2


# ══════════════════════════════════════════════════════════════════════
# Enregistrement de la RNG de référence (pour les tests de conformité)
# ══════════════════════════════════════════════════════════════════════
#
# Gymnasium tire ses nombres aléatoires via deux méthodes de
# np.random.Generator :
#
#     np_random.uniform(low, high, size=...)
#     np_random.integers(low, high, size=...)   # wind_idx / torque_idx
#
# Les deux sont des transformations affines d'un tirage brut u dans [0,1).
# RecordingGenerator remonte à ce u par inversion générique :
#
#     u = (valeur - low) / (high - low)
#
# ll_randf()/ll_randrange() côté C (lunar_lander.h) utilisent exactement
# la même transformation, donc injecter cette tape de u reproduit
# fidèlement les valeurs réelles de la référence, sans réimplémenter
# l'algorithme RNG de NumPy (PCG64) en C.
#
# Ordre des tirages consommés par la référence (vérifié dans le code
# source de Gymnasium) :
#
#   reset(seed) :
#     1.  uniform(0, H/2, size=12)              -> terrain (12 valeurs)
#     2.  uniform(-1000, 1000)                  -> force initiale x
#     3.  uniform(-1000, 1000)                  -> force initiale y
#     4.  [si enable_wind] integers(-9999,9999) -> wind_idx
#     5.  [si enable_wind] integers(-9999,9999) -> torque_idx
#     6.  uniform(-1, 1)                        -> dispersion[0] (step(0) interne)
#     7.  uniform(-1, 1)                        -> dispersion[1] (step(0) interne)
#
#   step(action), à chaque appel :
#     1.  uniform(-1, 1) -> dispersion[0]
#     2.  uniform(-1, 1) -> dispersion[1]
#
# Cet ordre doit rester strictement synchronisé avec celui consommé côté
# C (lunar_lander.h / c_reset / c_step).


class RecordingGenerator:
    """
    Enveloppe un np.random.Generator pour enregistrer, dans l'ordre,
    la version "brute dans [0,1)" de chaque tirage effectué.
    """

    def __init__(self, gen: np.random.Generator):
        self._gen = gen
        self.tape: list = []

    def uniform(self, low=0.0, high=1.0, size=None):
        val = self._gen.uniform(low, high, size)
        arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
        span = float(high) - float(low)
        raw = (arr - float(low)) / span
        self.tape.extend(raw.tolist())
        return val

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        val = self._gen.integers(low, high, size=size, dtype=dtype, endpoint=endpoint)
        arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
        span = float(high) - float(low) + (1.0 if endpoint else 0.0)
        raw = (arr - float(low)) / span
        self.tape.extend(raw.tolist())
        return val

    # délègue tout le reste (bit_generator, standard_normal, etc.)
    def __getattr__(self, name):
        return getattr(self._gen, name)


def _wrap_np_random_with_recorder(env, seed: int) -> RecordingGenerator:
    """
    Crée un Generator NumPy fraîchement seedé avec `seed`, l'enveloppe dans
    un RecordingGenerator, et l'installe directement sur l'env -- AVANT
    tout tirage.

    ATTENTION : n'appelez ensuite env.reset(...) qu'avec seed=None sur cet
    env. gym.Env.reset() contient :

        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

    Donc tout appel reset(seed=<non-None>) après celui-ci écrase
    silencieusement le recorder par un nouveau générateur (c'était le bug
    initial : une tape toujours vide).
    """
    gen, np_seed = seeding.np_random(seed)
    recorder = RecordingGenerator(gen)
    env.unwrapped._np_random = recorder
    env.unwrapped._np_random_seed = np_seed
    return recorder


def record_reference_tape(env, seed: int, actions: list):
    """
    Rejoue `actions` sur l'environnement Gymnasium `env` (déjà créé,
    PAS encore reseté), en enregistrant la tape RNG complète depuis le
    tout premier tirage, et renvoie (tape, observations, rewards,
    terminations) pour comparaison avec le port C.

    Parameters
    ----------
    env     : gymnasium.Env  (ex: gym.make("LunarLander-v3"))
    seed    : int
    actions : list[int]

    Returns
    -------
    tape  : np.ndarray, float32, à passer à LunarLander.set_rng_tape()
    obs   : liste des observations (y compris l'observation initiale)
    rews  : liste des récompenses (scalaires, REW_LANDING uniquement)
    dones : liste des booléens terminated/truncated combinés
    """
    recorder = _wrap_np_random_with_recorder(env, seed)

    # seed=None ici, sinon Gymnasium re-seede et écrase le recorder.
    obs0, _ = env.reset(seed=None)
    obs   = [np.asarray(obs0, dtype=np.float64)]
    rews  = []
    dones = []

    for a in actions:
        o, r, term, trunc, _ = env.step(a)
        obs.append(np.asarray(o, dtype=np.float64))
        rews.append(float(r))
        dones.append(bool(term or trunc))
        if term or trunc:
            break

    tape = np.array(recorder.tape, dtype=np.float32)
    return tape, obs, rews, dones


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