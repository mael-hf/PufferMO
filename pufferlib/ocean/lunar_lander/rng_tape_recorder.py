"""
rng_tape_recorder.py
=====================
Enregistre les tirages aléatoires réels effectués par l'environnement
Gymnasium de référence (LunarLander-v3), puis les convertit en une tape
de valeurs brutes dans [0,1) directement injectable dans le port C via
LunarLander.set_rng_tape().

Principe
--------
Gymnasium tire ses nombres aléatoires via deux méthodes de
np.random.Generator :

    np_random.uniform(low, high, size=...)
    np_random.integers(low, high, size=...)   # wind_idx / torque_idx

Les deux sont des transformations affines d'un tirage brut u dans [0,1).
On peut donc, à partir de la valeur RÉELLEMENT tirée par la référence,
remonter à ce u par inversion générique :

    u = (valeur - low) / (high - low)

Côté C, ll_randf()/ll_randrange() utilisent exactement la même
transformation (voir lunar_lander.h), donc injecter cette tape de u
reproduit fidèlement les valeurs réelles de la référence, sans avoir à
réimplémenter l'algorithme RNG de NumPy (PCG64) en C.

Ordre des tirages consommés par la référence (vérifié dans le code
source de Gymnasium) :

  reset(seed) :
    1.  uniform(0, H/2, size=12)              -> terrain (12 valeurs)
    2.  uniform(-1000, 1000)                  -> force initiale x
    3.  uniform(-1000, 1000)                  -> force initiale y
    4.  [si enable_wind] integers(-9999,9999) -> wind_idx
    5.  [si enable_wind] integers(-9999,9999) -> torque_idx
    6.  uniform(-1, 1)                        -> dispersion[0] (step(0) interne)
    7.  uniform(-1, 1)                        -> dispersion[1] (step(0) interne)

  step(action), à chaque appel :
    1.  uniform(-1, 1) -> dispersion[0]
    2.  uniform(-1, 1) -> dispersion[1]

Cet ordre doit rester strictement synchronisé avec celui consommé côté
C (lunar_lander.h / c_reset / c_step) : un tirage manqué ou ajouté
décale tout le reste de l'épisode.
"""

import numpy as np
from gymnasium.utils import seeding


class RecordingGenerator:
    """
    Enveloppe un np.random.Generator pour enregistrer, dans l'ordre,
    la version "brute dans [0,1)" de chaque tirage effectué.
    """

    def __init__(self, gen: np.random.Generator):
        self._gen = gen
        self.tape: list[float] = []

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
    silencieusement le recorder par un nouveau générateur -- c'est
    exactement le bug qui produisait une tape vide (len(tape) == 0).
    """
    gen, np_seed = seeding.np_random(seed)
    recorder = RecordingGenerator(gen)
    # Assignation directe à l'attribut privé : évite toute dépendance au
    # comportement exact du setter de la propriété np_random selon la
    # version de Gymnasium installée.
    env.unwrapped._np_random = recorder
    env.unwrapped._np_random_seed = np_seed
    return recorder


def record_reference_tape(env, seed: int, actions: list[int]):
    """
    Rejoue `actions` sur l'environnement Gymnasium `env` (déjà créé,
    PAS encore reseté), en enregistrant la tape RNG complète depuis le
    tout premier tirage, et renvoie (tape, observations, rewards,
    terminations) pour comparaison.

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

    # seed=None ici, sinon Gymnasium re-seede et écrase le recorder
    # (voir _wrap_np_random_with_recorder).
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
            # La référence ne se relance pas automatiquement comme le port C ;
            # si vous voulez poursuivre sur un nouvel épisode, ré-enveloppez
            # np_random après ce reset (le wrapper est perdu sinon, car ici
            # encore on appellerait reset(seed=None), donc en fait il N'EST
            # PAS perdu -- mais le flux logique de l'épisode s'arrête ici).
            break

    tape = np.array(recorder.tape, dtype=np.float32)
    return tape, obs, rews, dones


# ─────────────────────────────────────────────────────────────────────
# Exemple d'intégration dans test_conformity_LL.py
# ─────────────────────────────────────────────────────────────────────
"""
Remplacement suggéré pour compare_sequence() dans test_conformity_LL.py :

    from rng_tape_recorder import record_reference_tape

    def compare_sequence(actions, label, seed=0, ...):
        ref = make_reference(seed=seed)

        # 1. enregistrer la tape réelle de la référence sur CETTE séquence
        tape, ref_obs, ref_rews, ref_dones = record_reference_tape(
            ref, seed, actions
        )

        # 2. créer le port et injecter la même tape avant de la rejouer
        port = make_port(seed=seed)
        port.set_weights(np.array([1.0, 0.0], dtype=np.float32))
        port.set_rng_tape(tape)
        port.reset(seed=seed)

        # 3. comparer port_step(...) à ref_obs/ref_rews/ref_dones déjà
        #    enregistrés, au lieu de re-stepper ref (déjà fait à l'étape 1)
        ...

Important : comme record_reference_tape() rejoue déjà toute la séquence
sur `ref` pour produire la tape, il ne faut PAS rappeler ref.step(...)
une deuxième fois dans la boucle de comparaison — utilisez les valeurs
déjà capturées dans ref_obs / ref_rews / ref_dones.
"""