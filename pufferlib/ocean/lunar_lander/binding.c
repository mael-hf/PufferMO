#define Env LunarLander
#define MY_PUT
#define BINDING_C_LOADED       // bloque l'include depuis lunar_lander.h

#include "lunar_lander.h"      // types complets, env_binding_mo.h bloqué
#include "../env_binding_mo.h" // inclus ici, tous les types sont complets

/* ── my_init: parse kwargs into the env struct ── */
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->enable_wind      = (int)unpack(kwargs, "enable_wind");
    env->wind_power       = (float)unpack(kwargs, "wind_power");
    env->turbulence_power = (float)unpack(kwargs, "turbulence_power");
    env->manual_weights   = 0;

    /* État RNG injecté — désactivé par défaut, activé via set_rng_tape() */
    env->injected_rng     = NULL;
    env->injected_len     = 0;
    env->injected_idx     = 0;
    env->use_injected_rng = 0;
    return 0;
}

/* ── my_log: map each Log field to the returned dict ── */
static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score_landing", log->score_landing);
    assign_to_dict(dict, "score_fuel",    log->score_fuel);
    assign_to_dict(dict, "episode_len",   log->episode_len);
    return 0;
}

/*
 * ── my_put: validate and copy an array sent from Python ──
 *
 * Appelé par vec_put (Python: env.set_weights(...) ou env.set_rng_tape(...)).
 * Reçoit un tableau numpy float32 de longueur variable. La longueur du
 * buffer détermine son interprétation :
 *
 *   - longueur == REWARD_DIM (2)  -> poids de scalarisation (comportement
 *                                     original, set_weights)
 *   - longueur >  REWARD_DIM      -> tape de tirages RNG bruts dans [0,1),
 *                                     enregistrée depuis la référence
 *                                     Gymnasium (set_rng_tape, voir
 *                                     rng_tape_recorder.py)
 *
 * On réutilise volontairement le hook déjà câblé par env_binding_mo.h
 * (vec_put) plutôt que d'en créer un nouveau, dont la signature côté
 * binding générique n'est pas visible depuis ce fichier.
 */
static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        PyErr_SetString(PyExc_ValueError, "expected a single array argument");
        return -1;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE | PyBUF_FORMAT) < 0)
        return -1;

    Py_ssize_t n_floats = view.len / (Py_ssize_t)sizeof(float);

    if (n_floats == REWARD_DIM) {
        /* ── chemin existant : poids de scalarisation ── */
        float* w = (float*)view.buf;
        for (int i = 0; i < REWARD_DIM; i++)
            env->weights[i] = w[i];
        env->manual_weights = 1;

    } else if (n_floats > REWARD_DIM) {
        /* ── nouveau chemin : tape RNG injectée ── */
        if (env->injected_rng) {
            free(env->injected_rng);
            env->injected_rng = NULL;
        }
        env->injected_rng = (float*)malloc((size_t)n_floats * sizeof(float));
        if (!env->injected_rng) {
            PyErr_SetString(PyExc_MemoryError, "set_rng_tape: allocation failed");
            PyBuffer_Release(&view);
            return -1;
        }
        memcpy(env->injected_rng, view.buf, (size_t)n_floats * sizeof(float));
        env->injected_len     = (int)n_floats;
        env->injected_idx     = 0;
        env->use_injected_rng = 1;

    } else {
        PyErr_Format(PyExc_ValueError,
            "put: expected float32 array of length >= %d, got %zd floats",
            REWARD_DIM, n_floats);
        PyBuffer_Release(&view);
        return -1;
    }

    PyBuffer_Release(&view);
    return 0;
}
