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
    env->rng_exhaust_warned = 0;
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
 * ── my_put: reçoit les données via les kwargs de vec_put ──
 *
 * IMPORTANT : la fonction générique vec_put() (env_binding_mo.h) ne
 * transmet JAMAIS l'array via les arguments positionnels -- elle
 * n'accepte qu'1 argument positionnel (le handle c_envs) et appelle
 * ensuite my_put(env, empty_args, kwargs) avec un `args` TOUJOURS VIDE
 * pour chaque env du vecteur. Toute donnée doit donc passer par des
 * arguments nommés (kwargs), jamais par args.
 *
 * Côté Python (Lunar_Lander.py) :
 *   binding.vec_put(self.c_envs, weights=weights_array)
 *   binding.vec_put(self.c_envs, rng_tape=tape_array)
 *
 * Note : vec_put boucle sur tous les envs du vecteur et leur transmet
 * le MÊME dict kwargs à chacun -- donc le même tableau est appliqué à
 * tous les envs. Pour set_rng_tape(), c'est volontaire et documenté
 * (pensé pour num_envs=1, cf. Lunar_Lander.py).
 */
static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    if (kwargs == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "put: expected a 'weights' or 'rng_tape' keyword argument");
        return -1;
    }

    int handled = 0;

    /* ── chemin 'weights' : poids de scalarisation ── */
    PyObject* w_obj = PyDict_GetItemString(kwargs, "weights");
    if (w_obj != NULL) {
        Py_buffer view;
        if (PyObject_GetBuffer(w_obj, &view, PyBUF_SIMPLE | PyBUF_FORMAT) < 0)
            return -1;

        if (view.len != REWARD_DIM * (Py_ssize_t)sizeof(float)) {
            PyErr_Format(PyExc_ValueError,
                "weights: expected float32 array of length %d, got %zd bytes",
                REWARD_DIM, view.len);
            PyBuffer_Release(&view);
            return -1;
        }
        float* w = (float*)view.buf;
        for (int i = 0; i < REWARD_DIM; i++)
            env->weights[i] = w[i];
        env->manual_weights = 1;

        PyBuffer_Release(&view);
        handled = 1;
    }

    /* ── chemin 'rng_tape' : tape RNG injectée ── */
    PyObject* tape_obj = PyDict_GetItemString(kwargs, "rng_tape");
    if (tape_obj != NULL) {
        Py_buffer view;
        if (PyObject_GetBuffer(tape_obj, &view, PyBUF_SIMPLE | PyBUF_FORMAT) < 0)
            return -1;

        Py_ssize_t n_floats = view.len / (Py_ssize_t)sizeof(float);

        if (env->injected_rng) {
            free(env->injected_rng);
            env->injected_rng = NULL;
        }
        env->injected_rng = (float*)malloc((size_t)n_floats * sizeof(float));
        if (!env->injected_rng) {
            PyErr_SetString(PyExc_MemoryError, "rng_tape: allocation failed");
            PyBuffer_Release(&view);
            return -1;
        }
        memcpy(env->injected_rng, view.buf, (size_t)n_floats * sizeof(float));
        env->injected_len     = (int)n_floats;
        env->injected_idx     = 0;
        env->use_injected_rng = 1;
        env->rng_exhaust_warned = 0;

        PyBuffer_Release(&view);
        handled = 1;
    }

    if (!handled) {
        PyErr_SetString(PyExc_ValueError,
            "put: expected a 'weights' or 'rng_tape' keyword argument");
        return -1;
    }

    return 0;
}