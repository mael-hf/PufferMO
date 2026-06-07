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
 * ── my_put: validate and copy a manual weights array ──
 *
 * Called by vec_put (Python: env.set_weights(weights)).
 * Expects a numpy float32 array of length REWARD_DIM.
 * Sets manual_weights=1 so c_reset does not overwrite them.
 */
static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    /* Weights are already in env->weights via the shared numpy buffer.
       Just mark them as manually set so c_reset won't re-sample. */
    env->manual_weights = 1;
    return 0;
}