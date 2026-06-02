#include "lunar_lander.h"

#define Env LunarLander
#define MY_PUT
#include "../env_binding_mo.h"

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
    PyObject* w_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &w_obj)) {
        PyErr_SetString(PyExc_ValueError, "set_weights: expected a single array argument");
        return -1;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(w_obj, &view, PyBUF_SIMPLE | PyBUF_FORMAT) < 0)
        return -1;

    if (view.len != REWARD_DIM * (Py_ssize_t)sizeof(float)) {
        PyErr_Format(PyExc_ValueError,
            "set_weights: expected float32 array of length %d, got %zd bytes",
            REWARD_DIM, view.len);
        PyBuffer_Release(&view);
        return -1;
    }

    float* w = (float*)view.buf;
    for (int i = 0; i < REWARD_DIM; i++)
        env->weights[i] = w[i];

    env->manual_weights = 1;
    PyBuffer_Release(&view);
    return 0;
}