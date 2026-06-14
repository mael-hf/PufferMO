#include "minecart.h"

#define Env Minecart
#define MY_PUT
#include "../env_binding_mo.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->frame_skip = (int)unpack(kwargs, "frame_skip");
    env->incremental_frame_skip = (int)unpack(kwargs, "incremental_frame_skip");
    env->capacity = (float)unpack(kwargs, "capacity");
    env->mine_cnt = (int)unpack(kwargs, "mine_cnt");
    env->gamma = unpack(kwargs, "gamma");
    env->max_ticks = (int)unpack(kwargs, "max_ticks");
    env->max_ticks_offset_mod = (int)unpack(kwargs, "max_ticks_offset_mod");
    env->manual_weights = false;
    env->client = NULL;
    env->mines = (Mine*)calloc(env->mine_cnt, sizeof(Mine));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_return_ore0", log->episode_return_ore0);
    assign_to_dict(dict, "episode_return_ore1", log->episode_return_ore1);
    assign_to_dict(dict, "episode_return_fuel", log->episode_return_fuel);
    assign_to_dict(dict, "discounted_episode_return", log->discounted_episode_return);
    assign_to_dict(dict, "discounted_episode_return_ore0", log->discounted_episode_return_ore0);
    assign_to_dict(dict, "discounted_episode_return_ore1", log->discounted_episode_return_ore1);
    assign_to_dict(dict, "discounted_episode_return_fuel", log->discounted_episode_return_fuel);
    assign_to_dict(dict, "scalarized_episode_return", log->scalarized_episode_return);
    assign_to_dict(dict, "discounted_scalarized_episode_return", log->discounted_scalarized_episode_return);
    assign_to_dict(dict, "weight_ore0", log->weight_ore0);
    assign_to_dict(dict, "weight_ore1", log->weight_ore1);
    assign_to_dict(dict, "weight_fuel", log->weight_fuel);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* weights_obj = PyDict_GetItemString(kwargs, "weights");
    if (weights_obj == NULL) {
        PyErr_SetString(PyExc_KeyError, "Key 'weights' not found in kwargs");
        return 1;
    }
    if (!PyObject_TypeCheck(weights_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "weights must be a NumPy array");
        return 1;
    }

    PyArrayObject* weights_array = (PyArrayObject*)weights_obj;
    if (!PyArray_ISCONTIGUOUS(weights_array)) {
        PyErr_SetString(PyExc_ValueError, "weights must be contiguous");
        return 1;
    }

    npy_intp* dims = PyArray_DIMS(weights_array);
    if (PyArray_NDIM(weights_array) != 1 || dims[0] != REWARD_DIM) {
        PyErr_SetString(PyExc_ValueError, "weights must be a 1D array with 3 elements");
        return 1;
    }

    float* weights_data = (float*)PyArray_DATA(weights_array);
    for (int i = 0; i < REWARD_DIM; i++) {
        env->weights[i] = weights_data[i];
    }
    env->manual_weights = true;
    return 0;
}
