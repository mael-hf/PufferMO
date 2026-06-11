#include "deep_sea_treasure.h"

#define Env DeepSeaTreasure
#define MY_PUT
#include "../env_binding_mo.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->gamma = unpack(kwargs, "gamma");
    env->manual_weights = false;
    env->client = NULL;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict,"perf", log->perf);
    assign_to_dict(dict,"score", log->score);
    assign_to_dict(dict,"episode_return", log->episode_return);
    assign_to_dict(dict,"episode_return_treasure", log->episode_return_treasure);
    assign_to_dict(dict,"episode_return_time", log->episode_return_time);
    assign_to_dict(dict,"discounted_episode_return", log->discounted_episode_return);
    assign_to_dict(dict,"discounted_episode_return_treasure", log->discounted_episode_return_treasure);
    assign_to_dict(dict,"discounted_episode_return_time", log->discounted_episode_return_time);
    assign_to_dict(dict,"scalarized_episode_return", log->scalarized_episode_return);
    assign_to_dict(dict,"discounted_scalarized_episode_return", log->discounted_scalarized_episode_return);
    assign_to_dict(dict,"weight_treasure", log->weight_treasure);
    assign_to_dict(dict,"weight_time", log->weight_time);
    assign_to_dict(dict,"episode_length", log->episode_length);
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
        PyErr_SetString(PyExc_ValueError, "weights must be a 1D array with 2 elements");
        return 1;
    }

    float* weights_data = (float*)PyArray_DATA(weights_array);
    for (int i = 0; i < REWARD_DIM; i++) {
        env->weights[i] = weights_data[i];
    }
    env->manual_weights = true;
    return 0;
}