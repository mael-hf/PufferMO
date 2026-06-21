#define Env LunarLander
#define MY_PUT
#define BINDING_C_LOADED       

#include "lunar_lander.h"      
#include "../env_binding_mo.h" 

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->enable_wind      = (int)unpack(kwargs, "enable_wind");
    env->wind_power       = (float)unpack(kwargs, "wind_power");
    env->turbulence_power = (float)unpack(kwargs, "turbulence_power");
    env->manual_weights   = 0;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score_landing", log->score_landing);
    assign_to_dict(dict, "score_fuel",    log->score_fuel);
    assign_to_dict(dict, "episode_len",   log->episode_len);
    return 0;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    env->manual_weights = 1;
    return 0;
}