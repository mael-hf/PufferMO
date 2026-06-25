#include "pommerman.h"

#define Env Pommerman
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->game_mode = (int)unpack(kwargs, "game_mode");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "kills",          log->kills);
    assign_to_dict(dict, "deaths",         log->deaths);
    assign_to_dict(dict, "wins",           log->wins);
    assign_to_dict(dict, "score",          log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
