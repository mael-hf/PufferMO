#include "overcooked.h"
#include "../env_binding.h"

void my_init(COvercooked* env) {
    // Configuration de base pour l'environnement
    env->width = 7;
    env->height = 7;
    env->num_pots = 2;
    env->max_ticks = 500;
    
    // Allocation de la mémoire structurée dans overcooked.h
    allocate_overcooked(env);
}

void my_log(COvercooked* env, Log* log) {
    // Mapping direct des logs scalaires (Pas de MY_PUT)
    log->episode_return = env->log.episode_return;
    log->soupes_livrees = env->log.soupes_livrees;
    log->episode_length = env->log.episode_length;
}