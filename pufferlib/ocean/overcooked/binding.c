#include "overcooked.h"
#include "../env_binding.h"

void my_init(Overcooked* env) {
    // 1. Initialisation par défaut basée sur l'officiel
    env->layout_id = LAYOUT_CRAMPED_ROOM; // Changeable via Python si besoin
    env->num_agents = NUM_AGENTS;
    env->observation_size = 43; // La taille exacte du vecteur d'obs
    env->max_ticks = 400; // Horizon de 400 imposé par l'article de recherche
    
    // Reward shaping officiel
    env->rewards_config.dish_served_whole_team = 20.0f; // 20 points pour la soupe servie
    env->rewards_config.dish_served_agent = 0.0f;
    env->rewards_config.pot_started = 0.15f;
    env->rewards_config.ingredient_added = 0.15f;
    env->rewards_config.ingredient_picked = 0.05f;
    env->rewards_config.plate_picked = 0.05f;
    env->rewards_config.soup_plated = 0.20f;
    env->rewards_config.wrong_dish_served = 0.0f;
    env->rewards_config.step_penalty = 0.0f;

    // 2. Initialisation de la mémoire statique de l'environnement
    init(env);
    
    // PufferLib allouera observations, actions, rewards, terminals, log 
    // automatiquement.
}

void my_log(Overcooked* env, Log* log) {
    // Mapping direct (pas de multi-objectif)
    log->episode_return = env->log.episode_return;
    log->dishes_served = env->log.dishes_served;
    log->correct_dishes = env->log.correct_dishes;
    log->agent_collisions = env->log.agent_collisions;
}