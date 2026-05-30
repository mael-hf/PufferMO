#ifndef OVERCOOKED_H
#define OVERCOOKED_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// Inclusion obligatoire pour le binding PufferLib
#include "../env_binding.h"

// CONSTANTES DE L'ENVIRONNEMENT 
#define NUM_AGENTS 2

// Actions
#define ACT_UP 0
#define ACT_DOWN 1
#define ACT_LEFT 2
#define ACT_RIGHT 3
#define ACT_INTERACT 4

// Éléments tenus (Held items)
#define ITEM_NONE 0
#define ITEM_ONION 1
#define ITEM_PLATE 2
#define ITEM_SOUP 3

// Éléments de la grille (Grid tiles)
#define TILE_EMPTY 0
#define TILE_WALL 1
#define TILE_ONION_DISP 2
#define TILE_PLATE_DISP 3
#define TILE_POT 4
#define TILE_SERVE 5

// STRUCTURES 

typedef struct Log {
    float episode_return;
    float soupes_livrees;
    float episode_length;
    float n; // Obligatoire à la fin pour le calcul des moyennes
} Log;

typedef struct Agent {
    int x; // Colonne
    int y; // Ligne
    int dir; // Direction (0=Haut, 1=Bas, 2=Gauche, 3=Droite)
    int held_item;
} Agent;

typedef struct Pot {
    int x;
    int y;
    int num_onions;
    int time_remaining;
    bool is_cooking;
    bool is_ready;
} Pot;

typedef struct COvercooked {
    // API Parallèle
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    bool* alive;
    
    // Logs
    Log log;
    Log agent_logs[NUM_AGENTS];
    
    // État (Mémoire de l'environnement)
    int* grid;
    Agent agents[NUM_AGENTS];
    Pot* pots;
    
    // Paramètres
    int num_agents_per_env;
    int num_pots;
    int width;
    int height;
    int obs_size;
    
    // Gestion du temps
    int tick;
    int max_ticks;
    bool done;
} COvercooked;

// FONCTIONS DE GESTION DES LOGS

void add_log(COvercooked* env, int agent_id) {
    env->log.episode_return += env->agent_logs[agent_id].episode_return;
    env->log.soupes_livrees += env->agent_logs[agent_id].soupes_livrees;
    env->log.episode_length += env->agent_logs[agent_id].episode_length;
    env->log.n += 1.0f;
}

// INITIALISATION ET ALLOCATION

void init_overcooked(COvercooked* env) {
    env->num_agents_per_env = NUM_AGENTS;
    env->grid = (int*)calloc(env->width * env->height, sizeof(int));
    env->pots = (Pot*)calloc(env->num_pots, sizeof(Pot));
    env->tick = 0;
    env->done = false;
}

void allocate_overcooked(COvercooked* env) {
    // Taille basique de l'observation : Grille complète + état agents + état pots
    env->obs_size = (env->width * env->height) + (NUM_AGENTS * 4) + (env->num_pots * 4);
    
    env->observations = (float*)calloc(NUM_AGENTS * env->obs_size, sizeof(float));
    env->actions = (int*)calloc(NUM_AGENTS, sizeof(int));
    env->rewards = (float*)calloc(NUM_AGENTS, sizeof(float));
    env->terminals = (unsigned char*)calloc(NUM_AGENTS, sizeof(unsigned char));
    env->alive = (bool*)calloc(NUM_AGENTS, sizeof(bool));
    
    init_overcooked(env);
}

void free_overcooked(COvercooked* env) {
    free(env->grid);
    free(env->pots);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->alive);
}

//  OBSERVATIONS 

void compute_observations(COvercooked* env) {
    // Chaque agent reçoit la même observation globale pour cet exemple
    for (int i = 0; i < NUM_AGENTS; i++) {
        int offset = i * env->obs_size;
        int ptr = 0;
        
        // 1. Grille
        for(int j = 0; j < env->width * env->height; j++) {
            env->observations[offset + ptr++] = (float)env->grid[j];
        }
        
        // 2. Agents
        for(int a = 0; a < NUM_AGENTS; a++) {
            env->observations[offset + ptr++] = (float)env->agents[a].x;
            env->observations[offset + ptr++] = (float)env->agents[a].y;
            env->observations[offset + ptr++] = (float)env->agents[a].dir;
            env->observations[offset + ptr++] = (float)env->agents[a].held_item;
        }
        
        // 3. Pots
        for(int p = 0; p < env->num_pots; p++) {
            env->observations[offset + ptr++] = (float)env->pots[p].num_onions;
            env->observations[offset + ptr++] = (float)env->pots[p].time_remaining;
            env->observations[offset + ptr++] = (float)env->pots[p].is_cooking;
            env->observations[offset + ptr++] = (float)env->pots[p].is_ready;
        }
    }
}

// LOGIQUE DE JEU

void c_reset(COvercooked* env) {
    env->tick = 0;
    env->done = false;
    env->log = (Log){0};
    
    for (int i = 0; i < NUM_AGENTS; i++) {
        env->agent_logs[i] = (Log){0};
        env->alive[i] = true;
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
        env->agents[i].held_item = ITEM_NONE;
        env->agents[i].dir = ACT_DOWN;
    }
    
    // (À personnaliser) Placement arbitraire des agents pour l'exemple
    env->agents[0].x = 1; env->agents[0].y = 1;
    env->agents[1].x = env->width - 2; env->agents[1].y = env->height - 2;

    for(int p = 0; p < env->num_pots; p++) {
        env->pots[p].num_onions = 0;
        env->pots[p].time_remaining = 0;
        env->pots[p].is_cooking = false;
        env->pots[p].is_ready = false;
    }
    
    compute_observations(env);
}

void c_step(COvercooked* env) {
    if (env->done) return;
    
    env->tick++;
    memset(env->rewards, 0, NUM_AGENTS * sizeof(float));
    memset(env->terminals, 0, NUM_AGENTS * sizeof(unsigned char));

    int next_x[NUM_AGENTS];
    int next_y[NUM_AGENTS];

    //  PHASE 1 : INTENTIONS DE DÉPLACEMENT 
    for(int i = 0; i < NUM_AGENTS; i++) {
        env->agent_logs[i].episode_length += 1.0f;
        next_x[i] = env->agents[i].x;
        next_y[i] = env->agents[i].y;
        
        int act = env->actions[i];
        if (act == ACT_UP)    { next_y[i]--; env->agents[i].dir = act; }
        else if (act == ACT_DOWN)  { next_y[i]++; env->agents[i].dir = act; }
        else if (act == ACT_LEFT)  { next_x[i]--; env->agents[i].dir = act; }
        else if (act == ACT_RIGHT) { next_x[i]++; env->agents[i].dir = act; }
        
        // Collision avec le décor
        if (next_x[i] < 0 || next_x[i] >= env->width || 
            next_y[i] < 0 || next_y[i] >= env->height || 
            env->grid[next_y[i] * env->width + next_x[i]] != TILE_EMPTY) {
            next_x[i] = env->agents[i].x;
            next_y[i] = env->agents[i].y;
        }
    }

    //  PHASE 2 : COLLISIONS MULTI-AGENTS PARALLÈLES 
    // Collision frontale (swap)
    if (next_x[0] == env->agents[1].x && next_y[0] == env->agents[1].y &&
        next_x[1] == env->agents[0].x && next_y[1] == env->agents[0].y) {
        next_x[0] = env->agents[0].x; next_y[0] = env->agents[0].y;
        next_x[1] = env->agents[1].x; next_y[1] = env->agents[1].y;
    }
    // Collision sur la même case
    if (next_x[0] == next_x[1] && next_y[0] == next_y[1]) {
        next_x[0] = env->agents[0].x; next_y[0] = env->agents[0].y;
        next_x[1] = env->agents[1].x; next_y[1] = env->agents[1].y;
    }

    // Application des mouvements
    for(int i = 0; i < NUM_AGENTS; i++) {
        env->agents[i].x = next_x[i];
        env->agents[i].y = next_y[i];
    }

    //  PHASE 3 : INTERACTIONS 
    for(int i = 0; i < NUM_AGENTS; i++) {
        if(env->actions[i] == ACT_INTERACT) {
            int target_x = env->agents[i].x;
            int target_y = env->agents[i].y;
            
            // Calcul de la case ciblée
            if (env->agents[i].dir == ACT_UP) target_y--;
            else if (env->agents[i].dir == ACT_DOWN) target_y++;
            else if (env->agents[i].dir == ACT_LEFT) target_x--;
            else if (env->agents[i].dir == ACT_RIGHT) target_x++;
            
            if (target_x >= 0 && target_x < env->width && target_y >= 0 && target_y < env->height) {
                int tile = env->grid[target_y * env->width + target_x];
                
                // Distributeur d'oignons
                if (tile == TILE_ONION_DISP && env->agents[i].held_item == ITEM_NONE) {
                    env->agents[i].held_item = ITEM_ONION;
                }
                // Distributeur d'assiettes
                else if (tile == TILE_PLATE_DISP && env->agents[i].held_item == ITEM_NONE) {
                    env->agents[i].held_item = ITEM_PLATE;
                }
                // Marmite
                else if (tile == TILE_POT) {
                    Pot* current_pot = NULL;
                    for (int p = 0; p < env->num_pots; p++) {
                        if (env->pots[p].x == target_x && env->pots[p].y == target_y) {
                            current_pot = &env->pots[p];
                            break;
                        }
                    }
                    
                    if (current_pot) {
                        // Ajouter un oignon
                        if (env->agents[i].held_item == ITEM_ONION && current_pot->num_onions < 3 
                            && !current_pot->is_cooking && !current_pot->is_ready) {
                            current_pot->num_onions++;
                            env->agents[i].held_item = ITEM_NONE;
                            
                            // Lancement de la cuisson automatique à 3 oignons
                            if (current_pot->num_onions == 3) {
                                current_pot->is_cooking = true;
                                current_pot->time_remaining = 20; // 20 ticks pour cuire
                            }
                        }
                        // Récupérer la soupe avec une assiette
                        else if (env->agents[i].held_item == ITEM_PLATE && current_pot->is_ready) {
                            env->agents[i].held_item = ITEM_SOUP;
                            current_pot->num_onions = 0;
                            current_pot->is_ready = false;
                        }
                    }
                }
                // Zone de livraison
                else if (tile == TILE_SERVE && env->agents[i].held_item == ITEM_SOUP) {
                    env->agents[i].held_item = ITEM_NONE;
                    env->rewards[0] += 20.0f; // Récompense partagée
                    env->rewards[1] += 20.0f; // Récompense partagée
                    env->agent_logs[0].soupes_livrees += 1.0f;
                    env->agent_logs[1].soupes_livrees += 1.0f;
                }
            }
        }
    }

    //  PHASE 4 : MISE À JOUR DES MARMITES 
    for(int p = 0; p < env->num_pots; p++) {
        if (env->pots[p].is_cooking) {
            env->pots[p].time_remaining--;
            if (env->pots[p].time_remaining <= 0) {
                env->pots[p].is_cooking = false;
                env->pots[p].is_ready = true;
            }
        }
    }

    //  PHASE 5 : TRONCATURE ET FIN D'ÉPISODE 
    for(int i = 0; i < NUM_AGENTS; i++) {
        env->agent_logs[i].episode_return += env->rewards[i];
    }

    if (env->tick >= env->max_ticks) {
        for(int i = 0; i < NUM_AGENTS; i++) {
            add_log(env, i);
            env->terminals[i] = 1;
        }
        env->done = true;
        c_reset(env);
    }

    compute_observations(env);
}

#endif // OVERCOOKED_H