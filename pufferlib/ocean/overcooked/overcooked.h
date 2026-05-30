#ifndef OVERCOOKED_H
#define OVERCOOKED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "raylib.h"
#include "../env_binding.h"

// --- CONSTANTES ---
#define EMPTY 0
#define COUNTER 1
#define STOVE 2
#define CUTTING_BOARD 3
#define INGREDIENT_BOX 4
#define SERVING_AREA 5
#define WALL 6
#define PLATE_BOX 7
#define AGENT 8

#define NO_ITEM 10
#define TOMATO 11
#define ONION 12
#define PLATE 13
#define SOUP 14
#define PLATED_SOUP 15

#define NOT_COOKING 0
#define COOKING 1
#define COOKED 2

#define COOKING_TIME 20
#define MAX_INGREDIENTS 3

#define ACTION_NOOP 0
#define ACTION_UP 1
#define ACTION_DOWN 2
#define ACTION_LEFT 3
#define ACTION_RIGHT 4
#define ACTION_INTERACT 5

#define AGENT_EMPTY_HANDED 0
#define AGENT_HOLDING_ITEM 1
#define MAX_SPAWN_POSITIONS 8

// --- STRUCTURES ---
typedef enum {
    LAYOUT_CRAMPED_ROOM = 0,
    LAYOUT_ASYMMETRIC_ADVANTAGES = 1,
    LAYOUT_FORCED_COORDINATION = 2,
    LAYOUT_COORDINATION_RING = 3,
    LAYOUT_COUNTER_CIRCUIT = 4,
    LAYOUT_COUNT
} LayoutType;

typedef struct {
    const char* name;
    int width;
    int height;
    const char* grid;
    int spawn_positions[MAX_SPAWN_POSITIONS];
    int num_spawns;
} LayoutInfo;

typedef struct {
    float dish_served_whole_team;
    float dish_served_agent;
    float pot_started;
    float ingredient_added;
    float ingredient_picked;
    float plate_picked;
    float soup_plated;
    float wrong_dish_served;
    float step_penalty;
} RewardConfig;

typedef struct {
    float perf; 
    float score; 
    float episode_return; 
    float episode_length; 
    float dishes_served; 
    float correct_dishes; 
    float wrong_dishes; 
    float ingredients_picked; 
    float pots_started; 
    float items_dropped; 
    float agent_collisions; 
    float n; // Requis à la fin
} Log;

typedef struct {
    Texture2D floor; Texture2D counter; Texture2D pot; Texture2D serve;
    Texture2D onions_box; Texture2D tomatoes_box; Texture2D dishes_box; Texture2D wall;
    Texture2D onion; Texture2D tomato; Texture2D dish; Texture2D soup_onion; Texture2D soup_tomato;
    Texture2D soup_onion_cooking_1; Texture2D soup_onion_cooking_2; Texture2D soup_onion_cooking_3; Texture2D soup_onion_cooked;
    Texture2D soup_tomato_cooking_1; Texture2D soup_tomato_cooking_2; Texture2D soup_tomato_cooking_3; Texture2D soup_tomato_cooked;
    Texture2D chef_north; Texture2D chef_south; Texture2D chef_east; Texture2D chef_west;
    Texture2D chef_north_onion; Texture2D chef_south_onion; Texture2D chef_east_onion; Texture2D chef_west_onion;
    Texture2D chef_north_tomato; Texture2D chef_south_tomato; Texture2D chef_east_tomato; Texture2D chef_west_tomato;
    Texture2D chef_north_dish; Texture2D chef_south_dish; Texture2D chef_east_dish; Texture2D chef_west_dish;
    Texture2D chef_north_soup_onion; Texture2D chef_south_soup_onion; Texture2D chef_east_soup_onion; Texture2D chef_west_soup_onion;
    Texture2D chef_north_soup_tomato; Texture2D chef_south_soup_tomato; Texture2D chef_east_soup_tomato; Texture2D chef_west_soup_tomato;
    Texture2D soup_onion_dish; Texture2D soup_tomato_dish;
} Client;

typedef struct __attribute__((aligned(32))) {
    float x;
    float y;
    int facing_direction;
    int held_item;
    int held_soup_onions;
    int held_soup_tomatoes;
    int held_soup_total;
    int ticks_since_reward;
} Agent;

typedef struct __attribute__((aligned(32))) {
    int x; int y; int type; int state;
    int num_onions; int num_tomatoes; int total_ingredients;
} Item;

typedef struct {
    int cooking_state;
    int cooking_progress;
    int ingredient_types[MAX_INGREDIENTS];
    int ingredient_count;
    int num_onions;
    int num_tomatoes;
} CookingPot;

typedef struct {
    int ingredient_box_positions[20]; int ingredient_box_count;
    int plate_box_positions[20]; int plate_box_count;
    int serving_area_positions[20]; int serving_area_count;
    int stove_positions[20]; int stove_count;
    int counter_positions[100]; int counter_count;
    float inv_width; float inv_height;
} StaticCache;

typedef struct {
    Log log;
    Client* client;
    LayoutType layout_id;
    char* grid;
    Item* items;
    int num_items;
    int max_items;
    Agent* agents;
    int num_agents;
    uint64_t agent_position_mask;
    CookingPot* cooking_pots;
    int num_stoves;
    int* pot_index_grid;
    int* item_grid;
    float* observations;
    int* actions; // int* pour action discrete
    float* rewards;
    unsigned char* terminals; // char pour compatibilité Puffer
    int width;
    int height;
    int grid_size;
    RewardConfig rewards_config;
    int observation_size;
    StaticCache cache;
    unsigned int rng;
    int tick; // Ajout gestion temps
    int max_ticks; 
} Overcooked;

// --- LAYOUTS (Extraits) ---
static const char CRAMPED_ROOM[5][5] = {
    {'6', '1', '2', '1', '6'},
    {'4', ' ', ' ', ' ', '4'},
    {'1', ' ', ' ', ' ', '1'},
    {'1', ' ', ' ', ' ', '1'},
    {'6', '7', '1', '5', '6'}
};
static const LayoutInfo LAYOUTS[LAYOUT_COUNT] = {
    { "cramped_room", 5, 5, (const char*)CRAMPED_ROOM, {1, 2, 3, 2}, 2 },
    // Les autres layouts sont tronqués ici pour la lisibilité, mais tu peux copier-coller 
    // l'intégralité du tableau LAYOUTS de l'officiel ici.
};

static inline const LayoutInfo* get_layout_info(LayoutType id) { return &LAYOUTS[id]; }
static inline char get_layout_tile(const LayoutInfo* info, int x, int y) { return info->grid[y * info->width + x]; }

// --- LOGIQUE ITEMS ---
static inline Item* get_item_at(Overcooked* env, int x, int y) {
    int idx = env->item_grid[y * env->width + x];
    return (idx >= 0) ? &env->items[idx] : NULL;
}

static void add_item(Overcooked* env, int type, int x, int y) {
    if (env->num_items < env->max_items) {
        int idx = env->num_items;
        env->items[idx].type = type;
        env->items[idx].x = x; env->items[idx].y = y;
        env->items[idx].state = 0; env->items[idx].num_onions = 0;
        env->items[idx].num_tomatoes = 0; env->items[idx].total_ingredients = 0;
        env->item_grid[y * env->width + x] = idx;
        env->num_items++;
    }
}

static void remove_item(Overcooked* env, int x, int y) {
    int idx = env->item_grid[y * env->width + x];
    if (idx < 0) return;
    env->item_grid[y * env->width + x] = -1;
    if (idx < env->num_items - 1) {
        Item* last = &env->items[env->num_items - 1];
        env->items[idx] = *last;
        env->item_grid[last->y * env->width + last->x] = idx;
    }
    env->num_items--;
}

static void init_cooking_pots(Overcooked* env) {
    env->num_stoves = 0;
    for (int i = 0; i < env->width * env->height; i++) {
        if (env->grid[i] == STOVE) env->num_stoves++;
    }
    env->cooking_pots = calloc(env->num_stoves, sizeof(CookingPot));
    int pot_index = 0;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            if (env->grid[y * env->width + x] == STOVE) {
                CookingPot* pot = &env->cooking_pots[pot_index++];
                pot->cooking_state = NOT_COOKING;
                pot->ingredient_count = 0;
                for (int i = 0; i < MAX_INGREDIENTS; i++) pot->ingredient_types[i] = NO_ITEM;
            }
        }
    }
}

static void init_pot_indices(Overcooked* env) {
    env->pot_index_grid = calloc(env->width * env->height, sizeof(int));
    for (int i = 0; i < env->width * env->height; i++) env->pot_index_grid[i] = -1;
    int pot_idx = 0;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            if (env->grid[y * env->width + x] == STOVE) env->pot_index_grid[y * env->width + x] = pot_idx++;
        }
    }
}

static inline CookingPot* get_pot_at(Overcooked* env, int x, int y) {
    int idx = env->pot_index_grid[y * env->width + x];
    return (idx >= 0) ? &env->cooking_pots[idx] : NULL;
}

static void reset_item_grid(Overcooked* env) {
    for (int i = 0; i < env->width * env->height; i++) env->item_grid[i] = -1;
}

static void update_cooking(Overcooked* env) {
    for (int i = 0; i < env->num_stoves; i++) {
        if (env->cooking_pots[i].cooking_state == COOKING) {
            env->cooking_pots[i].cooking_progress++;
            if (env->cooking_pots[i].cooking_progress >= COOKING_TIME) {
                env->cooking_pots[i].cooking_state = COOKED;
            }
        }
    }
}

// --- LOGIQUE INTERACTION ---
static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx);

static void parse_grid(Overcooked* env) {
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            char tile = get_layout_tile(layout, x, y);
            int idx = y * env->width + x;
            switch (tile) {
                case '#': env->grid[idx] = WALL; break;
                case '1': env->grid[idx] = COUNTER; break;
                case '2': env->grid[idx] = STOVE; break;
                case '3': env->grid[idx] = CUTTING_BOARD; break;
                case '4': env->grid[idx] = INGREDIENT_BOX; break;
                case '5': env->grid[idx] = SERVING_AREA; break;
                case '6': env->grid[idx] = WALL; break;
                case '7': env->grid[idx] = PLATE_BOX; break;
                default: env->grid[idx] = EMPTY; break;
            }
        }
    }
}

static void init_static_cache(Overcooked* env) {
    env->cache.inv_width = 1.0f / env->width;
    env->cache.inv_height = 1.0f / env->height;
    env->cache.ingredient_box_count = 0; env->cache.plate_box_count = 0;
    env->cache.serving_area_count = 0; env->cache.stove_count = 0; env->cache.counter_count = 0;

    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int tile = env->grid[y * env->width + x];
            if (tile == INGREDIENT_BOX) { env->cache.ingredient_box_positions[env->cache.ingredient_box_count * 2] = x; env->cache.ingredient_box_positions[env->cache.ingredient_box_count * 2 + 1] = y; env->cache.ingredient_box_count++; }
            if (tile == PLATE_BOX) { env->cache.plate_box_positions[env->cache.plate_box_count * 2] = x; env->cache.plate_box_positions[env->cache.plate_box_count * 2 + 1] = y; env->cache.plate_box_count++; }
            if (tile == SERVING_AREA) { env->cache.serving_area_positions[env->cache.serving_area_count * 2] = x; env->cache.serving_area_positions[env->cache.serving_area_count * 2 + 1] = y; env->cache.serving_area_count++; }
            if (tile == STOVE) { env->cache.stove_positions[env->cache.stove_count * 2] = x; env->cache.stove_positions[env->cache.stove_count * 2 + 1] = y; env->cache.stove_count++; }
            if (tile == COUNTER) { env->cache.counter_positions[env->cache.counter_count * 2] = x; env->cache.counter_positions[env->cache.counter_count * 2 + 1] = y; env->cache.counter_count++; }
        }
    }
}

static inline void set_agent_position(Overcooked* env, int x, int y) { env->agent_position_mask |= (1ULL << (y * env->width + x)); }
static inline void clear_agent_position(Overcooked* env, int x, int y) { env->agent_position_mask &= ~(1ULL << (y * env->width + x)); }

static void handle_interaction(Overcooked* env, int agent_idx) {
    Agent* agent = &env->agents[agent_idx];
    int target_x = agent->x; int target_y = agent->y;

    switch (agent->facing_direction) {
        case 0: target_y -= 1; break; case 1: target_y += 1; break;
        case 2: target_x -= 1; break; case 3: target_x += 1; break;
    }
    if (target_x < 0 || target_x >= env->width || target_y < 0 || target_y >= env->height) return;

    int tile = env->grid[target_y * env->width + target_x];
    Item* item = get_item_at(env, target_x, target_y);
    CookingPot* pot = get_pot_at(env, target_x, target_y);

    if (tile == STOVE && pot != NULL) {
        if (agent->held_item == ONION || agent->held_item == TOMATO) {
            if (pot->cooking_state == NOT_COOKING && pot->ingredient_count < MAX_INGREDIENTS) {
                pot->ingredient_types[pot->ingredient_count++] = agent->held_item;
                if (agent->held_item == ONION) { pot->num_onions++; env->rewards[agent_idx] += env->rewards_config.ingredient_added; }
                else if (agent->held_item == TOMATO) pot->num_tomatoes++;
                agent->held_item = NO_ITEM;
            }
        }
        else if (agent->held_item == NO_ITEM && pot->ingredient_count > 0 && pot->cooking_state == NOT_COOKING) {
            pot->cooking_state = COOKING; pot->cooking_progress = 0; env->log.pots_started++;
            if (pot->num_onions == 3) env->rewards[agent_idx] += env->rewards_config.pot_started;
        }
        else if (agent->held_item == PLATE && pot->cooking_state == COOKED) {
            agent->held_item = PLATED_SOUP; agent->held_soup_onions = pot->num_onions; agent->held_soup_tomatoes = pot->num_tomatoes; agent->held_soup_total = pot->ingredient_count;
            env->rewards[agent_idx] += env->rewards_config.soup_plated;
            pot->cooking_state = NOT_COOKING; pot->ingredient_count = 0; pot->num_onions = 0; pot->num_tomatoes = 0;
        }
        return;
    }

    if (tile == SERVING_AREA && agent->held_item == PLATED_SOUP) {
        evaluate_dish_served(env, agent, agent_idx);
        agent->held_item = NO_ITEM; agent->held_soup_onions = 0; agent->held_soup_tomatoes = 0; agent->held_soup_total = 0;
        return;
    }

    if (agent->held_item != NO_ITEM) {
        if ((tile == COUNTER || tile == CUTTING_BOARD || tile == EMPTY) && item == NULL) {
            if (tile != EMPTY) {
                add_item(env, agent->held_item, target_x, target_y);
                if (agent->held_item == PLATED_SOUP) {
                    Item* placed = get_item_at(env, target_x, target_y);
                    placed->num_onions = agent->held_soup_onions; placed->num_tomatoes = agent->held_soup_tomatoes; placed->total_ingredients = agent->held_soup_total;
                }
            }
            agent->held_item = NO_ITEM; agent->held_soup_onions = 0; agent->held_soup_tomatoes = 0; agent->held_soup_total = 0;
            env->log.items_dropped++;
        }
    } else {
        if (item != NULL) {
            if (item->type == PLATED_SOUP) { agent->held_soup_onions = item->num_onions; agent->held_soup_tomatoes = item->num_tomatoes; agent->held_soup_total = item->total_ingredients; }
            agent->held_item = item->type; remove_item(env, target_x, target_y);
        }
        else if (tile == INGREDIENT_BOX) { agent->held_item = ONION; env->log.ingredients_picked++; env->rewards[agent_idx] += env->rewards_config.ingredient_picked; }
        else if (tile == PLATE_BOX) { agent->held_item = PLATE; env->rewards[agent_idx] += env->rewards_config.plate_picked; }
    }
}

static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx) {
    if (agent->held_soup_onions == 3) {
        env->rewards[agent_idx] += env->rewards_config.dish_served_agent;
        for (int i = 0; i < env->num_agents; i++) env->rewards[i] += env->rewards_config.dish_served_whole_team;
        env->log.episode_length += agent->ticks_since_reward;
        env->log.score += 25.0 / (agent->ticks_since_reward + 1);
        agent->ticks_since_reward = 0; env->log.correct_dishes++; env->log.n++;
    } else {
        for (int i = 0; i < env->num_agents; i++) env->rewards[i] += env->rewards_config.wrong_dish_served;
        env->log.wrong_dishes++;
    }
    env->log.dishes_served++;
}

// --- OBSERVATIONS ---
// (Colle ici exactement les fonctions de overcooked_obs.h de l'officiel : find_nearest_plated_soup, compute_tile_proximity_cached, find_nearest_empty_counter, et compute_observations)
// Assure-toi de garder la taille de 43 dims.

// --- RENDER ---
// (Colle ici exactement les fonctions get_agent_color, unload_textures, c_render de l'officiel)


// --- API PRINCIPALE ET PARALLÉLISME ---

static void init(Overcooked* env) {
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    env->width = layout->width;
    env->height = layout->height;
    env->grid = calloc(env->width * env->height, sizeof(char));
    env->max_items = 20;
    env->items = calloc(env->max_items, sizeof(Item));
    env->agents = calloc(env->num_agents, sizeof(Agent));
    parse_grid(env);
    init_static_cache(env);
    init_cooking_pots(env);
    init_pot_indices(env);
    env->item_grid = calloc(env->width * env->height, sizeof(int));
    env->client = NULL;
    memset(&env->log, 0, sizeof(Log));
}

void c_reset(Overcooked* env) {
    env->tick = 0;
    env->num_items = 0;
    reset_item_grid(env);
    parse_grid(env);

    for (int i = 0; i < env->num_stoves; i++) {
        env->cooking_pots[i].cooking_state = NOT_COOKING;
        env->cooking_pots[i].cooking_progress = 0;
        env->cooking_pots[i].ingredient_count = 0;
        env->cooking_pots[i].num_onions = 0;
    }
    
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    env->agent_position_mask = 0;
    for (int i = 0; i < env->num_agents; i++) {
        if (i < layout->num_spawns) {
            env->agents[i].x = layout->spawn_positions[i * 2];
            env->agents[i].y = layout->spawn_positions[i * 2 + 1];
        } else {
            env->agents[i].x = 1; env->agents[i].y = 1;
        }
        env->agents[i].held_item = NO_ITEM;
        env->agents[i].facing_direction = 0;
        env->agents[i].ticks_since_reward = 0;
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
        set_agent_position(env, env->agents[i].x, env->agents[i].y);
    }
    // compute_observations(env); <-- A appeler depuis ton binding.c si besoin
}

void c_step(Overcooked* env) {
    env->tick++;
    int new_x[MAX_SPAWN_POSITIONS];
    int new_y[MAX_SPAWN_POSITIONS];

    // 1. Calcul des intentions de base
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = env->rewards_config.step_penalty;
        env->agents[i].ticks_since_reward++;

        int action = env->actions[i];
        new_x[i] = env->agents[i].x;
        new_y[i] = env->agents[i].y;

        if (action == ACTION_UP) { new_y[i] -= 1; env->agents[i].facing_direction = 0; }
        else if (action == ACTION_DOWN) { new_y[i] += 1; env->agents[i].facing_direction = 1; }
        else if (action == ACTION_LEFT) { new_x[i] -= 1; env->agents[i].facing_direction = 2; }
        else if (action == ACTION_RIGHT) { new_x[i] += 1; env->agents[i].facing_direction = 3; }

        // Murs et obstacles (Vérification statique)
        if (new_x[i] < 0 || new_x[i] >= env->width || new_y[i] < 0 || new_y[i] >= env->height || env->grid[new_y[i] * env->width + new_x[i]] != EMPTY) {
            new_x[i] = env->agents[i].x;
            new_y[i] = env->agents[i].y;
        }
    }

    // 2. Gestion Parallèle des collisions Multi-Agents
    if (env->num_agents == 2) {
        // Swap
        if (new_x[0] == env->agents[1].x && new_y[0] == env->agents[1].y && new_x[1] == env->agents[0].x && new_y[1] == env->agents[0].y) {
            new_x[0] = env->agents[0].x; new_y[0] = env->agents[0].y;
            new_x[1] = env->agents[1].x; new_y[1] = env->agents[1].y;
            env->log.agent_collisions += 2;
        }
        // Case cible identique
        if (new_x[0] == new_x[1] && new_y[0] == new_y[1]) {
            new_x[0] = env->agents[0].x; new_y[0] = env->agents[0].y;
            new_x[1] = env->agents[1].x; new_y[1] = env->agents[1].y;
            env->log.agent_collisions += 2;
        }
    }

    // 3. Application des mouvements
    env->agent_position_mask = 0;
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].x = new_x[i];
        env->agents[i].y = new_y[i];
        set_agent_position(env, new_x[i], new_y[i]);
    }

    // 4. Interactions & Cuissons
    for (int i = 0; i < env->num_agents; i++) {
        if (env->actions[i] == ACTION_INTERACT) handle_interaction(env, i);
    }
    update_cooking(env);

    // 5. Anti-blocage & Truncation
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    for (int i = 0; i < env->num_agents; i++) {
        if (env->agents[i].ticks_since_reward % 512 == 0 && env->agents[i].ticks_since_reward > 0) {
            clear_agent_position(env, env->agents[i].x, env->agents[i].y);
            env->agents[i].x = layout->spawn_positions[i * 2]; env->agents[i].y = layout->spawn_positions[i * 2 + 1];
            set_agent_position(env, env->agents[i].x, env->agents[i].y);
            env->agents[i].held_item = NO_ITEM;
        }
        env->log.episode_return += env->rewards[i];
    }
    
    if (env->tick >= env->max_ticks) {
        for(int i = 0; i < env->num_agents; i++) env->terminals[i] = 1;
        c_reset(env);
    }
}

void c_close(Overcooked* env) {
    free(env->grid); free(env->items); free(env->agents);
    free(env->cooking_pots); free(env->pot_index_grid); free(env->item_grid);
    if (env->client != NULL) { unload_textures(env->client); free(env->client); }
}

#endif // OVERCOOKED_H