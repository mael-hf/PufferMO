#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "raylib.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define USE_GAMMA

#define GRID_SIZE 5
#define REWARD_DIM 3
#define REWARD_KILL_IDX 0
#define REWARD_GOLD_IDX 1
#define REWARD_GEM_IDX  2

// Uniform Dirichlet prior over the 3-objective simplex
static const double dirichlet_alpha[REWARD_DIM] = {1.0, 1.0, 1.0};

// Cell codes for the hardcoded map
#define CELL_EMPTY 0
#define CELL_R1    1   // gold
#define CELL_R2    2   // gem (diamond)
#define CELL_E1    3   // enemy 1
#define CELL_E2    4   // enemy 2
#define CELL_H     5   // home

// Colors for the renderer
static const Color PUFF_RED   = (Color){187,   0,   0, 255};
static const Color PUFF_CYAN  = (Color){  0, 187, 187, 255};
static const Color PUFF_GOLD  = (Color){218, 165,  32, 255};
static const Color PUFF_GEM   = (Color){  0, 200, 220, 255};
static const Color PUFF_HOME  = (Color){ 80,  80,  80, 255};
static const Color PUFF_WHITE = (Color){241, 241, 241, 255};
static const Color PUFF_BG    = (Color){  6,  24,  24, 255};

// Map layout matches MO-Gymnasium's self.map: row 0 = top.
//   row 0: . . R1 E2 .
//   row 1: . . E1 .  R2
//   row 2: . . .  .  .
//   row 3: . . .  .  .
//   row 4: . . H  .  .
static const int MAP[GRID_SIZE][GRID_SIZE] = {
    {0, 0, 1, 4, 0},
    {0, 0, 3, 0, 2},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 5, 0, 0},
};

// Action -> (delta_row, delta_col). Matches Python: 0=up, 1=down, 2=left, 3=right.
static const int DR[4] = {-1,  1,  0,  0};
static const int DC[4] = { 0,  0, -1,  1};

// Log struct — MUST contain only floats. The binding aggregates by treating
// this as a flat float array; non-float fields will corrupt logging.
typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float scalarized_episode_return;
    float discounted_episode_return;
    float discounted_scalarized_episode_return;
    float episode_return_kill;
    float episode_return_gold;
    float episode_return_gem;
    float discounted_episode_return_kill;
    float discounted_episode_return_gold;
    float discounted_episode_return_gem;
    float weight_kill;
    float weight_gold;
    float weight_gem;
    float episode_length;
    float n;   // must be last
};

typedef struct Client Client;
typedef struct ResourceGathering ResourceGathering;
struct ResourceGathering {
    // Required by env_binding_mo.h. Names and types must match exactly.
    unsigned char* observations;
    int*           actions;
    float*         rewards;
    float*         weights;
    unsigned char* terminals;
    gsl_rng*       gsl_rng;
    Log            log;

    // Env-specific state
    int   row;
    int   col;
    int   has_gold;
    int   has_gem;
    int   last_action;
    float kill_probability;

    // Discounting (USE_GAMMA path)
    double gamma;
    double gamma_t;

    // For evaluation-time weight override via my_put()
    bool manual_weights;

    // Lazy raylib renderer
    Client* client;
};

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

static void write_observations(ResourceGathering* env) {
    env->observations[0] = (unsigned char)env->row;
    env->observations[1] = (unsigned char)env->col;
    env->observations[2] = (unsigned char)env->has_gold;
    env->observations[3] = (unsigned char)env->has_gem;
}

static void sample_weights(ResourceGathering* env) {
    if (env->manual_weights) {
        return;
    }
    double buf[REWARD_DIM];
    gsl_ran_dirichlet(env->gsl_rng, REWARD_DIM, dirichlet_alpha, buf);
    env->weights[REWARD_KILL_IDX] = (float)buf[0];
    env->weights[REWARD_GOLD_IDX] = (float)buf[1];
    env->weights[REWARD_GEM_IDX]  = (float)buf[2];
}

// -----------------------------------------------------------------------------
// Reset and step
// -----------------------------------------------------------------------------

void c_reset(ResourceGathering* env) {
    env->row         = 4;
    env->col         = 2;
    env->has_gold    = 0;
    env->has_gem     = 0;
    env->last_action = -1;
    env->gamma_t     = env->gamma;
    env->log         = (Log){0};

    write_observations(env);
    sample_weights(env);
}

void c_step(ResourceGathering* env) {
    // Zero rewards and terminal at the start of every step
    for (int k = 0; k < REWARD_DIM; k++) {
        env->rewards[k] = 0.0f;
    }
    env->terminals[0] = 0;
    env->log.episode_length += 1.0f;
    env->gamma_t *= env->gamma;

    int action = env->actions[0];
    env->last_action = action;

    // Compute proposed move
    int next_row = env->row + DR[action];
    int next_col = env->col + DC[action];

    // Bounds check: off-grid moves are wasted (agent stays put)
    if (next_row >= 0 && next_row < GRID_SIZE &&
        next_col >= 0 && next_col < GRID_SIZE) {
        env->row = next_row;
        env->col = next_col;
    }

    // Cache weights for scalarization and logging
    float w_kill = env->weights[REWARD_KILL_IDX];
    float w_gold = env->weights[REWARD_GOLD_IDX];
    float w_gem  = env->weights[REWARD_GEM_IDX];
    env->log.weight_kill = w_kill;
    env->log.weight_gold = w_gold;
    env->log.weight_gem  = w_gem;

    int cell = MAP[env->row][env->col];

    if (cell == CELL_R1) {
        env->has_gold = 1;
    } else if (cell == CELL_R2) {
        env->has_gem = 1;
    } else if (cell == CELL_E1 || cell == CELL_E2) {
        double roll = gsl_rng_uniform(env->gsl_rng);
        if (roll < (double)env->kill_probability) {
            float r_kill = -1.0f;
            env->rewards[REWARD_KILL_IDX] = r_kill;
            env->terminals[0] = 1;

            float scalarized = r_kill * w_kill;
            env->log.episode_return_kill              += r_kill;
            env->log.discounted_episode_return_kill   += (float)env->gamma_t * r_kill;
            env->log.episode_return                   += r_kill;
            env->log.scalarized_episode_return        += scalarized;
            env->log.discounted_episode_return        += (float)env->gamma_t * r_kill;
            env->log.discounted_scalarized_episode_return += (float)env->gamma_t * scalarized;
            env->log.score = r_kill;
            env->log.perf  = env->log.score;
            env->log.n    += 1.0f;

            write_observations(env);
            c_reset(env);
            return;
        }
    } else if (cell == CELL_H) {
        env->terminals[0] = 1;
        float r_gold = (float)env->has_gold;
        float r_gem  = (float)env->has_gem;
        env->rewards[REWARD_GOLD_IDX] = r_gold;
        env->rewards[REWARD_GEM_IDX]  = r_gem;

        float scalarized = r_gold * w_gold + r_gem * w_gem;
        env->log.episode_return_gold              += r_gold;
        env->log.episode_return_gem               += r_gem;
        env->log.discounted_episode_return_gold   += (float)env->gamma_t * r_gold;
        env->log.discounted_episode_return_gem    += (float)env->gamma_t * r_gem;
        env->log.episode_return                   += r_gold + r_gem;
        env->log.scalarized_episode_return        += scalarized;
        env->log.discounted_episode_return        += (float)env->gamma_t * (r_gold + r_gem);
        env->log.discounted_scalarized_episode_return += (float)env->gamma_t * scalarized;
        env->log.score = r_gold + r_gem;
        env->log.perf  = env->log.score;
        env->log.n    += 1.0f;

        write_observations(env);
        c_reset(env);
        return;
    }
    // CELL_EMPTY falls through: no reward, just update observation

    write_observations(env);
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------

#define CELL_SIZE       64
#define WINDOW_PADDING  20
#define WINDOW_W (GRID_SIZE * CELL_SIZE + 2 * WINDOW_PADDING)
#define WINDOW_H (GRID_SIZE * CELL_SIZE + 2 * WINDOW_PADDING + 40)

struct Client {
    int placeholder;  // raylib manages the window globally; nothing to store here
};

void c_render(ResourceGathering* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (!IsWindowReady()) {
        InitWindow(WINDOW_W, WINDOW_H, "PufferLib Resource Gathering");
        SetTargetFPS(4);
    }

    BeginDrawing();
    ClearBackground(PUFF_BG);

    // Grid
    for (int r = 0; r < GRID_SIZE; r++) {
        for (int c = 0; c < GRID_SIZE; c++) {
            int x = WINDOW_PADDING + c * CELL_SIZE;
            int y = WINDOW_PADDING + r * CELL_SIZE;

            Color bg = ((r + c) % 2 == 0)
                ? (Color){40, 60, 40, 255}
                : (Color){50, 75, 50, 255};
            DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, bg);

            int cell = MAP[r][c];
            if (cell == CELL_R1) {
                if (!env->has_gold) {
                    DrawRectangle(x + 16, y + 16, CELL_SIZE - 32, CELL_SIZE - 32, PUFF_GOLD);
                }
            } else if (cell == CELL_R2) {
                if (!env->has_gem) {
                    DrawRectangle(x + 16, y + 16, CELL_SIZE - 32, CELL_SIZE - 32, PUFF_GEM);
                }
            } else if (cell == CELL_E1 || cell == CELL_E2) {
                DrawRectangle(x + 8, y + 8, CELL_SIZE - 16, CELL_SIZE - 16, PUFF_RED);
            } else if (cell == CELL_H) {
                DrawRectangle(x + 4, y + 4, CELL_SIZE - 8, CELL_SIZE - 8, PUFF_HOME);
            }
        }
    }

    // Agent on top
    int ax = WINDOW_PADDING + env->col * CELL_SIZE + CELL_SIZE / 4;
    int ay = WINDOW_PADDING + env->row * CELL_SIZE + CELL_SIZE / 4;
    DrawRectangle(ax, ay, CELL_SIZE / 2, CELL_SIZE / 2, PUFF_CYAN);

    // Inventory status
    char status[64];
    snprintf(status, sizeof(status), "Gold: %s   Gem: %s",
             env->has_gold ? "yes" : "no",
             env->has_gem  ? "yes" : "no");
    DrawText(status, WINDOW_PADDING, WINDOW_H - 30, 20, PUFF_WHITE);

    EndDrawing();
}

void c_close(ResourceGathering* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}