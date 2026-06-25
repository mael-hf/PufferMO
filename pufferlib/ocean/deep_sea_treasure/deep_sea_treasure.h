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

#define GRID_ROWS 11
#define GRID_COLS 11
#define MAX_EPISODE_STEPS 100   

#define REWARD_DIM 2
#define REWARD_TREASURE_IDX 0
#define REWARD_TIME_IDX     1

static const double dirichlet_alpha[REWARD_DIM] = {1.0, 1.0};

#define ROCK -10.0f

static const float MAP[GRID_ROWS][GRID_COLS] = {
    {  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {  0.7f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-10.0f,  8.2f,  0.0f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-10.0f,-10.0f, 11.5f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-10.0f,-10.0f,-10.0f, 14.0f, 15.1f, 16.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,19.6f,20.3f, 0.0f, 0.0f, 0.0f},
    {-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f, 0.0f, 0.0f, 0.0f},
    {-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,22.4f, 0.0f, 0.0f},
    {-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,23.7f,0.0f},
};

static const int DR[4] = {-1,  1,  0,  0};
static const int DC[4] = { 0,  0, -1,  1};

static const Color PUFF_SUB   = (Color){255, 221,  0, 255};
static const Color PUFF_CHEST = (Color){218, 165, 32, 255};
static const Color PUFF_ROCK  = (Color){ 40,  40, 55, 255};
static const Color PUFF_WATER = (Color){ 20,  60, 110, 255};
static const Color PUFF_WHITE = (Color){241, 241, 241, 255};
static const Color PUFF_BG    = (Color){  6,  24,  24, 255};


typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float scalarized_episode_return;
    float discounted_episode_return;
    float discounted_scalarized_episode_return;
    float episode_return_treasure;
    float episode_return_time;
    float discounted_episode_return_treasure;
    float discounted_episode_return_time;
    float weight_treasure;
    float weight_time;
    float episode_length;
    float n;   // must be last
};

typedef struct Client Client;
typedef struct DeepSeaTreasure DeepSeaTreasure;
struct DeepSeaTreasure {
    unsigned char* observations;
    int* actions;
    float* rewards;
    float* weights;
    unsigned char* terminals;
    gsl_rng* gsl_rng;
    Log log;
    int row;
    int col;
    int step_count;
    int last_action;
    double gamma;
    double gamma_t;
    bool manual_weights;

    Client* client;
    float ep_return;
    float ep_return_treasure;
    float ep_return_time;
    float ep_length;
    float ep_discounted_return;
    float ep_discounted_return_treasure;
    float ep_discounted_return_time;
    float ep_scalarized_return;
    float ep_discounted_scalarized_return;
};


static void write_observations(DeepSeaTreasure* env) {
    env->observations[0] = (unsigned char)env->row;
    env->observations[1] = (unsigned char)env->col;
}

static void sample_weights(DeepSeaTreasure* env) {
    if (env->manual_weights) {
        return;
    }
    double buf[REWARD_DIM];
    gsl_ran_dirichlet(env->gsl_rng, REWARD_DIM, dirichlet_alpha, buf);
    env->weights[REWARD_TREASURE_IDX] = (float)buf[0];
    env->weights[REWARD_TIME_IDX]     = (float)buf[1];
}

static bool is_valid_state(int row, int col) {
    if (row < 0 || row >= GRID_ROWS || col < 0 || col >= GRID_COLS) {
        return false;
    }
    if (MAP[row][col] == ROCK) {
        return false;
    }
    return true;
}


void c_reset(DeepSeaTreasure* env) {
    env->row = 0;
    env->col = 0;
    env->step_count  = 0;
    env->last_action = -1;
    env->gamma_t = env->gamma;
    env->log = (Log){0};

    write_observations(env);
    sample_weights(env);
}

void c_step(DeepSeaTreasure* env) {
    for (int k = 0; k < REWARD_DIM; k++) {
        env->rewards[k] = 0.0f;
    }
    env->terminals[0] = 0;
    env->step_count += 1;
    env->log.episode_length += 1.0f;
    env->gamma_t *= env->gamma;

    int action = env->actions[0];
    env->last_action = action;

    int next_row = env->row + DR[action];
    int next_col = env->col + DC[action];
    if (is_valid_state(next_row, next_col)) {
        env->row = next_row;
        env->col = next_col;
    }


    float w_treasure = env->weights[REWARD_TREASURE_IDX];
    float w_time     = env->weights[REWARD_TIME_IDX];
    env->log.weight_treasure = w_treasure;
    env->log.weight_time     = w_time;


    float cell = MAP[env->row][env->col];

    float r_time = -1.0f;

    float r_treasure = 0.0f;
    bool terminal = false;
    if (cell != 0.0f && cell != ROCK) {
        r_treasure = cell;
        terminal = true;
    }

    env->rewards[REWARD_TREASURE_IDX] = r_treasure;
    env->rewards[REWARD_TIME_IDX] = r_time;

    float scalarized = r_treasure * w_treasure + r_time * w_time;
    env->log.episode_return_treasure += r_treasure;
    env->log.episode_return_time += r_time;
    env->log.discounted_episode_return_treasure += (float)env->gamma_t * r_treasure;
    env->log.discounted_episode_return_time += (float)env->gamma_t * r_time;
    env->log.episode_return += r_treasure + r_time;
    env->log.scalarized_episode_return += scalarized;
    env->log.discounted_episode_return += (float)env->gamma_t * (r_treasure + r_time);
    env->log.discounted_scalarized_episode_return += (float)env->gamma_t * scalarized;

    // Truncate at the fixed horizon 
    if (env->step_count >= MAX_EPISODE_STEPS) {
        terminal = true;
    }

    if (terminal) {
        env->terminals[0] = 1;
        env->log.score = env->log.episode_return_treasure;
        env->log.perf = env->log.score;
        env->log.n += 1.0f;
        write_observations(env);
        c_reset(env);
        return;
    }

    write_observations(env);
}


#define CELL_SIZE 40
#define WINDOW_PADDING  20
#define WINDOW_W (GRID_COLS * CELL_SIZE + 2 * WINDOW_PADDING)
#define WINDOW_H (GRID_ROWS * CELL_SIZE + 2 * WINDOW_PADDING + 40)

struct Client {
    int placeholder; 
};

void c_render(DeepSeaTreasure* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (!IsWindowReady()) {
        InitWindow(WINDOW_W, WINDOW_H, "PufferLib Deep Sea Treasure");
        SetTargetFPS(4);
    }

    BeginDrawing();
    ClearBackground(PUFF_BG);

    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            int x = WINDOW_PADDING + c * CELL_SIZE;
            int y = WINDOW_PADDING + r * CELL_SIZE;
            float cell = MAP[r][c];

            if (cell == ROCK) {
                DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, PUFF_ROCK);
            } else {
                DrawRectangle(x, y, CELL_SIZE, CELL_SIZE, PUFF_WATER);
                if (cell != 0.0f) {
                    DrawRectangle(x + 6, y + 6, CELL_SIZE - 12, CELL_SIZE - 12, PUFF_CHEST);
                    char val[8];
                    snprintf(val, sizeof(val), "%.1f", cell);
                    DrawText(val, x + 2, y + 2, 10, PUFF_WHITE);
                }
            }
        }
    }

    // Submarine (agent)
    int ax = WINDOW_PADDING + env->col * CELL_SIZE + CELL_SIZE / 4;
    int ay = WINDOW_PADDING + env->row * CELL_SIZE + CELL_SIZE / 4;
    DrawRectangle(ax, ay, CELL_SIZE / 2, CELL_SIZE / 2, PUFF_SUB);

    char status[64];
    snprintf(status, sizeof(status), "Steps: %d", env->step_count);
    DrawText(status, WINDOW_PADDING, WINDOW_H - 30, 20, PUFF_WHITE);

    EndDrawing();
}

void c_close(DeepSeaTreasure* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}