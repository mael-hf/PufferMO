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

#define REWARD_DIM 3
#define REWARD_ORE0_IDX 0
#define REWARD_ORE1_IDX 1
#define REWARD_FUEL_IDX 2

#define OBS_DIM 7
#define NUM_ACTIONS 6

#define ACT_MINE 0
#define ACT_LEFT 1
#define ACT_RIGHT 2
#define ACT_ACCEL 3
#define ACT_BRAKE 4
#define ACT_NONE 5

#define EPS_SPEED 0.001f
#define HOME_X 0.0f
#define HOME_Y 0.0f
#define ROTATION 10.0f
#define MAX_SPEED 1.0f
#define ACCELERATION 0.0075f
#define DECELERATION 1.0f
#define MINE_RADIUS 0.14f
#define BASE_RADIUS 0.15f
#define FUEL_IDLE -0.005f
#define FUEL_ACC -0.025f
#define FUEL_MINE -0.05f
#define MARGIN 0.16f
#define MINE_LOCATION_TRIES 100

static const double dirichlet_alpha[REWARD_DIM] = {1.0, 1.0, 1.0};

typedef struct {
    float x, y;
    float dist_mean[2];
    float dist_std[2];
} Mine;

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float scalarized_episode_return;
    float discounted_episode_return;
    float discounted_scalarized_episode_return;
    float episode_return_ore0;
    float episode_return_ore1;
    float episode_return_fuel;
    float discounted_episode_return_ore0;
    float discounted_episode_return_ore1;
    float discounted_episode_return_fuel;
    float weight_ore0;
    float weight_ore1;
    float weight_fuel;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct Minecart Minecart;
struct Minecart {
    float* observations;
    int* actions;
    float* rewards;
    float* weights;
    unsigned char* terminals;
    gsl_rng* gsl_rng;
    Log log;

    float pos_x;
    float pos_y;
    float speed;
    float angle;
    float content[2];
    bool departed;

    float capacity;
    int frame_skip;
    int incremental_frame_skip;
    double gamma;
    double gamma_t;
    bool manual_weights;

    int mine_cnt;
    Mine* mines;

    int tick;
    int max_ticks;
    int max_ticks_offset_mod;
    int current_max_ticks;
    bool done;

    Client* client;
};

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float mag2d(float x, float y) {
    return sqrtf(x*x + y*y);
}

static void write_observations(Minecart* env) {
    float angle_rad = env->angle * M_PI / 180.0f;
    env->observations[0] = env->pos_x;
    env->observations[1] = env->pos_y;
    env->observations[2] = env->speed;
    env->observations[3] = sinf(angle_rad);
    env->observations[4] = cosf(angle_rad);
    env->observations[5] = env->content[0] / env->capacity;
    env->observations[6] = env->content[1] / env->capacity;
}

static void sample_weights(Minecart* env) {
    if (env->manual_weights) return;
    double buf[REWARD_DIM];
    gsl_ran_dirichlet(env->gsl_rng, REWARD_DIM, dirichlet_alpha, buf);
    env->weights[REWARD_ORE0_IDX] = (float)buf[0];
    env->weights[REWARD_ORE1_IDX] = (float)buf[1];
    env->weights[REWARD_FUEL_IDX] = (float)buf[2];
}

static void step_cart(Minecart* env) {
    if (env->speed < EPS_SPEED) return;

    float angle_rad = env->angle * M_PI / 180.0f;
    float x_vel = env->speed * cosf(angle_rad);
    float y_vel = env->speed * sinf(angle_rad);
    float x = env->pos_x;
    float y = env->pos_y;

    if (y != 0.0f && y != 1.0f && (y_vel > EPS_SPEED || y_vel < -EPS_SPEED)) {
        if (x >= 1.0f && x_vel > 0.0f)
            env->angle += copysignf(ROTATION, y_vel);
        if (x <= 0.0f && x_vel < 0.0f)
            env->angle -= copysignf(ROTATION, y_vel);
    }
    if (x != 0.0f && x != 1.0f && (x_vel > EPS_SPEED || x_vel < -EPS_SPEED)) {
        if (y >= 1.0f && y_vel > 0.0f)
            env->angle -= copysignf(ROTATION, x_vel);
        if (y <= 0.0f && y_vel < 0.0f)
            env->angle += copysignf(ROTATION, x_vel);
    }

    env->angle = fmodf(env->angle, 360.0f);
    if (env->angle < 0.0f) env->angle += 360.0f;

    angle_rad = env->angle * M_PI / 180.0f;
    x_vel = env->speed * cosf(angle_rad);
    y_vel = env->speed * sinf(angle_rad);

    float new_x = clampf(x + x_vel, 0.0f, 1.0f);
    float new_y = clampf(y + y_vel, 0.0f, 1.0f);
    env->speed = mag2d(new_x - x, new_y - y);
    env->pos_x = new_x;
    env->pos_y = new_y;
}

static void perform_mine(Minecart* env) {
    if (env->speed >= EPS_SPEED) return;

    int closest = -1;
    float min_dist = 1e9f;
    for (int i = 0; i < env->mine_cnt; i++) {
        float d = mag2d(env->pos_x - env->mines[i].x, env->pos_y - env->mines[i].y);
        if (d < min_dist) { min_dist = d; closest = i; }
    }
    if (closest < 0 || min_dist > MINE_RADIUS) return;

    Mine* mine = &env->mines[closest];
    float total = env->content[0] + env->content[1];
    float free_space = env->capacity - total;
    if (free_space <= 0.0f) return;

    float mined0 = fmaxf(0.0f, mine->dist_mean[0] + (float)gsl_ran_gaussian(env->gsl_rng, mine->dist_std[0]));
    float mined1 = fmaxf(0.0f, mine->dist_mean[1] + (float)gsl_ran_gaussian(env->gsl_rng, mine->dist_std[1]));
    float total_mined = mined0 + mined1;
    if (total_mined > free_space) {
        float scale = free_space / total_mined;
        mined0 *= scale;
        mined1 *= scale;
    }
    env->content[0] += mined0;
    env->content[1] += mined1;
}

static void generate_mines(Minecart* env) {
    for (int i = 0; i < env->mine_cnt; i++) {
        float x, y;
        int tries = 0;
        do {
            x = (float)rand() / (float)RAND_MAX;
            y = (float)rand() / (float)RAND_MAX;
            tries++;
        } while (mag2d(x, y) < BASE_RADIUS + MARGIN && tries < MINE_LOCATION_TRIES);

        env->mines[i].x = x;
        env->mines[i].y = y;
        env->mines[i].dist_mean[0] = (float)rand() / (float)RAND_MAX;
        env->mines[i].dist_std[0] = (float)rand() / (float)RAND_MAX;
        env->mines[i].dist_mean[1] = (float)rand() / (float)RAND_MAX;
        env->mines[i].dist_std[1] = (float)rand() / (float)RAND_MAX;
    }
}

void c_reset(Minecart* env) {
    env->pos_x = HOME_X;
    env->pos_y = HOME_Y;
    env->speed = 0.0f;
    env->angle = 45.0f;
    env->content[0] = 0.0f;
    env->content[1] = 0.0f;
    env->departed = false;
    env->done = false;
    env->tick = 0;
    env->gamma_t = env->gamma;
    env->log = (Log){0};

    generate_mines(env);
    write_observations(env);
    sample_weights(env);
}

void c_step(Minecart* env) {
    if (env->done) {
        env->terminals[0] = 1;
        return;
    }

    env->tick++;
    env->gamma_t *= env->gamma;
    env->log.episode_length += 1.0f;

    for (int k = 0; k < REWARD_DIM; k++)
        env->rewards[k] = 0.0f;
    env->terminals[0] = 0;

    int action = env->actions[0];

    float reward_ore0 = 0.0f;
    float reward_ore1 = 0.0f;
    float reward_fuel = FUEL_IDLE * env->frame_skip;
    if (action == ACT_ACCEL)
        reward_fuel += FUEL_ACC * env->frame_skip;
    else if (action == ACT_MINE)
        reward_fuel += FUEL_MINE * env->frame_skip;

    int outer = env->incremental_frame_skip ? env->frame_skip : 1;
    for (int s = 0; s < outer; s++) {
        float af = env->incremental_frame_skip ? 1.0f : (float)env->frame_skip;

        if (action == ACT_LEFT)
            env->angle -= ROTATION * af;
        else if (action == ACT_RIGHT)
            env->angle += ROTATION * af;
        else if (action == ACT_ACCEL)
            env->speed = fminf(env->speed + ACCELERATION * af, MAX_SPEED);
        else if (action == ACT_BRAKE)
            env->speed = fmaxf(env->speed - DECELERATION * af, 0.0f);
        else if (action == ACT_MINE) {
            int inner = env->incremental_frame_skip ? 1 : env->frame_skip;
            for (int m = 0; m < inner; m++)
                perform_mine(env);
        }

        int inner = env->incremental_frame_skip ? 1 : env->frame_skip;
        for (int m = 0; m < inner; m++)
            step_cart(env);

        if (env->done) break;

        float dist = mag2d(env->pos_x, env->pos_y);
        if (dist < BASE_RADIUS) {
            if (env->departed) {
                env->done = true;
                env->terminals[0] = 1;
                reward_ore0 = env->content[0];
                reward_ore1 = env->content[1];
                env->content[0] = 0.0f;
                env->content[1] = 0.0f;
            }
        } else {
            env->departed = true;
        }
    }

    env->angle = fmodf(env->angle, 360.0f);
    if (env->angle < 0.0f) env->angle += 360.0f;

    env->rewards[REWARD_ORE0_IDX] = reward_ore0;
    env->rewards[REWARD_ORE1_IDX] = reward_ore1;
    env->rewards[REWARD_FUEL_IDX] = reward_fuel;

    float w_ore0 = env->weights[REWARD_ORE0_IDX];
    float w_ore1 = env->weights[REWARD_ORE1_IDX];
    float w_fuel = env->weights[REWARD_FUEL_IDX];
    env->log.weight_ore0 = w_ore0;
    env->log.weight_ore1 = w_ore1;
    env->log.weight_fuel = w_fuel;

    float scalarized = reward_ore0 * w_ore0 + reward_ore1 * w_ore1 + reward_fuel * w_fuel;

    env->log.episode_return_ore0 += reward_ore0;
    env->log.episode_return_ore1 += reward_ore1;
    env->log.episode_return_fuel += reward_fuel;
    env->log.episode_return += reward_ore0 + reward_ore1 + reward_fuel;
    env->log.scalarized_episode_return += scalarized;
    env->log.discounted_episode_return_ore0 += (float)env->gamma_t * reward_ore0;
    env->log.discounted_episode_return_ore1 += (float)env->gamma_t * reward_ore1;
    env->log.discounted_episode_return_fuel += (float)env->gamma_t * reward_fuel;
    env->log.discounted_episode_return += (float)env->gamma_t * (reward_ore0 + reward_ore1 + reward_fuel);
    env->log.discounted_scalarized_episode_return += (float)env->gamma_t * scalarized;

    if (env->terminals[0]) {
        env->log.score = reward_ore0 + reward_ore1;
        env->log.perf = env->log.score / env->log.episode_length;
        env->log.n += 1.0f;
    }

    write_observations(env);

    if (env->max_ticks > 0 && env->tick >= env->current_max_ticks) {
        if (!env->terminals[0]) {
            env->terminals[0] = 1;
            env->log.score = env->log.episode_return_ore0 + env->log.episode_return_ore1;
            env->log.perf = env->log.score / env->log.episode_length;
            env->log.n += 1.0f;
        }
    }
}

#define CELL_SIZE 480
#define HOME_RADIUS 50

struct Client {
    int placeholder;
};

void c_render(Minecart* env) {
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (!IsWindowReady()) {
        InitWindow(CELL_SIZE, CELL_SIZE, "PufferLib Minecart");
        SetTargetFPS(10);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int margin = (int)(MARGIN * CELL_SIZE);
    int play_w = CELL_SIZE - 2 * margin;
    int play_h = CELL_SIZE - 2 * margin;

    for (int r = 0; r < 10; r++) {
        for (int c = 0; c < 10; c++) {
            Color bg = ((r + c) % 2 == 0)
                ? (Color){40, 60, 40, 255}
                : (Color){50, 75, 50, 255};
            DrawRectangle(margin + c * play_w / 10, margin + r * play_h / 10,
                          play_w / 10 + 1, play_h / 10 + 1, bg);
        }
    }

    DrawCircle(HOME_X * play_w + margin, HOME_Y * play_h + margin,
               HOME_RADIUS, RED);

    for (int i = 0; i < env->mine_cnt; i++) {
        int mx = (int)(env->mines[i].x * play_w + margin);
        int my = (int)(env->mines[i].y * play_h + margin);
        DrawCircle(mx, my, 20, GOLD);
        DrawCircle(mx - 4, my - 4, 8, (Color){255, 215, 0, 255});
        DrawCircle(mx + 4, my + 4, 8, (Color){220, 180, 0, 255});
    }

    int cx = (int)(env->pos_x * play_w + margin);
    int cy = (int)(env->pos_y * play_h + margin);
    float angle_rad = env->angle * M_PI / 180.0f;
    Vector2 tip = {(float)(cx + 15 * cosf(angle_rad)), (float)(cy + 15 * sinf(angle_rad))};
    Vector2 left = {(float)(cx + 10 * cosf(angle_rad + 2.5f)), (float)(cy + 10 * sinf(angle_rad + 2.5f))};
    Vector2 right = {(float)(cx + 10 * cosf(angle_rad - 2.5f)), (float)(cy + 10 * sinf(angle_rad - 2.5f))};
    DrawTriangle(tip, left, right, SKYBLUE);

    char info[128];
    snprintf(info, sizeof(info), "Speed: %.2f  Ore: %.1f/%.1f  %.1f/%.1f",
             env->speed, env->content[0], env->capacity, env->content[1], env->capacity);
    DrawText(info, 10, CELL_SIZE - 30, 20, WHITE);

    EndDrawing();
}

void c_close(Minecart* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (env->mines) {
        free(env->mines);
        env->mines = NULL;
    }
}
