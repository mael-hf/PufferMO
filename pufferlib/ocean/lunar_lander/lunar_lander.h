#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>   /* FIX 1 : RNG GSL pour reproduire NumPy PCG64 */
#include "raylib.h"

/* ── Couleurs PufferLib ── */
const Color PUFF_RED        = (Color){187,   0,   0, 255};
const Color PUFF_CYAN       = (Color){  0, 187, 187, 255};
const Color PUFF_WHITE      = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){  6,  24,  24, 255};
const Color PUFF_GREEN      = (Color){  0, 187,   0, 255};
const Color PUFF_YELLOW     = (Color){187, 187,   0, 255};
const Color PUFF_GRAY       = (Color){ 80,  80,  80, 255};
const Color PUFF_PURPLE     = (Color){128, 102, 230, 255};

/* ═══════════════════════════════════════════════════════════════════
 * Physical constants — identical to Python Gymnasium LunarLander-v3
 *
 * Action order (as in Python):
 *   0 = nothing
 *   1 = left orientation engine
 *   2 = main engine
 *   3 = right orientation engine
 * ═══════════════════════════════════════════════════════════════════ */
#define FPS                    50
#define DT                     (1.0f / FPS)

#define VIEWPORT_W             600.0f
#define VIEWPORT_H             400.0f
#define SCALE                  30.0f
#define W                      (VIEWPORT_W / SCALE)
#define H                      (VIEWPORT_H / SCALE)

#define MAIN_ENGINE_POWER      13.0f
#define SIDE_ENGINE_POWER       0.6f
#define SIDE_ENGINE_HEIGHT     14.0f
#define SIDE_ENGINE_AWAY       12.0f
#define MAIN_ENGINE_Y_LOCATION  4.0f

/* Legs — in Box2D units (pixels / SCALE) */
#define LEG_AWAY_PX            20.0f
#define LEG_DOWN_PX            18.0f
#define LEG_AWAY               (LEG_AWAY_PX / SCALE)
#define LEG_DOWN               (LEG_DOWN_PX / SCALE)

/* Gravity and initial random force */
#define GRAVITY               -10.0f
#define INITIAL_RANDOM       1000.0f

/* Terrain */
#define CHUNKS                 11
#define HELIPAD_Y              (H / 4.0f)

/* Observation / action spaces */
#define OBS_DIM                 8
#define ACT_DIM                 4

/* Wind constant (same as Python: k = 0.01) */
#define WIND_K                  0.01f

/* ══════════════════════════════════════════════════════════════════
 * Multi-objective reward dimensions
 * ══════════════════════════════════════════════════════════════════ */
#define REWARD_DIM              2

/* Per-objective reward indices */
#define REW_LANDING             0   /* shaping + landing bonus / crash penalty */
#define REW_FUEL                1   /* fuel efficiency (negated engine usage)  */

/* ══════════════════════════════════════════════════════════════════
 * Log — PufferLib required: floats only, n must be last
 * ══════════════════════════════════════════════════════════════════ */
typedef struct {
    float score_landing;  /* cumulative landing reward across episodes */
    float score_fuel;     /* cumulative fuel reward across episodes     */
    float episode_len;    /* cumulative episode length                  */
    float n;              /* number of completed episodes — MUST BE LAST */
} Log;

/* ══════════════════════════════════════════════════════════════════
 * LunarLander — main environment struct
 * Required PufferLib MO fields (in this exact order):
 *   Log log, observations, actions, rewards, weights, terminals, gsl_rng
 * ══════════════════════════════════════════════════════════════════ */
typedef struct {
    /* ── Required PufferLib MO fields ── */
    Log            log;
    float*         observations;   /* float[OBS_DIM]       */
    int*           actions;        /* int[1], 0..3         */
    float*         rewards;        /* float[REWARD_DIM]    */
    float*         weights;        /* float[REWARD_DIM] — scalarisation weights */
    unsigned char* terminals;      /* bool[1]              */
    void*          gsl_rng;        /* RNG handle (MO binding) */
    int            manual_weights; /* 1 if weights set externally */

    /* ── Lander dynamics ── */
    float x, y;       /* centre-of-mass position [m]   */
    float vx, vy;     /* linear velocities [m/s]       */
    float angle;      /* angle [rad]                   */
    float omega;      /* angular velocity [rad/s]      */

    /* ── Legs ── */
    int   leg_contact[2];

    /* ── Terrain ── */
    float terrain_y[CHUNKS + 1];  /* height at each chunk vertex [m] */
    float helipad_x1, helipad_x2; /* helipad x bounds [m]            */
    float helipad_y;               /* helipad height [m]              */

    /* ── Internal tracking ── */
    float prev_shaping;   /* shaping from previous step (NaN = first step) */
    int   step_count;
    int   game_over;

    /* ── Wind ── */
    int   enable_wind;
    float wind_power;
    float turbulence_power;
    float wind_idx;
    float torque_idx;

    /* ── Episode accumulation ── */
    float ep_reward_landing;
    float ep_reward_fuel;
    int   ep_len;
} LunarLander;

/* ══════════════════════════════════════════════════════════════════
 * Internal utilities
 * ══════════════════════════════════════════════════════════════════
 *
 * FIX 1 : ll_randf() et ll_randrange() utilisaient rand() (libc), dont
 * la séquence est incompatible avec le NumPy PCG64 seedé par Python.
 * On utilise maintenant le gsl_rng fourni par le binding PufferLib MO,
 * qui est seedé depuis Python via vec_reset(seed).  Toutes les fonctions
 * aléatoires reçoivent env* pour accéder à env->gsl_rng.
 * ══════════════════════════════════════════════════════════════════ */

/* FIX 1a — remplace ll_randf() sans contexte */
static inline float ll_randf_env(LunarLander* env) {
    return (float)gsl_rng_uniform((gsl_rng*)env->gsl_rng);
}

/* FIX 1b — remplace ll_randrange() sans contexte */
static inline float ll_randrange_env(LunarLander* env, float range) {
    return (ll_randf_env(env) * 2.0f - 1.0f) * range;
}

static float ll_ground_at(const LunarLander* env, float x) {
    float cw  = W / (CHUNKS - 1);
    int   idx = (int)(x / cw);
    if (idx < 0)           idx = 0;
    if (idx >= CHUNKS - 1) idx = CHUNKS - 2;
    float t = (x - idx * cw) / cw;
    return env->terrain_y[idx] * (1.0f - t) + env->terrain_y[idx + 1] * t;
}

/* FIX 1c — terrain_gen passe par ll_randf_env */
static void ll_terrain_gen(LunarLander* env) {
    float height[CHUNKS + 1];
    for (int i = 0; i <= CHUNKS; i++)
        height[i] = ll_randf_env(env) * (H / 2.0f);   /* FIX 1 */

    int mid = CHUNKS / 2;
    height[mid - 2] = HELIPAD_Y;
    height[mid - 1] = HELIPAD_Y;
    height[mid    ] = HELIPAD_Y;
    height[mid + 1] = HELIPAD_Y;
    height[mid + 2] = HELIPAD_Y;

    for (int i = 0; i < CHUNKS; i++) {
        float prev  = (i > 0)          ? height[i - 1] : height[i];
        float next  = (i < CHUNKS - 1) ? height[i + 1] : height[i];
        env->terrain_y[i] = (prev + height[i] + next) / 3.0f;
    }
    env->terrain_y[CHUNKS] = env->terrain_y[CHUNKS - 1];

    env->terrain_y[mid - 2] = HELIPAD_Y;
    env->terrain_y[mid - 1] = HELIPAD_Y;
    env->terrain_y[mid    ] = HELIPAD_Y;
    env->terrain_y[mid + 1] = HELIPAD_Y;
    env->terrain_y[mid + 2] = HELIPAD_Y;

    float cw = W / (CHUNKS - 1);
    env->helipad_x1 = cw * (mid - 1);
    env->helipad_x2 = cw * (mid + 1);
    env->helipad_y  = HELIPAD_Y;
}

static void ll_fill_obs(LunarLander* env) {
    env->observations[0] = (env->x - W / 2.0f) / (W / 2.0f);
    env->observations[1] = (env->y - (env->helipad_y + LEG_DOWN)) / (H / 2.0f);
    env->observations[2] = env->vx * (W / 2.0f) / FPS;
    env->observations[3] = env->vy * (H / 2.0f) / FPS;
    env->observations[4] = env->angle;
    env->observations[5] = 20.0f * env->omega / FPS;
    env->observations[6] = (float)env->leg_contact[0];
    env->observations[7] = (float)env->leg_contact[1];
}

/* ══════════════════════════════════════════════════════════════════
 * Dirichlet weight sampling (if not using manual weights)
 * ══════════════════════════════════════════════════════════════════ */
/* FIX 1d — sample_weights passe par ll_randf_env */
static void ll_sample_weights(LunarLander* env) {
    if (env->manual_weights) return;
    /* Sample from Dirichlet(1,1) = Uniform on simplex */
    float e0 = -logf(ll_randf_env(env) + 1e-8f);   /* FIX 1 */
    float e1 = -logf(ll_randf_env(env) + 1e-8f);   /* FIX 1 */
    float s  = e0 + e1;
    env->weights[REW_LANDING] = e0 / s;
    env->weights[REW_FUEL]    = e1 / s;
}

/* ══════════════════════════════════════════════════════════════════
 * ll_compute_shaping — helper partagé entre c_reset et c_step
 * ══════════════════════════════════════════════════════════════════ */
static inline float ll_compute_shaping(LunarLander* env) {
    float s0 = env->observations[0];
    float s1 = env->observations[1];
    float s2 = env->observations[2];
    float s3 = env->observations[3];
    float s4 = env->observations[4];
    float s6 = env->observations[6];
    float s7 = env->observations[7];
    return -100.0f * sqrtf(s0*s0 + s1*s1)
           -100.0f * sqrtf(s2*s2 + s3*s3)
           -100.0f * fabsf(s4)
           +  10.0f * s6
           +  10.0f * s7;
}

/* ══════════════════════════════════════════════════════════════════
 * c_reset
 * ══════════════════════════════════════════════════════════════════ */
void c_reset(LunarLander* env) {
    ll_terrain_gen(env);

    env->x     = W / 2.0f;
    env->y     = H;
    env->angle = 0.0f;
    env->omega = 0.0f;

    float mass_approx = 5.0f;
    env->vx = ll_randrange_env(env, INITIAL_RANDOM) / (mass_approx * FPS);   /* FIX 1 */
    env->vy = ll_randrange_env(env, INITIAL_RANDOM) / (mass_approx * FPS);   /* FIX 1 */

    env->leg_contact[0]    = 0;
    env->leg_contact[1]    = 0;
    env->game_over         = 0;
    env->step_count        = 0;
    env->ep_reward_landing = 0.0f;
    env->ep_reward_fuel    = 0.0f;
    env->ep_len            = 0;

    if (env->enable_wind) {
        env->wind_idx   = ll_randrange_env(env, 9999.0f);   /* FIX 1 */
        env->torque_idx = ll_randrange_env(env, 9999.0f);   /* FIX 1 */
    }

    /* Sample new Dirichlet weights at episode start */
    ll_sample_weights(env);

    ll_fill_obs(env);

    /* FIX 2 : initialiser prev_shaping depuis la position de reset,
     * exactement comme Gymnasium le fait dans reset().
     * Sans ce calcul, le step 0 retourne toujours rew_landing = 0
     * au lieu du delta shaping_step0 - shaping_reset. */
    env->prev_shaping = ll_compute_shaping(env);
}

/* ══════════════════════════════════════════════════════════════════
 * c_step — MO version
 *
 * Writes each reward objective separately into rewards[]:
 *   rewards[REW_LANDING] = shaping delta + landing/crash bonus
 *   rewards[REW_FUEL]    = -fuel_cost (positive = efficient)
 * ══════════════════════════════════════════════════════════════════ */
void c_step(LunarLander* env) {
    int action = env->actions[0];

    /* Zero all reward objectives */
    for (int i = 0; i < REWARD_DIM; i++)
        env->rewards[i] = 0.0f;

    /* ── 1. Wind ── */
    if (env->enable_wind && !(env->leg_contact[0] || env->leg_contact[1])) {
        float wind_mag = tanhf(
            sinf(0.02f * env->wind_idx) + sinf((float)M_PI * 0.01f * env->wind_idx)
        ) * env->wind_power;
        env->wind_idx += 1.0f;

        float torque_mag = tanhf(
            sinf(0.02f * env->torque_idx) + sinf((float)M_PI * 0.01f * env->torque_idx)
        ) * env->turbulence_power;
        env->torque_idx += 1.0f;

        env->vx    += (wind_mag / 5.0f) * DT;
        env->omega += (torque_mag / 5.0f) * DT;
    }

    /* ── 2. Engine impulses ── */
    float tip_x  =  sinf(env->angle);
    float tip_y  =  cosf(env->angle);
    float side_x = -tip_y;
    float side_y =  tip_x;

    float disp0 = ll_randrange_env(env, 1.0f / SCALE);   /* FIX 1 */
    float disp1 = ll_randrange_env(env, 1.0f / SCALE);   /* FIX 1 */

    float ax = 0.0f, ay = 0.0f, atorque = 0.0f;
    float m_power = 0.0f, s_power = 0.0f;

    if (action == 2) {
        m_power = 1.0f;
        float ox = tip_x  * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0f * disp0) + side_x * disp1;
        float oy = -tip_y * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0f * disp0) - side_y * disp1;
        ax += -ox * MAIN_ENGINE_POWER / 5.0f;
        ay += -oy * MAIN_ENGINE_POWER / 5.0f;
    }

    if (action == 1 || action == 3) {
        s_power = 1.0f;
        float direction = (action == 1) ? -1.0f : 1.0f;
        float ox = tip_x  * disp0 + side_x * (3.0f * disp1 + direction * SIDE_ENGINE_AWAY / SCALE);
        float oy = -tip_y * disp0 - side_y * (3.0f * disp1 + direction * SIDE_ENGINE_AWAY / SCALE);
        ax += -ox * SIDE_ENGINE_POWER / 5.0f;
        ay += -oy * SIDE_ENGINE_POWER / 5.0f;
        atorque += direction * SIDE_ENGINE_POWER * (SIDE_ENGINE_AWAY / SCALE) / 5.0f;
    }

    /* ── 3. Euler integration ── */
    env->vx    += ax              * DT;
    env->vy    += (ay + GRAVITY)  * DT;
    env->omega += atorque         * DT * 20.0f;

    env->vx    *= 0.999f;
    env->vy    *= 0.999f;
    env->omega *= 0.997f;

    env->x     += env->vx    * DT;
    env->y     += env->vy    * DT;
    env->angle += env->omega * DT;

    while (env->angle >  (float)M_PI) env->angle -= 2.0f * (float)M_PI;
    while (env->angle < -(float)M_PI) env->angle += 2.0f * (float)M_PI;

    /* ── 4. Leg contact detection ── */
    float sa = sinf(env->angle), ca = cosf(env->angle);
    for (int leg = 0; leg < 2; leg++) {
        float sign = (leg == 0) ? -1.0f : 1.0f;
        float lx = env->x + sign * LEG_AWAY * ca - LEG_DOWN * sa;
        float ly = env->y + sign * LEG_AWAY * sa + LEG_DOWN * ca;
        float gy = ll_ground_at(env, lx);
        env->leg_contact[leg] = (ly <= gy + 0.05f) ? 1 : 0;
    }

    float gy_body = ll_ground_at(env, env->x);
    if (env->y <= gy_body + 0.05f)
        env->game_over = 1;

    /* ── 5. Fill observations ── */
    ll_fill_obs(env);

    /* ── 6. Shaping + multi-objective rewards ── */
    float shaping = ll_compute_shaping(env);

    float rew_landing = 0.0f;
    float rew_fuel    = 0.0f;

    /* Landing reward: potential-based shaping delta
     * FIX 2 : prev_shaping est maintenant toujours un nombre valide
     * dès le step 0 (initialisé dans c_reset), donc le isnan guard
     * ne masque plus la récompense du premier step. */
    rew_landing = shaping - env->prev_shaping;
    env->prev_shaping = shaping;

    /* Fuel reward: negative cost (0 = no engine used = best) */
    rew_fuel = -(m_power * 0.30f + s_power * 0.03f);

    /* ── Termination ── */
    env->terminals[0] = 0;
    env->step_count++;

    float s0   = env->observations[0];
    int out_x  = (fabsf(s0) >= 1.0f);
    int awake  = !(env->leg_contact[0] && env->leg_contact[1]
                   && fabsf(env->vx) < 0.1f && fabsf(env->vy) < 0.1f
                   && fabsf(env->omega) < 0.1f);

    if (env->game_over || out_x) {
        env->terminals[0] = 1;
        rew_landing = -100.0f;
    } else if (!awake) {
        env->terminals[0] = 1;
        rew_landing = +100.0f;
    }

    /* Write objectives into rewards array */
    env->rewards[REW_LANDING] = rew_landing;
    env->rewards[REW_FUEL]    = rew_fuel;

    /* Accumulate per-objective episode totals */
    env->ep_reward_landing += rew_landing;
    env->ep_reward_fuel    += rew_fuel;
    env->ep_len++;

    /* Log + auto-reset at episode end */
    if (env->terminals[0]) {
        env->log.score_landing += env->ep_reward_landing;
        env->log.score_fuel    += env->ep_reward_fuel;
        env->log.episode_len   += (float)env->ep_len;
        env->log.n             += 1.0f;
        c_reset(env);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * c_render
 * ══════════════════════════════════════════════════════════════════ */
void c_render(LunarLander* env) {
    if (!IsWindowReady()) {
        InitWindow((int)VIEWPORT_W, (int)VIEWPORT_H, "PufferLib — LunarLander MO");
        SetTargetFPS(FPS);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(BLACK);

    float cw = W / (CHUNKS - 1);

    /* ── Terrain ── */
    for (int i = 0; i < CHUNKS - 1; i++) {
        float x0 = i       * cw * SCALE;
        float x1 = (i + 1) * cw * SCALE;
        float y0 = VIEWPORT_H - env->terrain_y[i]     * SCALE;
        float y1 = VIEWPORT_H - env->terrain_y[i + 1] * SCALE;

        int mid = CHUNKS / 2;
        Color col = (i >= mid - 2 && i <= mid + 1) ? PUFF_YELLOW : PUFF_GRAY;

        DrawTriangle(
            (Vector2){x0, y0},
            (Vector2){x1, y1},
            (Vector2){x1, VIEWPORT_H},
            col
        );
        DrawTriangle(
            (Vector2){x0, y0},
            (Vector2){x1, VIEWPORT_H},
            (Vector2){x0, VIEWPORT_H},
            col
        );
        DrawLine((int)x0, (int)y0, (int)x1, (int)y1, PUFF_WHITE);
    }

    /* ── Helipad flags ── */
    float flag_xs[2] = { env->helipad_x1 * SCALE, env->helipad_x2 * SCALE };
    for (int f = 0; f < 2; f++) {
        float fx     = flag_xs[f];
        float flagy1 = VIEWPORT_H - env->helipad_y * SCALE;
        float flagy2 = flagy1 - 50.0f;

        DrawLine((int)fx, (int)flagy1, (int)fx, (int)flagy2, PUFF_WHITE);
        DrawTriangle(
            (Vector2){fx,         flagy2},
            (Vector2){fx,         flagy2 + 10.0f},
            (Vector2){fx + 25.0f, flagy2 + 5.0f},
            PUFF_YELLOW
        );
    }

    /* ── Legs ── */
    float sa = sinf(env->angle), ca = cosf(env->angle);
    int bx = (int)(env->x * SCALE);
    int by = (int)(VIEWPORT_H - env->y * SCALE);
    for (int leg = 0; leg < 2; leg++) {
        float sign = (leg == 0) ? -1.0f : 1.0f;
        float lx = env->x + sign * LEG_AWAY * ca - LEG_DOWN * sa;
        float ly = env->y + sign * LEG_AWAY * sa + LEG_DOWN * ca;
        int sx = (int)(lx * SCALE);
        int sy = (int)(VIEWPORT_H - ly * SCALE);
        Color leg_col = env->leg_contact[leg] ? PUFF_GREEN : PUFF_PURPLE;
        DrawLine(bx, by, sx, sy, leg_col);
        DrawCircle(sx, sy, 3, leg_col);
    }

    /* ── Lander body ── */
    float poly_local[6][2] = {
        {-14, +17}, {-17, 0}, {-17, -10}, {+17, -10}, {+17, 0}, {+14, +17}
    };
    Vector2 poly_world[6];
    for (int v = 0; v < 6; v++) {
        float px = poly_local[v][0] / SCALE;
        float py = poly_local[v][1] / SCALE;
        float rx = px * ca - py * sa;
        float ry = px * sa + py * ca;
        poly_world[v].x = env->x * SCALE + rx * SCALE;
        poly_world[v].y = VIEWPORT_H - (env->y * SCALE + ry * SCALE);
    }
    Vector2 center = {(float)bx, (float)by};
    for (int v = 0; v < 6; v++) {
        DrawTriangle(center, poly_world[v], poly_world[(v + 1) % 6], PUFF_PURPLE);
    }
    for (int v = 0; v < 6; v++) {
        DrawLine(
            (int)poly_world[v].x,       (int)poly_world[v].y,
            (int)poly_world[(v+1)%6].x, (int)poly_world[(v+1)%6].y,
            PUFF_CYAN
        );
    }

    /* ── Engine flames ── */
    float tip_x = sa;
    float tip_y = ca;

    if (env->actions[0] == 2) {
        float fx = (float)bx - tip_x * MAIN_ENGINE_Y_LOCATION;
        float fy = (float)by + tip_y * MAIN_ENGINE_Y_LOCATION;
        DrawCircle((int)fx, (int)fy, 10, PUFF_RED);
        DrawCircle((int)fx, (int)fy,  6, PUFF_YELLOW);
    }

    if (env->actions[0] == 1 || env->actions[0] == 3) {
        float direction = (env->actions[0] == 1) ? -1.0f : 1.0f;
        float fx = (float)bx + direction * (-ca) * SIDE_ENGINE_AWAY * 0.5f;
        float fy = (float)by - direction * ( sa)  * SIDE_ENGINE_AWAY * 0.5f;
        DrawCircle((int)fx, (int)fy, 6, PUFF_RED);
        DrawCircle((int)fx, (int)fy, 4, PUFF_YELLOW);
    }

    /* ── HUD ── */
    int hud_y = 8;
    DrawText(TextFormat("Landing rew: %.1f",  env->ep_reward_landing),   8, hud_y, 16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Fuel rew   : %.1f",  env->ep_reward_fuel),      8, hud_y, 16, PUFF_CYAN);  hud_y += 20;
    DrawText(TextFormat("W(land/fuel): %.2f/%.2f",
             env->weights[REW_LANDING], env->weights[REW_FUEL]),          8, hud_y, 16, PUFF_YELLOW); hud_y += 20;
    DrawText(TextFormat("Steps      : %d",    env->step_count),           8, hud_y, 16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Action     : %d",    env->actions[0]),           8, hud_y, 16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Vel X/Y    : %.2f / %.2f", env->vx, env->vy),   8, hud_y, 16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Angle      : %.2f rad",    env->angle),          8, hud_y, 16, PUFF_WHITE); hud_y += 20;
    if (env->enable_wind)
        DrawText(TextFormat("Wind idx   : %.0f", env->wind_idx),          8, hud_y, 16, PUFF_CYAN);
    if (env->leg_contact[0] && env->leg_contact[1])
        DrawText("LANDED!", (int)VIEWPORT_W / 2 - 40, 12, 24, PUFF_GREEN);
    if (env->game_over && !(env->leg_contact[0] && env->leg_contact[1]))
        DrawText("CRASH!", (int)VIEWPORT_W / 2 - 35, 12, 24, PUFF_RED);

    EndDrawing();
}

/* ══════════════════════════════════════════════════════════════════
 * c_close
 * ══════════════════════════════════════════════════════════════════ */
void c_close(LunarLander* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#ifndef BINDING_C_LOADED
#include "../env_binding_mo.h"
#endif