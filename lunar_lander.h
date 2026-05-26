#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

/* ── Couleurs PufferLib ── */
const Color PUFF_RED        = (Color){187,   0,   0, 255};
const Color PUFF_CYAN       = (Color){  0, 187, 187, 255};
const Color PUFF_WHITE      = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){  6,  24,  24, 255};
const Color PUFF_GREEN      = (Color){  0, 187,   0, 255};
const Color PUFF_YELLOW     = (Color){187, 187,   0, 255};
const Color PUFF_GRAY       = (Color){ 80,  80,  80, 255};

/* ═══════════════════════════════════════════════════════════════════
 * Constantes physiques (identiques à Gymnasium LunarLander-v2)
 * ═══════════════════════════════════════════════════════════════════ */
#define FPS                  50
#define DT                   (1.0f / FPS)

#define VIEWPORT_W           600.0f
#define VIEWPORT_H           400.0f
#define SCALE                30.0f
#define W                    (VIEWPORT_W / SCALE)   /* ≈ 20 m */
#define H                    (VIEWPORT_H / SCALE)   /* ≈ 13.3 m */

#define MAIN_ENGINE_POWER    13.0f
#define SIDE_ENGINE_POWER     0.6f
#define SIDE_ENGINE_HEIGHT   14.0f
#define SIDE_ENGINE_AWAY     12.0f
#define LANDER_DENSITY        5.0f
#define LANDER_FRICTION       0.1f
#define LEG_AWAY              0.2f
#define LEG_DOWN              0.32f

#define GRAVITY              -9.8f

/* Terrain */
#define CHUNKS               11
#define HELIPAD_Y            (H / 4.0f)

/* Espace obs/action */
#define OBS_DIM               8
#define ACT_DIM               4

/* ══════════════════════════════════════════════════════════════════
 * Log — obligatoire, uniquement des floats, n en dernier champ
 * ══════════════════════════════════════════════════════════════════ */
typedef struct {
    float score;      /* somme des récompenses d'épisode */
    float episode_len;/* longueur moyenne des épisodes   */
    float n;          /* nombre d'épisodes terminés (DOIT être le dernier champ) */
} Log;

/* ══════════════════════════════════════════════════════════════════
 * LunarLander — structure principale de l'environnement
 * ══════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Champs obligatoires PufferLib */
    Log            log;           /* stats épisodiques     */
    float*         observations;  /* float[OBS_DIM]        */
    int*           actions;       /* int[1] — action 0..3  */
    float*         rewards;       /* float[1]              */
    unsigned char* terminals;     /* bool[1]               */

    /* Dynamique du lander */
    float x, y;          /* position [m]              */
    float vx, vy;        /* vitesse [m/s]             */
    float angle;         /* [rad]                     */
    float omega;         /* vitesse angulaire [rad/s] */

    /* Jambes */
    int leg_contact[2];

    /* Terrain */
    float terrain_y[CHUNKS + 1];
    float helipad_y;

    /* Suivi interne */
    float prev_shaping;
    int   step_count;

    /* Accumulation épisodique (pour le log) */
    float ep_reward;
    int   ep_len;
} LunarLander;

/* ══════════════════════════════════════════════════════════════════
 * Utilitaires internes
 * ══════════════════════════════════════════════════════════════════ */

static inline float ll_randf(void) {
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

static float ll_ground_at(const LunarLander* env, float x) {
    float cw  = W / CHUNKS;
    int   idx = (int)(x / cw);
    if (idx < 0)       idx = 0;
    if (idx >= CHUNKS) idx = CHUNKS - 1;
    float t = (x - idx * cw) / cw;
    return env->terrain_y[idx] * (1.0f - t) + env->terrain_y[idx + 1] * t;
}

static void ll_terrain_gen(LunarLander* env) {
    float raw[CHUNKS + 1];
    for (int i = 0; i <= CHUNKS; i++)
        raw[i] = HELIPAD_Y + (ll_randf() - 0.5f) * H / 2.0f;

    int pl = CHUNKS / 2 - 1, pr = CHUNKS / 2 + 1;
    for (int i = pl; i <= pr; i++) raw[i] = HELIPAD_Y;

    for (int pass = 0; pass < 3; pass++) {
        for (int i = 1; i < CHUNKS; i++)
            raw[i] = (raw[i-1] + raw[i] + raw[i+1]) / 3.0f;
        for (int i = pl; i <= pr; i++) raw[i] = HELIPAD_Y;
    }

    for (int i = 0; i <= CHUNKS; i++) env->terrain_y[i] = raw[i];
    env->helipad_y = HELIPAD_Y;
}

/* ══════════════════════════════════════════════════════════════════
 * c_reset
 * ══════════════════════════════════════════════════════════════════ */
void c_reset(LunarLander* env) {
    ll_terrain_gen(env);

    env->x     = W / 2.0f + (ll_randf() - 0.5f) * W / 3.0f;
    env->y     = H * 0.75f + ll_randf() * H * 0.1f;
    env->vx    = (ll_randf() - 0.5f) * 2.0f;
    env->vy    = -(ll_randf() * 0.5f);
    env->angle = (ll_randf() - 0.5f) * 0.4f;
    env->omega = (ll_randf() - 0.5f) * 0.4f;

    env->leg_contact[0] = 0;
    env->leg_contact[1] = 0;
    env->prev_shaping   = 0.0f;
    env->step_count     = 0;
    env->ep_reward      = 0.0f;
    env->ep_len         = 0;

    /* Calcul du shaping initial pour éviter un spike de récompense au 1er step */
    float dist_x = env->x - W / 2.0f;
    float dist_y = env->y - env->helipad_y;
    env->prev_shaping =
        -100.0f * sqrtf(dist_x * dist_x + dist_y * dist_y)
        -100.0f * sqrtf(env->vx * env->vx + env->vy * env->vy)
        -100.0f * fabsf(env->angle);

    /* Remplissage de l'observation initiale */
    env->observations[0] = (env->x - W / 2.0f)   / (W / 2.0f);
    env->observations[1] = (env->y - HELIPAD_Y)   / (H / 2.0f);
    env->observations[2] = env->vx * (VIEWPORT_W / SCALE / 2.0f) / FPS;
    env->observations[3] = env->vy * (VIEWPORT_H / SCALE / 2.0f) / FPS;
    env->observations[4] = env->angle;
    env->observations[5] = env->omega * 20.0f / FPS;
    env->observations[6] = 0.0f;
    env->observations[7] = 0.0f;
}

/* ══════════════════════════════════════════════════════════════════
 * c_step
 * ══════════════════════════════════════════════════════════════════ */
void c_step(LunarLander* env) {
    int action = env->actions[0];

    float sa = sinf(env->angle), ca = cosf(env->angle);
    float Fx = 0.0f, Fy = 0.0f, torque = 0.0f;

    if (action == 1) {
        Fx += -sa * MAIN_ENGINE_POWER;
        Fy +=  ca * MAIN_ENGINE_POWER;
    }
    if (action == 2) {
        Fx +=  ca * SIDE_ENGINE_POWER;
        Fy +=  sa * SIDE_ENGINE_POWER;
        torque -= SIDE_ENGINE_POWER * 0.12f;
    }
    if (action == 3) {
        Fx += -ca * SIDE_ENGINE_POWER;
        Fy += -sa * SIDE_ENGINE_POWER;
        torque += SIDE_ENGINE_POWER * 0.12f;
    }

    /* Intégration Euler */
    env->vx    += Fx * DT;
    env->vy    += (Fy + GRAVITY) * DT;
    env->omega += torque * DT * 20.0f;

    /* Drag aérien */
    env->vx    *= 0.999f;
    env->vy    *= 0.999f;
    env->omega *= 0.997f;

    env->x     += env->vx    * DT;
    env->y     += env->vy    * DT;
    env->angle += env->omega * DT;

    /* Normalisation angle */
    while (env->angle >  (float)M_PI) env->angle -= 2.0f * (float)M_PI;
    while (env->angle < -(float)M_PI) env->angle += 2.0f * (float)M_PI;

    /* ── Contact jambes ── */
    float sa2 = sinf(env->angle), ca2 = cosf(env->angle);
    for (int leg = 0; leg < 2; leg++) {
        float sign = (leg == 0) ? -1.0f : 1.0f;
        float lx = env->x + sign * LEG_AWAY * ca2 - LEG_DOWN * sa2;
        float ly = env->y + sign * LEG_AWAY * sa2 + LEG_DOWN * ca2;
        float gy = ll_ground_at(env, lx);
        env->leg_contact[leg] = (ly <= gy + 0.05f) ? 1 : 0;
    }

    /* ── Récompense (identique Gymnasium) ── */
    float dist_x = env->x - W / 2.0f;
    float dist_y = env->y - env->helipad_y;

    float shaping =
        -100.0f * sqrtf(dist_x * dist_x + dist_y * dist_y)
        -100.0f * sqrtf(env->vx * env->vx + env->vy * env->vy)
        -100.0f * fabsf(env->angle)
        +  10.0f * env->leg_contact[0]
        +  10.0f * env->leg_contact[1];

    float reward = shaping - env->prev_shaping;
    env->prev_shaping = shaping;

    if (action == 1) reward -= 0.30f;
    if (action == 2 || action == 3) reward -= 0.03f;

    /* ── Terminaison ── */
    env->terminals[0] = 0;
    env->step_count++;

    float gy_body = ll_ground_at(env, env->x);
    int crashed = (env->y <= gy_body + 0.05f);
    int out_x   = (env->x < 0.0f || env->x > W);
    int out_top = (env->y > H);
    int timeout = (env->step_count >= 1000);

    if (crashed) {
        env->terminals[0] = 1;
        int good_land = env->leg_contact[0] && env->leg_contact[1]
                        && fabsf(env->vx) < 0.5f && fabsf(env->vy) < 0.5f
                        && fabsf(env->angle) < 0.25f;
        reward += good_land ? 200.0f : -100.0f;
    } else if (out_x || out_top || timeout) {
        env->terminals[0] = 1;
        reward -= 100.0f;
    }

    env->rewards[0] = reward;
    env->ep_reward += reward;
    env->ep_len++;

    /* ── Observation ── */
    env->observations[0] = (env->x - W / 2.0f)   / (W / 2.0f);
    env->observations[1] = (env->y - HELIPAD_Y)   / (H / 2.0f);
    env->observations[2] = env->vx * (VIEWPORT_W / SCALE / 2.0f) / FPS;
    env->observations[3] = env->vy * (VIEWPORT_H / SCALE / 2.0f) / FPS;
    env->observations[4] = env->angle;
    env->observations[5] = env->omega * 20.0f / FPS;
    env->observations[6] = (float)env->leg_contact[0];
    env->observations[7] = (float)env->leg_contact[1];

    /* ── Fin d'épisode : mise à jour du log et reset ── */
    if (env->terminals[0]) {
        env->log.score      += env->ep_reward;
        env->log.episode_len += (float)env->ep_len;
        env->log.n          += 1.0f;
        c_reset(env);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * c_render
 * ══════════════════════════════════════════════════════════════════ */
void c_render(LunarLander* env) {
    if (!IsWindowReady()) {
        InitWindow((int)VIEWPORT_W, (int)VIEWPORT_H, "PufferLib — LunarLander");
        SetTargetFPS(FPS);
    }

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    /* ── Terrain ── */
    float cw = VIEWPORT_W / CHUNKS;
    for (int i = 0; i < CHUNKS; i++) {
        float x0 = i * cw;
        float y0 = VIEWPORT_H - env->terrain_y[i]   * SCALE;
        float x1 = (i + 1) * cw;
        float y1 = VIEWPORT_H - env->terrain_y[i+1] * SCALE;

        /* Zone hélipad (chunks centraux) */
        Color col = (i == CHUNKS / 2 - 1 || i == CHUNKS / 2) ? PUFF_YELLOW : PUFF_GRAY;
        DrawLine((int)x0, (int)y0, (int)x1, (int)y1, col);
        /* Sol plein sous le terrain */
        DrawRectangle((int)x0, (int)y0, (int)cw + 1, (int)(VIEWPORT_H - y0), col);
    }

    /* ── Jambes ── */
    float sa = sinf(env->angle), ca = cosf(env->angle);
    for (int leg = 0; leg < 2; leg++) {
        float sign = (leg == 0) ? -1.0f : 1.0f;
        float lx = env->x + sign * LEG_AWAY * ca - LEG_DOWN * sa;
        float ly = env->y + sign * LEG_AWAY * sa + LEG_DOWN * ca;
        int sx = (int)(lx * SCALE);
        int sy = (int)(VIEWPORT_H - ly * SCALE);
        int bx = (int)(env->x * SCALE);
        int by = (int)(VIEWPORT_H - env->y * SCALE);
        Color leg_col = env->leg_contact[leg] ? PUFF_GREEN : PUFF_WHITE;
        DrawLine(bx, by, sx, sy, leg_col);
        DrawCircle(sx, sy, 4, leg_col);
    }

    /* ── Corps du lander (rectangle orienté) ── */
    float hw = 0.5f * SCALE;   /* demi-largeur en pixels */
    float hh = 0.4f * SCALE;   /* demi-hauteur en pixels */
    float cx = env->x * SCALE;
    float cy = VIEWPORT_H - env->y * SCALE;

    Vector2 origin = {hw, hh};
    Rectangle rect = {cx - hw, cy - hh, hw * 2.0f, hh * 2.0f};
    DrawRectanglePro(rect, origin, -env->angle * (180.0f / (float)M_PI), PUFF_CYAN);

    /* ── Flamme moteur principal ── */
    if (env->actions[0] == 1) {
        float fx = cx - sa * hh * 2.0f;
        float fy = cy + ca * hh * 2.0f;
        DrawCircle((int)fx, (int)fy, 8, PUFF_RED);
    }
    /* ── Flamme propulseurs latéraux ── */
    if (env->actions[0] == 2 || env->actions[0] == 3) {
        float sign = (env->actions[0] == 2) ? -1.0f : 1.0f;
        float fx = cx + sign * ca * hw * 1.5f;
        float fy = cy + sign * sa * hw * 1.5f;
        DrawCircle((int)fx, (int)fy, 5, PUFF_YELLOW);
    }

    /* ── HUD ── */
    DrawText(TextFormat("Score ep : %.1f", env->ep_reward),   8,  8, 18, PUFF_WHITE);
    DrawText(TextFormat("Steps    : %d",   env->step_count),  8, 30, 18, PUFF_WHITE);
    DrawText(TextFormat("Action   : %d",   env->actions[0]),  8, 52, 18, PUFF_WHITE);
    DrawText(TextFormat("Vel X/Y  : %.2f / %.2f", env->vx, env->vy), 8, 74, 18, PUFF_WHITE);

    /* Indicateur atterrissage */
    if (env->leg_contact[0] && env->leg_contact[1])
        DrawText("CONTACT!", (int)VIEWPORT_W/2 - 50, 16, 24, PUFF_GREEN);

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
