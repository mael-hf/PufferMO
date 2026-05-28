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
const Color PUFF_PURPLE     = (Color){128, 102, 230, 255};

/* ═══════════════════════════════════════════════════════════════════
 * Constantes physiques — identiques au Python Gymnasium LunarLander-v3
 *
 * Notes d'ordre des actions (comme dans le Python) :
 *   0 = rien
 *   1 = propulseur gauche  (left orientation engine)
 *   2 = moteur principal   (main engine)
 *   3 = propulseur droit   (right orientation engine)
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
#define MAIN_ENGINE_Y_LOCATION  4.0f   /* position Y du moteur sur le corps */

/* Jambes — en unités Box2D (pixels / SCALE) */
#define LEG_AWAY_PX            20.0f
#define LEG_DOWN_PX            18.0f
#define LEG_AWAY               (LEG_AWAY_PX / SCALE)
#define LEG_DOWN               (LEG_DOWN_PX / SCALE)

/* Gravité et force initiale aléatoire */
#define GRAVITY               -10.0f
#define INITIAL_RANDOM       1000.0f

/* Terrain */
#define CHUNKS                 11
#define HELIPAD_Y              (H / 4.0f)

/* Espace obs/action */
#define OBS_DIM                 8
#define ACT_DIM                 4

/* Constante vent (identique Python : k = 0.01) */
#define WIND_K                  0.01f

/* ══════════════════════════════════════════════════════════════════
 * Log — obligatoire PufferLib, uniquement des floats, n en dernier
 * ══════════════════════════════════════════════════════════════════ */
typedef struct {
    float score;       /* somme des récompenses d'épisode            */
    float episode_len; /* longueur cumulée des épisodes              */
    float n;           /* nb épisodes terminés — DOIT être le dernier */
} Log;

/* ══════════════════════════════════════════════════════════════════
 * LunarLander — structure principale
 * ══════════════════════════════════════════════════════════════════ */
typedef struct {
    /* ── Champs obligatoires PufferLib (dans cet ordre) ── */
    Log            log;
    float*         observations;   /* float[OBS_DIM]  */
    int*           actions;        /* int[1], 0..3    */
    float*         rewards;        /* float[1]        */
    unsigned char* terminals;      /* bool[1]         */

    /* ── Dynamique du lander ── */
    float x, y;       /* position centre de masse [m]     */
    float vx, vy;     /* vitesses linéaires [m/s]         */
    float angle;      /* angle [rad]                      */
    float omega;      /* vitesse angulaire [rad/s]        */

    /* ── Jambes ── */
    int   leg_contact[2];

    /* ── Terrain ── */
    float terrain_y[CHUNKS + 1];  /* hauteur de chaque sommet de chunk [m] */
    float helipad_x1, helipad_x2; /* bornes x de l'hélipad [m]             */
    float helipad_y;               /* hauteur hélipad [m]                   */

    /* ── Suivi interne ── */
    float prev_shaping;   /* shaping du step précédent (NaN = premier step) */
    int   step_count;
    int   game_over;      /* 1 si le corps a touché le sol (comme Python)   */

    /* ── Vent (enable_wind) ── */
    int   enable_wind;
    float wind_power;
    float turbulence_power;
    float wind_idx;    /* compteur de phase vent      */
    float torque_idx;  /* compteur de phase turbulence */

    /* ── Accumulation épisodique ── */
    float ep_reward;
    int   ep_len;
} LunarLander;

/* ══════════════════════════════════════════════════════════════════
 * Utilitaires internes
 * ══════════════════════════════════════════════════════════════════ */

/* Uniforme [0, 1) */
static inline float ll_randf(void) {
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

/* Uniforme [-range, +range] */
static inline float ll_randrange(float range) {
    return (ll_randf() * 2.0f - 1.0f) * range;
}

/* Hauteur du sol au point x (interpolation linéaire entre chunks) */
static float ll_ground_at(const LunarLander* env, float x) {
    float cw  = W / (CHUNKS - 1);
    int   idx = (int)(x / cw);
    if (idx < 0)           idx = 0;
    if (idx >= CHUNKS - 1) idx = CHUNKS - 2;
    float t = (x - idx * cw) / cw;
    return env->terrain_y[idx] * (1.0f - t) + env->terrain_y[idx + 1] * t;
}

/* Génération du terrain — identique au Python Gymnasium */
static void ll_terrain_gen(LunarLander* env) {
    /*
     * Python :
     *   height = np_random.uniform(0, H/2, size=(CHUNKS+1,))
     *   chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
     *   helipad sur les 5 chunks centraux à HELIPAD_Y
     *   smooth_y = moyenne glissante à 3 voisins (1 passe, CHUNKS points)
     */
    float height[CHUNKS + 1];
    for (int i = 0; i <= CHUNKS; i++)
        height[i] = ll_randf() * (H / 2.0f);

    int mid = CHUNKS / 2;
    /* 5 chunks centraux à HELIPAD_Y */
    height[mid - 2] = HELIPAD_Y;
    height[mid - 1] = HELIPAD_Y;
    height[mid    ] = HELIPAD_Y;
    height[mid + 1] = HELIPAD_Y;
    height[mid + 2] = HELIPAD_Y;

    /* smooth_y : une seule passe, moyenne de 3 voisins */
    for (int i = 0; i < CHUNKS; i++) {
        float prev  = (i > 0)          ? height[i - 1] : height[i];
        float next  = (i < CHUNKS - 1) ? height[i + 1] : height[i];
        env->terrain_y[i] = (prev + height[i] + next) / 3.0f;
    }
    /* Dernier sommet = dernier chunk */
    env->terrain_y[CHUNKS] = env->terrain_y[CHUNKS - 1];

    /* Restaurer le plateau hélipad après lissage */
    env->terrain_y[mid - 2] = HELIPAD_Y;
    env->terrain_y[mid - 1] = HELIPAD_Y;
    env->terrain_y[mid    ] = HELIPAD_Y;
    env->terrain_y[mid + 1] = HELIPAD_Y;
    env->terrain_y[mid + 2] = HELIPAD_Y;

    /* Coordonnées x de l'hélipad */
    float cw = W / (CHUNKS - 1);
    env->helipad_x1 = cw * (mid - 1);
    env->helipad_x2 = cw * (mid + 1);
    env->helipad_y  = HELIPAD_Y;
}

/* ══════════════════════════════════════════════════════════════════
 * Remplissage du vecteur d'observation (identique Python v3)
 *
 * Python :
 *   state[0] = (pos.x - W/2)       / (W/2)
 *   state[1] = (pos.y - helipad_y - LEG_DOWN/SCALE) / (H/2)
 *   state[2] = vel.x * (W/2) / FPS
 *   state[3] = vel.y * (H/2) / FPS
 *   state[4] = lander.angle
 *   state[5] = 20 * lander.angularVelocity / FPS
 *   state[6] = 1 si jambe gauche contact
 *   state[7] = 1 si jambe droite contact
 * ══════════════════════════════════════════════════════════════════ */
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
 * c_reset
 *
 * Fidèle au Python :
 *   - Lander spawne en (W/2, H) avec angle=0
 *   - Impulsion aléatoire initiale dans [-INITIAL_RANDOM, +INITIAL_RANDOM]²
 *     divisée par FPS pour avoir des vitesses initiales (≈ ApplyForceToCenter)
 *   - prev_shaping = NaN (signale "premier step")
 *   - Indices de vent tirés aléatoirement dans [-9999, 9999]
 * ══════════════════════════════════════════════════════════════════ */
void c_reset(LunarLander* env) {
    ll_terrain_gen(env);

    /* Position initiale : centre haut de l'écran */
    env->x     = W / 2.0f;
    env->y     = H;          /* VIEWPORT_H / SCALE */
    env->angle = 0.0f;
    env->omega = 0.0f;

    /*
     * Python : ApplyForceToCenter(uniform(-INITIAL_RANDOM, INITIAL_RANDOM) x2)
     * En Euler : force = masse * accélération, masse ≈ 1 normalisée.
     * On convertit en vitesse initiale : v = F / (masse * FPS)
     * avec masse = LANDER_DENSITY * volume ≈ 5.0 (même ordre que Box2D)
     */
    float mass_approx = 5.0f;
    env->vx = ll_randrange(INITIAL_RANDOM) / (mass_approx * FPS);
    env->vy = ll_randrange(INITIAL_RANDOM) / (mass_approx * FPS);

    env->leg_contact[0] = 0;
    env->leg_contact[1] = 0;
    env->game_over      = 0;
    env->step_count     = 0;
    env->ep_reward      = 0.0f;
    env->ep_len         = 0;

    /* prev_shaping = NaN pour signaler "pas encore de shaping précédent"
     * (Python : self.prev_shaping = None, puis if prev_shaping is not None) */
    env->prev_shaping = NAN;

    /* Indices de vent aléatoires dans [-9999, 9999] */
    if (env->enable_wind) {
        env->wind_idx   = ll_randrange(9999.0f);
        env->torque_idx = ll_randrange(9999.0f);
    }

    ll_fill_obs(env);
}

/* ══════════════════════════════════════════════════════════════════
 * c_step
 *
 * Suit fidèlement la logique du Python :
 *   1. Application du vent (si enable_wind et pas de contact jambe)
 *   2. Calcul des impulsions moteur (tip / side / dispersion)
 *   3. Intégration Euler (substitut à world.Step)
 *   4. Détection contact jambes
 *   5. Calcul récompense (shaping + fuel + terminaison)
 *   6. Remplissage observation
 * ══════════════════════════════════════════════════════════════════ */
void c_step(LunarLander* env) {
    int action = env->actions[0];  /* 0=rien 1=gauche 2=principal 3=droite */

    /* ── 1. Vent ── */
    if (env->enable_wind && !(env->leg_contact[0] || env->leg_contact[1])) {
        /*
         * Python :
         *   wind_mag = tanh(sin(0.02 * wind_idx) + sin(pi * 0.01 * wind_idx)) * wind_power
         *   torque_mag = tanh(sin(0.02 * torque_idx) + sin(pi * 0.01 * torque_idx)) * turbulence_power
         */
        float wind_mag = tanhf(
            sinf(0.02f * env->wind_idx) + sinf((float)M_PI * 0.01f * env->wind_idx)
        ) * env->wind_power;
        env->wind_idx += 1.0f;

        float torque_mag = tanhf(
            sinf(0.02f * env->torque_idx) + sinf((float)M_PI * 0.01f * env->torque_idx)
        ) * env->turbulence_power;
        env->torque_idx += 1.0f;

        /* ApplyForceToCenter → accélération = F/masse * DT */
        env->vx    += (wind_mag / 5.0f) * DT;
        env->omega += (torque_mag / 5.0f) * DT;
    }

    /* ── 2. Impulsions moteurs ── */
    /*
     * Python :
     *   tip  = (sin(angle), cos(angle))   — axe longitudinal du lander
     *   side = (-cos(angle), sin(angle))  — axe latéral
     *   dispersion = uniform(-1/SCALE, +1/SCALE) x2
     *
     * Moteur principal (action == 2) :
     *   ox = tip[0] * (MAIN_ENGINE_Y_LOCATION/SCALE + 2*disp[0]) + side[0]*disp[1]
     *   oy = -tip[1]* (MAIN_ENGINE_Y_LOCATION/SCALE + 2*disp[0]) - side[1]*disp[1]
     *   impulse = (-ox, -oy) * MAIN_ENGINE_POWER
     *
     * Propulseurs latéraux (action 1 = gauche, action 3 = droite) :
     *   direction = action - 2  → -1 pour gauche, +1 pour droite
     *   ox = tip[0]*disp[0] + side[0]*(3*disp[1] + direction*SIDE_ENGINE_AWAY/SCALE)
     *   oy = -tip[1]*disp[0] - side[1]*(3*disp[1] + direction*SIDE_ENGINE_AWAY/SCALE)
     *   impulse = (-ox, -oy) * SIDE_ENGINE_POWER
     */
    float tip_x  =  sinf(env->angle);
    float tip_y  =  cosf(env->angle);
    float side_x = -tip_y;
    float side_y =  tip_x;

    float disp0 = ll_randrange(1.0f / SCALE);
    float disp1 = ll_randrange(1.0f / SCALE);

    float ax = 0.0f, ay = 0.0f, atorque = 0.0f;
    float m_power = 0.0f, s_power = 0.0f;

    if (action == 2) {
        /* Moteur principal */
        m_power = 1.0f;
        float ox = tip_x  * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0f * disp0) + side_x * disp1;
        float oy = -tip_y * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0f * disp0) - side_y * disp1;
        /* ApplyLinearImpulse(-ox, -oy) * MAIN_ENGINE_POWER → Δv = impulse / masse */
        ax += -ox * MAIN_ENGINE_POWER / 5.0f;
        ay += -oy * MAIN_ENGINE_POWER / 5.0f;
    }

    if (action == 1 || action == 3) {
        /* Propulseurs latéraux */
        s_power = 1.0f;
        float direction = (action == 1) ? -1.0f : 1.0f;  /* 1=gauche=-1, 3=droite=+1 */
        float ox = tip_x  * disp0 + side_x * (3.0f * disp1 + direction * SIDE_ENGINE_AWAY / SCALE);
        float oy = -tip_y * disp0 - side_y * (3.0f * disp1 + direction * SIDE_ENGINE_AWAY / SCALE);
        ax += -ox * SIDE_ENGINE_POWER / 5.0f;
        ay += -oy * SIDE_ENGINE_POWER / 5.0f;
        /*
         * Torque induit par les propulseurs latéraux.
         * Dans Box2D le point d'application est décalé, ce qui crée un couple.
         * On l'approxime : τ = F × bras de levier ≈ SIDE_ENGINE_AWAY / SCALE
         */
        atorque += direction * SIDE_ENGINE_POWER * (SIDE_ENGINE_AWAY / SCALE) / 5.0f;
    }

    /* ── 3. Intégration Euler ── */
    env->vx    += (ax)              * DT;
    env->vy    += (ay + GRAVITY)    * DT;
    env->omega += atorque           * DT * 20.0f;

    /* Drag aérien léger (Box2D linear/angular damping) */
    env->vx    *= 0.999f;
    env->vy    *= 0.999f;
    env->omega *= 0.997f;

    env->x     += env->vx    * DT;
    env->y     += env->vy    * DT;
    env->angle += env->omega * DT;

    /* Normalisation angle dans [-π, π] */
    while (env->angle >  (float)M_PI) env->angle -= 2.0f * (float)M_PI;
    while (env->angle < -(float)M_PI) env->angle += 2.0f * (float)M_PI;

    /* ── 4. Contact jambes ── */
    /*
     * Python : Box2D gère les contacts via ContactDetector.
     * On recalcule géométriquement la position de chaque pied.
     * LEG_AWAY et LEG_DOWN sont déjà en mètres (px/SCALE).
     */
    float sa = sinf(env->angle), ca = cosf(env->angle);
    for (int leg = 0; leg < 2; leg++) {
        float sign = (leg == 0) ? -1.0f : 1.0f;   /* 0=gauche, 1=droite */
        float lx = env->x + sign * LEG_AWAY * ca - LEG_DOWN * sa;
        float ly = env->y + sign * LEG_AWAY * sa + LEG_DOWN * ca;
        float gy = ll_ground_at(env, lx);
        env->leg_contact[leg] = (ly <= gy + 0.05f) ? 1 : 0;
    }

    /* Contact corps → game_over (Python : ContactDetector.BeginContact) */
    float gy_body = ll_ground_at(env, env->x);
    if (env->y <= gy_body + 0.05f)
        env->game_over = 1;

    /* ── 5. Récompense ── */
    ll_fill_obs(env);   /* on remplit les obs ici pour utiliser state[] */

    float s0 = env->observations[0];
    float s1 = env->observations[1];
    float s2 = env->observations[2];
    float s3 = env->observations[3];
    float s4 = env->observations[4];
    float s6 = env->observations[6];
    float s7 = env->observations[7];

    float shaping =
        -100.0f * sqrtf(s0*s0 + s1*s1)
        -100.0f * sqrtf(s2*s2 + s3*s3)
        -100.0f * fabsf(s4)
        +  10.0f * s6
        +  10.0f * s7;

    float reward = 0.0f;
    /* Python : if self.prev_shaping is not None: reward = shaping - prev_shaping */
    if (!isnan(env->prev_shaping))
        reward = shaping - env->prev_shaping;
    env->prev_shaping = shaping;

    /* Pénalité carburant */
    reward -= m_power * 0.30f;
    reward -= s_power * 0.03f;

    /* ── Terminaison ── */
    env->terminals[0] = 0;
    env->step_count++;

    /*
     * Python :
     *   if game_over or abs(state[0]) >= 1.0 → terminated, reward = -100
     *   if not lander.awake → terminated, reward = +100
     *
     * Substitut "awake" : le lander est considéré endormi (posé) si :
     *   - les deux jambes sont en contact ET
     *   - vitesses très faibles (vx,vy < 0.1, omega < 0.1)
     */
    int out_x  = (fabsf(s0) >= 1.0f);
    int awake  = !(env->leg_contact[0] && env->leg_contact[1]
                   && fabsf(env->vx) < 0.1f && fabsf(env->vy) < 0.1f
                   && fabsf(env->omega) < 0.1f);

    if (env->game_over || out_x) {
        env->terminals[0] = 1;
        reward = -100.0f;
    } else if (!awake) {
        env->terminals[0] = 1;
        reward = +100.0f;
    }

    env->rewards[0]  = reward;
    env->ep_reward  += reward;
    env->ep_len++;

    /* Log + reset automatique en fin d'épisode */
    if (env->terminals[0]) {
        env->log.score       += env->ep_reward;
        env->log.episode_len += (float)env->ep_len;
        env->log.n           += 1.0f;
        c_reset(env);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * c_render
 *
 * Fidèle au rendu Python (Pygame → Raylib) :
 *   - Fond noir
 *   - Polygones terrain (sky_polys)
 *   - Drapeaux hélipad (deux mâts + triangle jaune)
 *   - Corps du lander (polygone LANDER_POLY orienté)
 *   - Jambes
 *   - Flammes moteurs
 *   - HUD
 * ══════════════════════════════════════════════════════════════════ */
void c_render(LunarLander* env) {
    if (!IsWindowReady()) {
        InitWindow((int)VIEWPORT_W, (int)VIEWPORT_H, "PufferLib — LunarLander v3");
        SetTargetFPS(FPS);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(BLACK);   /* Python : fond noir */

    float cw = W / (CHUNKS - 1);   /* largeur d'un chunk en mètres */

    /* ── Terrain (sky_polys) ── */
    for (int i = 0; i < CHUNKS - 1; i++) {
        float x0 = i       * cw * SCALE;
        float x1 = (i + 1) * cw * SCALE;
        float y0 = VIEWPORT_H - env->terrain_y[i]     * SCALE;
        float y1 = VIEWPORT_H - env->terrain_y[i + 1] * SCALE;

        /* Couleur hélipad (chunks centraux) */
        int mid = CHUNKS / 2;
        Color col = (i >= mid - 2 && i <= mid + 1) ? PUFF_YELLOW : PUFF_GRAY;

        /* Triangle supérieur du polygone du chunk */
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
        /* Bord supérieur bien visible */
        DrawLine((int)x0, (int)y0, (int)x1, (int)y1, PUFF_WHITE);
    }

    /* ── Drapeaux hélipad ── */
    /*
     * Python :
     *   for x in [helipad_x1, helipad_x2]:
     *     flagy1 = helipad_y * SCALE
     *     flagy2 = flagy1 + 50
     *     DrawLine (mât blanc)
     *     DrawPolygon (triangle jaune) : (x, flagy2), (x, flagy2-10), (x+25, flagy2-5)
     * Note : en Pygame Y est vers le bas ; après flip vertical les drapeaux
     * pointent vers le haut. En Raylib on adapte directement.
     */
    float flag_xs[2] = { env->helipad_x1 * SCALE, env->helipad_x2 * SCALE };
    for (int f = 0; f < 2; f++) {
        float fx     = flag_xs[f];
        float flagy1 = VIEWPORT_H - env->helipad_y * SCALE;
        float flagy2 = flagy1 - 50.0f;   /* vers le haut en Raylib */

        /* Mât */
        DrawLine((int)fx, (int)flagy1, (int)fx, (int)flagy2, PUFF_WHITE);
        /* Triangle jaune */
        DrawTriangle(
            (Vector2){fx,        flagy2},
            (Vector2){fx,        flagy2 + 10.0f},
            (Vector2){fx + 25.0f, flagy2 + 5.0f},
            PUFF_YELLOW
        );
    }

    /* ── Jambes ── */
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

    /* ── Corps du lander ── */
    /*
     * Python : polygone LANDER_POLY = [(-14,+17),(-17,0),(-17,-10),(+17,-10),(+17,0),(+14,+17)]
     * en unités lander (pixels avant SCALE). On applique la rotation et la translation.
     */
    float poly_local[6][2] = {
        {-14, +17}, {-17, 0}, {-17, -10}, {+17, -10}, {+17, 0}, {+14, +17}
    };
    Vector2 poly_world[6];
    for (int v = 0; v < 6; v++) {
        float px = poly_local[v][0] / SCALE;
        float py = poly_local[v][1] / SCALE;
        /* Rotation */
        float rx = px * ca - py * sa;
        float ry = px * sa + py * ca;
        /* Translation + flip Y */
        poly_world[v].x = env->x * SCALE + rx * SCALE;
        poly_world[v].y = VIEWPORT_H - (env->y * SCALE + ry * SCALE);
    }
    /* Décomposition en 4 triangles (fan depuis le centre) */
    Vector2 center = {(float)bx, (float)by};
    for (int v = 0; v < 6; v++) {
        DrawTriangle(
            center,
            poly_world[v],
            poly_world[(v + 1) % 6],
            PUFF_PURPLE
        );
    }
    /* Contour */
    for (int v = 0; v < 6; v++) {
        DrawLine(
            (int)poly_world[v].x,             (int)poly_world[v].y,
            (int)poly_world[(v+1)%6].x, (int)poly_world[(v+1)%6].y,
            PUFF_CYAN
        );
    }

    /* tip = (sin(angle), cos(angle)) — axe longitudinal du lander */
    float tip_x =  sa;   /* sa = sinf(env->angle), déjà calculé ci-dessus */
    float tip_y =  ca;   /* ca = cosf(env->angle) */

    /* ── Flamme moteur principal (action == 2) ── */
    if (env->actions[0] == 2) {
        /* Sortie du moteur : en bas du lander selon son axe */
        float fx = (float)bx - tip_x * MAIN_ENGINE_Y_LOCATION;
        float fy = (float)by + tip_y * MAIN_ENGINE_Y_LOCATION;
        DrawCircle((int)fx, (int)fy, 10, PUFF_RED);
        DrawCircle((int)fx, (int)fy,  6, PUFF_YELLOW);
    }

    /* ── Flammes propulseurs latéraux (action 1 = gauche, 3 = droite) ── */
    if (env->actions[0] == 1 || env->actions[0] == 3) {
        float direction = (env->actions[0] == 1) ? -1.0f : 1.0f;
        /* Côté du lander : axe perpendiculaire */
        float fx = (float)bx + direction * (-ca) * SIDE_ENGINE_AWAY * 0.5f;
        float fy = (float)by - direction * ( sa) * SIDE_ENGINE_AWAY * 0.5f;
        DrawCircle((int)fx, (int)fy, 6, PUFF_RED);
        DrawCircle((int)fx, (int)fy, 4, PUFF_YELLOW);
    }

    /* ── HUD ── */
    int hud_y = 8;
    DrawText(TextFormat("Reward ep : %.1f",    env->ep_reward),          8, hud_y,      16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Steps     : %d",      env->step_count),         8, hud_y,      16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Action    : %d",      env->actions[0]),         8, hud_y,      16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Vel X/Y   : %.2f / %.2f", env->vx, env->vy),   8, hud_y,      16, PUFF_WHITE); hud_y += 20;
    DrawText(TextFormat("Angle     : %.2f rad", env->angle),             8, hud_y,      16, PUFF_WHITE); hud_y += 20;
    if (env->enable_wind)
        DrawText(TextFormat("Wind idx  : %.0f", env->wind_idx),          8, hud_y,      16, PUFF_CYAN);
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