#include "lunar_lander.h"

int main(void) {
    LunarLander env = {0};

    env.observations = (float*)calloc(OBS_DIM, sizeof(float));
    env.actions      = (int*)calloc(1, sizeof(int));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));
#include "lunar_lander.h"

/*
 * Lunar Lander — PufferLib format, fidèle à Gymnasium LunarLander-v3
 * ====================================================================
 * Compilation (avec Raylib) :
 *   gcc -O2 -o lunar_lander lunar_lander.c \
 *       -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
 *
 * Contrôle humain (maintenir Shift) :
 *   ↑        → moteur principal   (action 2)
 *   ← ou A   → propulseur gauche  (action 1)
 *   → ou D   → propulseur droit   (action 3)
 *   (rien)   → rien               (action 0)
 *
 * Sans Shift : politique aléatoire.
 *
 * Variables d'environnement :
 *   WIND=1          active le vent
 *   WIND_POWER=15   puissance du vent (0..20, défaut 15)
 *   TURB=1.5        turbulence angulaire (0..2, défaut 1.5)
 */

int main(void) {
    LunarLander env = {0};

    /* ── Allocation des buffers PufferLib ── */
    env.observations = (float*)calloc(OBS_DIM, sizeof(float));
    env.actions      = (int*)calloc(1, sizeof(int));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));

    /* ── Configuration vent ── */
    env.enable_wind       = 0;     /* désactivé par défaut */
    env.wind_power        = 15.0f;
    env.turbulence_power  = 1.5f;

    c_reset(&env);
    c_render(&env);   /* ouvre la fenêtre Raylib */

    while (!WindowShouldClose()) {

        /* ── Entrées clavier ── */
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            /* Contrôle humain :
             *   action 0 = rien
             *   action 1 = propulseur gauche  (←/A)
             *   action 2 = moteur principal   (↑)
             *   action 3 = propulseur droit   (→/D)
             */
            if (IsKeyDown(KEY_UP)) {
                env.actions[0] = 2;
            } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.actions[0] = 1;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.actions[0] = 3;
            } else {
                env.actions[0] = 0;
            }
        } else {
            /* Politique aléatoire */
            env.actions[0] = rand() % ACT_DIM;
        }

        /* Touche W : toggle vent en cours de simulation */
        if (IsKeyPressed(KEY_W)) {
            env.enable_wind = !env.enable_wind;
        }

        c_step(&env);
        c_render(&env);
    }

    /* ── Libération mémoire ── */
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
    c_reset(&env);
    c_render(&env);   

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            /* Contrôle humain (Shift maintenu) */
            if (IsKeyDown(KEY_UP)) {
                env.actions[0] = 1;   /* moteur principal */
            } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.actions[0] = 2;   /* propulseur gauche */
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.actions[0] = 3;   /* propulseur droit  */
            } else {
                env.actions[0] = 0;   /* rien              */
            }
        } else {
            env.actions[0] = rand() % ACT_DIM;
        }

        c_step(&env);
        c_render(&env);
    }

    /* Libération mémoire */
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
