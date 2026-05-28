#include "lunar_lander.h"

int main(void) {
    LunarLander env = {0};
    env.observations = (float*)calloc(OBS_DIM, sizeof(float));
    env.actions      = (int*)calloc(1, sizeof(int));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));

    env.enable_wind       = 0;     
    env.wind_power        = 15.0f;
    env.turbulence_power  = 1.5f;

    c_reset(&env);
    c_render(&env);   

    while (!WindowShouldClose()) {

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
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
            env.actions[0] = rand() % ACT_DIM;
        }

        if (IsKeyPressed(KEY_W)) {
            env.enable_wind = !env.enable_wind;
        }

        c_step(&env);
        c_render(&env);
    }

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
