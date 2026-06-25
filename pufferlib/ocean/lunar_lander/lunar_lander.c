#include "lunar_lander.h"
#include <gsl/gsl_rng.h>

int main(void) {
    LunarLander env = {0};
    env.observations = (float*)calloc(OBS_DIM, sizeof(float));
    env.actions      = (int*)calloc(1, sizeof(int));
    env.rewards      = (float*)calloc(REWARD_DIM, sizeof(float));
    env.weights      = (float*)calloc(REWARD_DIM, sizeof(float));
    env.terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));

    env.gsl_rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set((gsl_rng*)env.gsl_rng, 0);   /* seed = 0, modifiable */

    env.weights[REW_LANDING] = 0.5f;
    env.weights[REW_FUEL]    = 0.5f;
    env.manual_weights       = 1;

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
    free(env.weights);
    free(env.terminals);
    gsl_rng_free((gsl_rng*)env.gsl_rng);   
    c_close(&env);
    return 0;
}