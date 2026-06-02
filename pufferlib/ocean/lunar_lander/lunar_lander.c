#include "lunar_lander.h"

int main(void) {
    LunarLander env = {0};
    env.observations = (float*)calloc(OBS_DIM, sizeof(float));
    env.actions      = (int*)calloc(1, sizeof(int));
    env.rewards      = (float*)calloc(REWARD_DIM, sizeof(float));
    env.weights      = (float*)calloc(REWARD_DIM, sizeof(float));
    env.terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));

    /* Default scalarisation weights: equal preference */
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
            /* Human control (Shift held) */
            if (IsKeyDown(KEY_UP)) {
                env.actions[0] = 2;   /* main engine      */
            } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.actions[0] = 1;   /* left thruster    */
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.actions[0] = 3;   /* right thruster   */
            } else {
                env.actions[0] = 0;   /* nothing          */
            }
        } else {
            env.actions[0] = rand() % ACT_DIM;
        }

        /* Toggle wind with W */
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
    c_close(&env);
    return 0;
}