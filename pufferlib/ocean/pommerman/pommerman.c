#include <time.h>
#include "pommerman.h"

void demo(void) {
    srand((unsigned)time(NULL));
    Pommerman env = {
        .game_mode = MODE_FFA,
        .max_steps = DEFAULT_MAX_STEPS,
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        // Step every 10 frames so the game is watchable at 10 FPS
        // Random actions for all agents except agent 0 (keyboard-controlled)
        for (int a = 1; a < NUM_AGENTS; a++) {
            env.actions[a] = rand() % NUM_ACTIONS;
        }

        // Agent 0: keyboard control
        env.actions[0] = ACTION_STOP;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = ACTION_UP;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = ACTION_LEFT;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = ACTION_DOWN;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = ACTION_RIGHT;
        if (IsKeyDown(KEY_SPACE))                     env.actions[0] = ACTION_BOMB;

        c_step(&env);

        // Auto-reset when the episode ends
        if (env.terminals[0]) {
            c_reset(&env);
        }

        c_render(&env);
        tick++;
    }

    free_allocated(&env);
}

void performance_test(void) {
    srand(42);
    Pommerman env = {
        .game_mode = MODE_FFA,
        .max_steps = DEFAULT_MAX_STEPS,
    };
    allocate(&env);
    c_reset(&env);

    long start = time(NULL);
    long steps  = 0;
    while (time(NULL) - start < 10) {
        for (int a = 0; a < NUM_AGENTS; a++) {
            env.actions[a] = rand() % NUM_ACTIONS;
        }
        c_step(&env);
        if (env.terminals[0]) c_reset(&env);
        steps++;
    }
    printf("SPS (4 agents): %ld\n", steps * NUM_AGENTS / 10);
    free_allocated(&env);
}

int main(int argc, char** argv) {
    if (argc > 1 && argv[1][0] == 'p') {
        performance_test();
    } else {
        demo();
    }
    return 0;
}
