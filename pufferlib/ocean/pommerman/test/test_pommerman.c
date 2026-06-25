/* raylib.h stub is in this test/ directory */
#include "../pommerman.h"

/* ── Test helpers ─────────────────────────────────────────────────────────── */
#include <stdio.h>
static int tests_run = 0;
static int tests_ok  = 0;

#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_ok++; printf("  [OK]  %s\n", msg); } \
    else        { printf("  [FAIL] %s\n", msg); } \
} while(0)

/* ── Tests ────────────────────────────────────────────────────────────────── */

static void test_reset(Pommerman* env) {
    printf("\n=== test_reset ===\n");
    c_reset(env);
    CHECK(env->step_count == 0,          "step_count == 0 after reset");
    CHECK(env->num_bombs  == 0,          "no bombs after reset");
    CHECK(env->agent_alive[0] == 1,      "agent 0 alive");
    CHECK(env->agent_alive[1] == 1,      "agent 1 alive");
    CHECK(env->agent_alive[2] == 1,      "agent 2 alive");
    CHECK(env->agent_alive[3] == 1,      "agent 3 alive");
    CHECK(env->agent_ammo[0]  == DEFAULT_AMMO,  "agent 0 ammo correct");
    CHECK(env->agent_blast[0] == DEFAULT_BLAST_STRENGTH, "agent 0 blast correct");
    CHECK(env->agent_can_kick[0] == 0,   "agent 0 cannot kick");
    // Agents at correct corners
    CHECK(env->agent_row[0] == 0  && env->agent_col[0] == 0,   "agent 0 at (0,0)");
    CHECK(env->agent_row[1] == 0  && env->agent_col[1] == 10,  "agent 1 at (0,10)");
    CHECK(env->agent_row[2] == 10 && env->agent_col[2] == 0,   "agent 2 at (10,0)");
    CHECK(env->agent_row[3] == 10 && env->agent_col[3] == 10,  "agent 3 at (10,10)");
    // Board cell under each agent
    CHECK(env->board[pmm_idx(0,0)]   == CELL_AGENT_0, "cell (0,0)  = AGENT_0");
    CHECK(env->board[pmm_idx(0,10)]  == CELL_AGENT_1, "cell (0,10) = AGENT_1");
    CHECK(env->board[pmm_idx(10,0)]  == CELL_AGENT_2, "cell (10,0) = AGENT_2");
    CHECK(env->board[pmm_idx(10,10)] == CELL_AGENT_3, "cell (10,10)= AGENT_3");
    // Rigid walls at interior even-even positions
    CHECK(env->board[pmm_idx(2,2)] == CELL_RIGID, "rigid at (2,2)");
    CHECK(env->board[pmm_idx(4,4)] == CELL_RIGID, "rigid at (4,4)");
    CHECK(env->board[pmm_idx(8,8)] == CELL_RIGID, "rigid at (8,8)");
    // Corner clear zones are not rigid
    CHECK(env->board[pmm_idx(0,1)] != CELL_RIGID, "no rigid at (0,1) clear zone");
    CHECK(env->board[pmm_idx(1,0)] != CELL_RIGID, "no rigid at (1,0) clear zone");
}

static void test_movement(Pommerman* env) {
    printf("\n=== test_movement ===\n");
    c_reset(env);
    // Agent 0 starts at (0,0). Move right (ACTION_RIGHT = 4)
    // (0,1) should be PASSAGE in the clear zone
    env->board[pmm_idx(0,1)] = CELL_PASSAGE; // ensure it's clear
    env->actions[0] = ACTION_RIGHT;
    for (int a = 1; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    c_step(env);
    CHECK(env->agent_col[0] == 1, "agent 0 moved right to col 1");
    CHECK(env->agent_row[0] == 0, "agent 0 still at row 0");

    // Move down
    env->board[pmm_idx(1,1)] = CELL_PASSAGE;
    env->actions[0] = ACTION_DOWN;
    c_step(env);
    CHECK(env->agent_row[0] == 1, "agent 0 moved down to row 1");

    // Try to move into a rigid wall (2,2) — should be blocked
    // Agent is at (1,1). Move right to (1,2), then down to (2,2) which is rigid
    env->board[pmm_idx(1,2)] = CELL_PASSAGE;
    env->actions[0] = ACTION_RIGHT;
    c_step(env);
    CHECK(env->agent_col[0] == 2, "agent 0 moved to col 2");
    env->actions[0] = ACTION_DOWN; // try to go to (2,2) rigid
    c_step(env);
    CHECK(env->agent_row[0] == 1, "agent 0 blocked by rigid wall at (2,2)");
}

static void test_bomb_placement(Pommerman* env) {
    printf("\n=== test_bomb_placement ===\n");
    c_reset(env);
    // Agent 0 at (0,0) places a bomb
    env->actions[0] = ACTION_BOMB;
    for (int a = 1; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    c_step(env);
    CHECK(env->num_bombs == 1,                        "one bomb on board");
    CHECK(env->bombs[0].row == 0,                     "bomb at row 0");
    CHECK(env->bombs[0].col == 0,                     "bomb at col 0");
    CHECK(env->bombs[0].life == BOMB_LIFE - 1,        "bomb life ticked down");
    CHECK(env->bombs[0].blast_strength == DEFAULT_BLAST_STRENGTH, "bomb blast = default");
    CHECK(env->bombs[0].bomber_id == 0,               "bomb owner = agent 0");
    CHECK(env->agent_ammo[0] == DEFAULT_AMMO - 1,     "agent 0 ammo reduced");
    CHECK(env->board[pmm_idx(0,0)] == CELL_BOMB,      "cell (0,0) = BOMB");

    // Cannot place another bomb (no ammo)
    c_step(env);
    CHECK(env->num_bombs == 1, "still only one bomb (no ammo)");
}

static void test_bomb_explosion(Pommerman* env) {
    printf("\n=== test_bomb_explosion ===\n");
    c_reset(env);
    // Manually place a bomb at (5,5) with life=1 (will expire next step)
    env->bombs[env->num_bombs].row           = 5;
    env->bombs[env->num_bombs].col           = 5;
    env->bombs[env->num_bombs].life          = 1;
    env->bombs[env->num_bombs].blast_strength= 2;
    env->bombs[env->num_bombs].bomber_id     = 0;
    env->bombs[env->num_bombs].moving        = 0;
    env->board[pmm_idx(5,5)]                 = CELL_BOMB;
    env->bomb_blast_map[pmm_idx(5,5)]        = 2;
    env->bomb_life_map[pmm_idx(5,5)]         = 1;
    env->num_bombs++;
    env->agent_ammo[0]--;

    for (int a = 0; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    c_step(env);

    // Bomb should be gone
    CHECK(env->num_bombs == 0,                  "bomb removed after explosion");
    CHECK(env->agent_ammo[0] == DEFAULT_AMMO,   "ammo returned after explosion");
    // Flames should be at (5,5) and adjacent cells (up to blast=2)
    // (5,5) center
    CHECK(env->flame_map[pmm_idx(5,5)] > 0,     "flame at (5,5) center");
    // (5,3),(5,4),(5,6),(5,7) horizontal
    // (3,5),(4,5),(6,5),(7,5) vertical
    // Note: some may be blocked by wood walls — just check center
    CHECK(env->board[pmm_idx(5,5)] == CELL_FLAMES || env->flame_map[pmm_idx(5,5)] > 0,
          "center cell is flamed");
}

static void test_agent_death(Pommerman* env) {
    printf("\n=== test_agent_death ===\n");
    c_reset(env);
    // Place a bomb at (0,1) with life=1, blast=5 — should reach agent 0 at (0,0)
    // First make sure (0,1) is clear
    env->board[pmm_idx(0,1)] = CELL_PASSAGE;
    env->bombs[env->num_bombs].row           = 0;
    env->bombs[env->num_bombs].col           = 1;
    env->bombs[env->num_bombs].life          = 1;
    env->bombs[env->num_bombs].blast_strength= 5;
    env->bombs[env->num_bombs].bomber_id     = 1;
    env->bombs[env->num_bombs].moving        = 0;
    env->board[pmm_idx(0,1)]                 = CELL_BOMB;
    env->num_bombs++;

    for (int a = 0; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    c_step(env);

    CHECK(env->agent_alive[0] == 0,          "agent 0 died from bomb blast");
    CHECK(env->rewards[0]     == REWARD_DEATH, "agent 0 got death reward");
}

static void test_powerup_collection(Pommerman* env) {
    printf("\n=== test_powerup_collection ===\n");
    c_reset(env);
    // Place EXTRA_BOMB power-up at (0,1) (clear zone, passable)
    env->board[pmm_idx(0,1)]     = CELL_EXTRA_BOMB;
    env->power_up_map[pmm_idx(0,1)] = PU_NONE; // already revealed

    // Agent 0 moves right to collect it
    env->actions[0] = ACTION_RIGHT;
    for (int a = 1; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    c_step(env);

    CHECK(env->agent_col[0] == 1,               "agent 0 moved to col 1");
    CHECK(env->agent_ammo[0] == DEFAULT_AMMO + 1, "agent 0 collected extra bomb");

    // Place CAN_KICK power-up at (0,2)
    env->board[pmm_idx(0,2)]     = CELL_CAN_KICK;
    env->actions[0] = ACTION_RIGHT;
    c_step(env);
    CHECK(env->agent_can_kick[0] == 1,          "agent 0 collected can_kick");

    // Place INCR_RANGE at (0,3)
    env->board[pmm_idx(0,3)]     = CELL_INCR_RANGE;
    env->actions[0] = ACTION_RIGHT;
    c_step(env);
    CHECK(env->agent_blast[0] == DEFAULT_BLAST_STRENGTH + 1, "agent 0 collected range");
}

static void test_termination_ffa(Pommerman* env) {
    printf("\n=== test_termination_ffa ===\n");
    c_reset(env);
    // Kill agents 1, 2, 3 manually and step — agent 0 should win
    env->agent_alive[1] = 0;
    env->agent_alive[2] = 0;
    env->agent_alive[3] = 0;
    for (int a = 0; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    c_step(env);
    CHECK(env->terminals[0] == 1,            "episode terminates");
    CHECK(env->rewards[0]   == REWARD_WIN,   "surviving agent gets win reward");
}

static void test_max_steps(Pommerman* env) {
    printf("\n=== test_max_steps ===\n");
    c_reset(env);
    env->max_steps = 5;
    for (int a = 0; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    for (int s = 0; s < 4; s++) {
        c_step(env);
        CHECK(env->terminals[0] == 0, "episode not done before max_steps");
    }
    c_step(env);
    CHECK(env->terminals[0] == 1,   "episode done at max_steps");
    env->max_steps = DEFAULT_MAX_STEPS;
}

static void test_observations(Pommerman* env) {
    printf("\n=== test_observations ===\n");
    c_reset(env);
    // Observations array must be non-null and have plausible values
    float* obs0 = &env->observations[0 * OBS_PER_AGENT];
    CHECK(obs0 != NULL, "observations pointer valid");
    // Board channel: all values in [0,1]
    int all_in_range = 1;
    for (int i = 0; i < NUM_CELLS; i++) {
        if (obs0[i] < 0.0f || obs0[i] > 1.0f) { all_in_range = 0; break; }
    }
    CHECK(all_in_range, "board channel values in [0,1]");
    // Agent 0 position scalars (idx = 3*NUM_CELLS + 0,1)
    float row_obs = obs0[NUM_CELLS * 3 + 0];
    float col_obs = obs0[NUM_CELLS * 3 + 1];
    CHECK(row_obs == 0.0f / 10.0f, "agent 0 row obs = 0.0");
    CHECK(col_obs == 0.0f / 10.0f, "agent 0 col obs = 0.0");
}

static void test_chain_explosion(Pommerman* env) {
    printf("\n=== test_chain_explosion ===\n");
    c_reset(env);
    // Two bombs: A at (5,5) life=1, B at (5,7) life=10
    // A's blast (strength=3) should reach B and chain-explode it
    // Make row 5 clear
    for (int c = 4; c <= 8; c++) {
        if (env->board[pmm_idx(5,c)] == CELL_WOOD || env->board[pmm_idx(5,c)] == CELL_RIGID)
            env->board[pmm_idx(5,c)] = CELL_PASSAGE;
    }

    Bomb* bA = &env->bombs[0];
    bA->row = 5; bA->col = 5; bA->life = 1; bA->blast_strength = 3;
    bA->bomber_id = 0; bA->moving = 0;
    env->board[pmm_idx(5,5)] = CELL_BOMB;
    env->bomb_blast_map[pmm_idx(5,5)] = 3;
    env->bomb_life_map[pmm_idx(5,5)]  = 1;

    Bomb* bB = &env->bombs[1];
    bB->row = 5; bB->col = 7; bB->life = 10; bB->blast_strength = 2;
    bB->bomber_id = 1; bB->moving = 0;
    env->board[pmm_idx(5,7)] = CELL_BOMB;
    env->bomb_blast_map[pmm_idx(5,7)] = 2;
    env->bomb_life_map[pmm_idx(5,7)]  = 10;
    env->num_bombs = 2;

    for (int a = 0; a < NUM_AGENTS; a++) env->actions[a] = ACTION_STOP;
    c_step(env);

    CHECK(env->num_bombs == 0,            "both bombs exploded (chain reaction)");
    CHECK(env->flame_map[pmm_idx(5,5)] > 0, "flames at bomb A center");
    CHECK(env->flame_map[pmm_idx(5,7)] > 0, "flames at bomb B center (chained)");
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
int main(void) {
    srand(42);
    Pommerman env = {
        .game_mode = MODE_FFA,
        .max_steps = DEFAULT_MAX_STEPS,
    };
    allocate(&env);

    test_reset(&env);
    test_movement(&env);
    test_bomb_placement(&env);
    test_bomb_explosion(&env);
    test_agent_death(&env);
    test_powerup_collection(&env);
    test_termination_ffa(&env);
    test_max_steps(&env);
    test_observations(&env);
    test_chain_explosion(&env);

    printf("\n==============================\n");
    printf("Results: %d/%d tests passed\n", tests_ok, tests_run);
    printf("==============================\n");

    free_allocated(&env);
    return (tests_ok == tests_run) ? 0 : 1;
}
