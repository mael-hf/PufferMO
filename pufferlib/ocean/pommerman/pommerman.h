#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "raylib.h"

// Board
#define BOARD_SIZE 11
#define NUM_CELLS  (BOARD_SIZE * BOARD_SIZE)
#define NUM_AGENTS 4
#define MAX_BOMBS  64

// Game parameters
#define BOMB_LIFE              10
#define FLAME_LIFE             2
#define DEFAULT_AMMO           1
#define DEFAULT_BLAST_STRENGTH 2
#define DEFAULT_MAX_STEPS      800

// Actions
#define ACTION_STOP  0
#define ACTION_UP    1
#define ACTION_LEFT  2
#define ACTION_DOWN  3
#define ACTION_RIGHT 4
#define ACTION_BOMB  5
#define NUM_ACTIONS  6

// Cell types (match original Pommerman encoding)
#define CELL_PASSAGE    0
#define CELL_RIGID      1
#define CELL_WOOD       2
#define CELL_BOMB       3
#define CELL_FLAMES     4
#define CELL_FOG        5
#define CELL_EXTRA_BOMB 6
#define CELL_INCR_RANGE 7
#define CELL_CAN_KICK   8
#define CELL_AGENT_0    10
#define CELL_AGENT_1    11
#define CELL_AGENT_2    12
#define CELL_AGENT_3    13

// Hidden power-up types stored separately from board
#define PU_NONE       0
#define PU_EXTRA_BOMB 1
#define PU_INCR_RANGE 2
#define PU_CAN_KICK   3

// Game modes
#define MODE_FFA  0
#define MODE_TEAM 1

// Observation per agent:
//   3 channels of NUM_CELLS (board, bomb_blast_strength, bomb_life)
//   + 8 scalar features (row, col, ammo, blast, can_kick, alive, mode, progress)
#define OBS_PER_AGENT (NUM_CELLS * 3 + 8)  // 371

// Rendering
#define GRID_SQ      56
#define RENDER_W     (BOARD_SIZE * GRID_SQ)
#define RENDER_H     (BOARD_SIZE * GRID_SQ + 48)

// Rewards
#define REWARD_WIN   1.0f
#define REWARD_KILL  1.0f
#define REWARD_DEATH (-1.0f)

// Direction deltas (row, col) for UP=0, LEFT=1, DOWN=2, RIGHT=3
static const int DIR_DR[4] = {-1,  0,  1, 0};
static const int DIR_DC[4] = { 0, -1,  0, 1};

// Agent start corners
static const int AGENT_START_R[4] = { 0,  0, 10, 10};
static const int AGENT_START_C[4] = { 0, 10,  0, 10};

// Team assignments for MODE_TEAM:
//   team 0 = agents {0,3} (opposite corners), team 1 = agents {1,2}
static const int TEAM_OF[4]       = {0, 1, 1, 0};
static const int TEAM_AGENTS[2][2] = {{0, 3}, {1, 2}};

// ── Structs ───────────────────────────────────────────────────────────────────

typedef struct {
    float kills;
    float deaths;
    float wins;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int row, col;
    int life;
    int blast_strength;
    int bomber_id;
    int moving;
    int move_dir;  // 0=UP 1=LEFT 2=DOWN 3=RIGHT
} Bomb;

typedef struct {
    int initialized;
} Client;

typedef struct {
    Client* client;

    // RL interface (required fields)
    float*         observations;  // [NUM_AGENTS * OBS_PER_AGENT]
    int*           actions;       // [NUM_AGENTS]
    float*         rewards;       // [NUM_AGENTS]
    unsigned char* terminals;     // [NUM_AGENTS]

    Log log;
    Log agent_logs[NUM_AGENTS];

    // Board state
    int   board[NUM_CELLS];
    int   power_up_map[NUM_CELLS];    // hidden PU under each wood wall
    float bomb_blast_map[NUM_CELLS];  // blast strength of bomb at cell (0 if none)
    float bomb_life_map[NUM_CELLS];   // remaining life of bomb at cell (0 if none)
    int   flame_map[NUM_CELLS];       // remaining flame ticks at cell (0 if none)

    // Agent state
    int agent_row[NUM_AGENTS];
    int agent_col[NUM_AGENTS];
    int agent_ammo[NUM_AGENTS];
    int agent_blast[NUM_AGENTS];
    int agent_can_kick[NUM_AGENTS];
    int agent_alive[NUM_AGENTS];

    // Bombs currently on the board
    Bomb bombs[MAX_BOMBS];
    int  num_bombs;

    // Explosion queue (per-env, avoids global state for vectorised use)
    int  exp_queue[MAX_BOMBS];
    bool exp_visited[MAX_BOMBS];
    int  exp_head;
    int  exp_tail;

    // Config
    int game_mode;
    int max_steps;
    int step_count;
    int width;
    int height;
} Pommerman;

// ── Small helpers ─────────────────────────────────────────────────────────────

static inline bool pmm_in_bounds(int r, int c) {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
}

static inline int pmm_idx(int r, int c) {
    return r * BOARD_SIZE + c;
}

// Returns true if (r,c) is inside the 2x2 clear zone at any agent spawn corner
static inline bool pmm_is_clear_zone(int r, int c) {
    return (r <= 1 && c <= 1)
        || (r <= 1 && c >= BOARD_SIZE - 2)
        || (r >= BOARD_SIZE - 2 && c <= 1)
        || (r >= BOARD_SIZE - 2 && c >= BOARD_SIZE - 2);
}

// Find the index of the bomb at (r,c), returns -1 if none
static int pmm_find_bomb(Pommerman* env, int r, int c) {
    for (int i = 0; i < env->num_bombs; i++) {
        if (env->bombs[i].row == r && env->bombs[i].col == c) return i;
    }
    return -1;
}

// Remove bomb at index idx (swap with last)
static void pmm_remove_bomb(Pommerman* env, int idx) {
    int ci = pmm_idx(env->bombs[idx].row, env->bombs[idx].col);
    if (env->board[ci] == CELL_BOMB) env->board[ci] = CELL_PASSAGE;
    env->bomb_blast_map[ci] = 0.0f;
    env->bomb_life_map[ci]  = 0.0f;
    env->bombs[idx] = env->bombs[--env->num_bombs];
}

static void pmm_add_log(Pommerman* env, Log* al) {
    env->log.kills          += al->kills;
    env->log.deaths         += al->deaths;
    env->log.wins           += al->wins;
    env->log.score          += al->score;
    env->log.episode_return += al->episode_return;
    env->log.episode_length += al->episode_length;
    env->log.n              += 1.0f;
}

// ── Board generation ─────────────────────────────────────────────────────────

static void pmm_generate_board(Pommerman* env) {
    memset(env->board,          0, NUM_CELLS * sizeof(int));
    memset(env->power_up_map,   0, NUM_CELLS * sizeof(int));
    memset(env->bomb_blast_map, 0, NUM_CELLS * sizeof(float));
    memset(env->bomb_life_map,  0, NUM_CELLS * sizeof(float));
    memset(env->flame_map,      0, NUM_CELLS * sizeof(int));
    env->num_bombs = 0;

    // Interior rigid-wall pillars at every even-(row,col) cross point (not borders)
    for (int r = 2; r < BOARD_SIZE - 1; r += 2) {
        for (int c = 2; c < BOARD_SIZE - 1; c += 2) {
            env->board[pmm_idx(r, c)] = CELL_RIGID;
        }
    }

    // Random wood walls on remaining passable cells outside the corner clearings
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (pmm_is_clear_zone(r, c)) continue;
            if (env->board[pmm_idx(r, c)] != CELL_PASSAGE) continue;
            if (rand() % 2 == 0) {
                env->board[pmm_idx(r, c)] = CELL_WOOD;
                // ~50% of wood walls hide a power-up
                if (rand() % 2 == 0) {
                    env->power_up_map[pmm_idx(r, c)] = 1 + rand() % 3;
                }
            }
        }
    }

    // Place agents in their spawn corners
    for (int a = 0; a < NUM_AGENTS; a++) {
        env->agent_row[a]     = AGENT_START_R[a];
        env->agent_col[a]     = AGENT_START_C[a];
        env->agent_ammo[a]    = DEFAULT_AMMO;
        env->agent_blast[a]   = DEFAULT_BLAST_STRENGTH;
        env->agent_can_kick[a]= 0;
        env->agent_alive[a]   = 1;
        env->board[pmm_idx(AGENT_START_R[a], AGENT_START_C[a])] = CELL_AGENT_0 + a;
    }
}

// ── Observations ─────────────────────────────────────────────────────────────

static void pmm_compute_obs(Pommerman* env) {
    float (*obs)[OBS_PER_AGENT] = (float(*)[OBS_PER_AGENT])env->observations;

    for (int a = 0; a < NUM_AGENTS; a++) {
        float* o = obs[a];
        int idx = 0;

        // Channel 0: board cell type, normalised to [0,1]
        for (int i = 0; i < NUM_CELLS; i++) o[idx++] = env->board[i] / 13.0f;

        // Channel 1: bomb blast-strength map, normalised
        for (int i = 0; i < NUM_CELLS; i++) o[idx++] = env->bomb_blast_map[i] / 10.0f;

        // Channel 2: bomb remaining-life map, normalised
        for (int i = 0; i < NUM_CELLS; i++) o[idx++] = env->bomb_life_map[i] / (float)BOMB_LIFE;

        // Scalars (8)
        o[idx++] = env->agent_row[a]      / (float)(BOARD_SIZE - 1);
        o[idx++] = env->agent_col[a]      / (float)(BOARD_SIZE - 1);
        o[idx++] = env->agent_ammo[a]     / 10.0f;
        o[idx++] = env->agent_blast[a]    / 10.0f;
        o[idx++] = (float)env->agent_can_kick[a];
        o[idx++] = (float)env->agent_alive[a];
        o[idx++] = (env->game_mode == MODE_TEAM) ? 1.0f : 0.0f;
        o[idx++] = env->step_count / (float)env->max_steps;
    }
}

// ── Explosion mechanics ───────────────────────────────────────────────────────

static void pmm_queue_bomb(Pommerman* env, int bi) {
    if (!env->exp_visited[bi]) {
        env->exp_visited[bi] = true;
        env->exp_queue[env->exp_tail++ % MAX_BOMBS] = bi;
    }
}

// Spread flames from one bomb; chains into other bombs hit by the blast
static void pmm_explode_one(Pommerman* env, int bi) {
    Bomb* b = &env->bombs[bi];
    int br = b->row, bc = b->col;

    // Flame at bomb's own cell
    env->flame_map[pmm_idx(br, bc)] = FLAME_LIFE;

    for (int d = 0; d < 4; d++) {
        for (int s = 1; s <= b->blast_strength; s++) {
            int nr = br + DIR_DR[d] * s;
            int nc = bc + DIR_DC[d] * s;
            if (!pmm_in_bounds(nr, nc)) break;

            int ci   = pmm_idx(nr, nc);
            int cell = env->board[ci];

            if (cell == CELL_RIGID) break;  // blocked; no flame on rigid

            env->flame_map[ci] = FLAME_LIFE;

            if (cell == CELL_WOOD) {
                // Destroy wood; reveal power-up if present, otherwise passage
                int pu = env->power_up_map[ci];
                if (pu == PU_EXTRA_BOMB) {
                    env->board[ci] = CELL_EXTRA_BOMB;
                } else if (pu == PU_INCR_RANGE) {
                    env->board[ci] = CELL_INCR_RANGE;
                } else if (pu == PU_CAN_KICK) {
                    env->board[ci] = CELL_CAN_KICK;
                } else {
                    env->board[ci] = CELL_FLAMES;
                }
                env->power_up_map[ci] = PU_NONE;
                break;  // flame does not pass through wood
            }

            if (cell == CELL_BOMB) {
                int chain = pmm_find_bomb(env, nr, nc);
                if (chain >= 0) {
                    env->bombs[chain].life = 0;
                    pmm_queue_bomb(env, chain);
                }
                break;  // flame stops at the chained bomb cell
            }
            // Passage, flames, power-ups, agents: flame continues
        }
    }
}

// Process all queued explosions (handles chains)
static void pmm_process_explosions(Pommerman* env) {
    while (env->exp_head < env->exp_tail) {
        int bi = env->exp_queue[env->exp_head++ % MAX_BOMBS];
        pmm_explode_one(env, bi);
    }
}

// Apply flame damage to agents and update board cells
static void pmm_apply_flame_damage(Pommerman* env) {
    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!env->agent_alive[a]) continue;
        if (env->flame_map[pmm_idx(env->agent_row[a], env->agent_col[a])] > 0) {
            env->agent_alive[a] = 0;
            env->agent_logs[a].deaths += 1.0f;
        }
    }

    // Mark passable/bomb cells that are on fire as CELL_FLAMES
    for (int i = 0; i < NUM_CELLS; i++) {
        if (env->flame_map[i] <= 0) continue;
        int cell = env->board[i];
        if (cell == CELL_PASSAGE || cell == CELL_BOMB) {
            env->board[i] = CELL_FLAMES;
        }
        // Power-up cells keep their type (collectible after flame dies)
        // Agent cells handled below when we refresh agent positions
    }
}

// ── Power-up collection ───────────────────────────────────────────────────────

static void pmm_collect_powerups(Pommerman* env) {
    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!env->agent_alive[a]) continue;
        int ci = pmm_idx(env->agent_row[a], env->agent_col[a]);
        switch (env->board[ci]) {
            case CELL_EXTRA_BOMB:
                env->agent_ammo[a]++;
                env->board[ci] = CELL_AGENT_0 + a;
                break;
            case CELL_INCR_RANGE:
                env->agent_blast[a]++;
                env->board[ci] = CELL_AGENT_0 + a;
                break;
            case CELL_CAN_KICK:
                env->agent_can_kick[a] = 1;
                env->board[ci] = CELL_AGENT_0 + a;
                break;
            default: break;
        }
    }
}

// ── Termination helpers ───────────────────────────────────────────────────────

static int pmm_alive_count(Pommerman* env) {
    int n = 0;
    for (int a = 0; a < NUM_AGENTS; a++) n += env->agent_alive[a];
    return n;
}

// For MODE_TEAM: returns winning team (0 or 1), -1 if game continues, -2 if draw
static int pmm_team_result(Pommerman* env) {
    bool alive0 = env->agent_alive[TEAM_AGENTS[0][0]] || env->agent_alive[TEAM_AGENTS[0][1]];
    bool alive1 = env->agent_alive[TEAM_AGENTS[1][0]] || env->agent_alive[TEAM_AGENTS[1][1]];
    if ( alive0 && !alive1) return 0;
    if (!alive0 &&  alive1) return 1;
    if (!alive0 && !alive1) return -2;
    return -1;
}

// ── Reset ─────────────────────────────────────────────────────────────────────

void c_reset(Pommerman* env) {
    env->step_count = 0;
    memset(env->rewards,   0, NUM_AGENTS * sizeof(float));
    memset(env->terminals, 0, NUM_AGENTS * sizeof(unsigned char));
    pmm_generate_board(env);
    pmm_compute_obs(env);
}

// ── Init / Allocate / Free ────────────────────────────────────────────────────

void init(Pommerman* env) {
    if (env->max_steps <= 0) env->max_steps = DEFAULT_MAX_STEPS;
    if (env->game_mode != MODE_TEAM) env->game_mode = MODE_FFA;
    env->width  = RENDER_W;
    env->height = RENDER_H;
    memset(env->agent_logs, 0, sizeof(env->agent_logs));
    memset(&env->log,       0, sizeof(Log));
    c_reset(env);
}

void allocate(Pommerman* env) {
    env->observations = (float*)        calloc(NUM_AGENTS * OBS_PER_AGENT, sizeof(float));
    env->actions      = (int*)          calloc(NUM_AGENTS,                  sizeof(int));
    env->rewards      = (float*)        calloc(NUM_AGENTS,                  sizeof(float));
    env->terminals    = (unsigned char*)calloc(NUM_AGENTS,                  sizeof(unsigned char));
    init(env);
}

void c_close(Pommerman* env) {
    if (env->client) {
        if (env->client->initialized) CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

void free_allocated(Pommerman* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

// ── Step ──────────────────────────────────────────────────────────────────────

void c_step(Pommerman* env) {
    memset(env->rewards,   0, NUM_AGENTS * sizeof(float));
    memset(env->terminals, 0, NUM_AGENTS * sizeof(unsigned char));
    env->step_count++;

    // ── Phase 1: Resolve movement ─────────────────────────────────────────────

    int want_r[NUM_AGENTS], want_c[NUM_AGENTS];
    for (int a = 0; a < NUM_AGENTS; a++) {
        want_r[a] = env->agent_row[a];
        want_c[a] = env->agent_col[a];
        if (!env->agent_alive[a]) continue;

        int act = env->actions[a];
        if (act < ACTION_UP || act > ACTION_RIGHT) continue;  // STOP or BOMB

        int dir = act - 1;  // ACTION_UP=1 → dir=0 (UP), etc.
        int nr  = env->agent_row[a] + DIR_DR[dir];
        int nc  = env->agent_col[a] + DIR_DC[dir];
        if (!pmm_in_bounds(nr, nc)) continue;

        int cell = env->board[pmm_idx(nr, nc)];
        if (cell == CELL_RIGID || cell == CELL_WOOD) continue;

        if (cell == CELL_BOMB) {
            if (env->agent_can_kick[a]) {
                int bi = pmm_find_bomb(env, nr, nc);
                if (bi >= 0 && !env->bombs[bi].moving) {
                    env->bombs[bi].moving   = 1;
                    env->bombs[bi].move_dir = dir;
                }
            }
            continue;  // agent does not enter bomb cell
        }

        want_r[a] = nr;
        want_c[a] = nc;
    }

    // Resolve conflicts: same target cell → neither moves; swap → neither moves
    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!env->agent_alive[a]) continue;
        for (int b = a + 1; b < NUM_AGENTS; b++) {
            if (!env->agent_alive[b]) continue;
            bool same_target = (want_r[a] == want_r[b] && want_c[a] == want_c[b]);
            bool swap = (want_r[a] == env->agent_row[b] && want_c[a] == env->agent_col[b] &&
                         want_r[b] == env->agent_row[a] && want_c[b] == env->agent_col[a]);
            if (same_target || swap) {
                want_r[a] = env->agent_row[a]; want_c[a] = env->agent_col[a];
                want_r[b] = env->agent_row[b]; want_c[b] = env->agent_col[b];
            }
        }
    }

    // Apply movement
    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!env->agent_alive[a]) continue;
        int old_ci = pmm_idx(env->agent_row[a], env->agent_col[a]);
        int new_ci = pmm_idx(want_r[a],          want_c[a]);
        if (old_ci == new_ci) continue;
        if (env->board[old_ci] == CELL_AGENT_0 + a) env->board[old_ci] = CELL_PASSAGE;
        env->agent_row[a] = want_r[a];
        env->agent_col[a] = want_c[a];
    }
    // Re-stamp agent cells (after all agents have moved)
    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!env->agent_alive[a]) continue;
        int ci = pmm_idx(env->agent_row[a], env->agent_col[a]);
        if (env->board[ci] == CELL_PASSAGE) env->board[ci] = CELL_AGENT_0 + a;
    }

    // ── Phase 2: Bomb placement ───────────────────────────────────────────────

    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!env->agent_alive[a])          continue;
        if (env->actions[a] != ACTION_BOMB) continue;
        if (env->agent_ammo[a] <= 0)        continue;
        if (env->num_bombs >= MAX_BOMBS)     continue;
        int r = env->agent_row[a], c = env->agent_col[a];
        int ci = pmm_idx(r, c);
        if (env->board[ci] == CELL_BOMB) continue;  // bomb already here

        Bomb* b          = &env->bombs[env->num_bombs++];
        b->row           = r;
        b->col           = c;
        b->life          = BOMB_LIFE;
        b->blast_strength= env->agent_blast[a];
        b->bomber_id     = a;
        b->moving        = 0;
        b->move_dir      = 0;
        env->agent_ammo[a]--;
        // Stamp bomb over agent cell
        env->board[ci]            = CELL_BOMB;
        env->bomb_blast_map[ci]   = (float)b->blast_strength;
        env->bomb_life_map[ci]    = (float)b->life;
    }

    // ── Phase 3: Move kicked bombs ────────────────────────────────────────────

    for (int i = 0; i < env->num_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->moving) continue;
        int nr = b->row + DIR_DR[b->move_dir];
        int nc = b->col + DIR_DC[b->move_dir];
        if (!pmm_in_bounds(nr, nc)) { b->moving = 0; continue; }

        int nci  = pmm_idx(nr, nc);
        int cell = env->board[nci];
        if (cell == CELL_RIGID || cell == CELL_WOOD || cell == CELL_BOMB ||
            (cell >= CELL_AGENT_0 && cell <= CELL_AGENT_3)) {
            b->moving = 0;
            continue;
        }

        int old_ci = pmm_idx(b->row, b->col);
        env->board[old_ci]           = CELL_PASSAGE;
        env->bomb_blast_map[old_ci]  = 0.0f;
        env->bomb_life_map[old_ci]   = 0.0f;

        b->row = nr;  b->col = nc;
        env->board[nci]             = CELL_BOMB;
        env->bomb_blast_map[nci]    = (float)b->blast_strength;
        env->bomb_life_map[nci]     = (float)b->life;
    }

    // ── Phase 4: Tick bomb timers, queue expired ──────────────────────────────

    env->exp_head = env->exp_tail = 0;
    memset(env->exp_visited, 0, MAX_BOMBS * sizeof(bool));

    for (int i = 0; i < env->num_bombs; i++) {
        env->bombs[i].life--;
        int ci = pmm_idx(env->bombs[i].row, env->bombs[i].col);
        env->bomb_life_map[ci] = (float)env->bombs[i].life;
        if (env->bombs[i].life <= 0) pmm_queue_bomb(env, i);
    }

    // ── Phase 5: Process explosions (with chain reactions) ───────────────────

    pmm_process_explosions(env);

    // Return ammo and remove exploded bombs (iterate in reverse for safe removal)
    for (int i = env->num_bombs - 1; i >= 0; i--) {
        if (env->bombs[i].life <= 0) {
            int owner = env->bombs[i].bomber_id;
            if (owner >= 0 && owner < NUM_AGENTS) env->agent_ammo[owner]++;
            pmm_remove_bomb(env, i);
        }
    }

    // ── Phase 6: Apply flame damage ───────────────────────────────────────────

    int was_alive[NUM_AGENTS];
    for (int a = 0; a < NUM_AGENTS; a++) was_alive[a] = env->agent_alive[a];

    pmm_apply_flame_damage(env);

    // Death rewards
    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!was_alive[a] || env->agent_alive[a]) continue;
        // Agent a just died this step
        env->rewards[a] += REWARD_DEATH;
        env->agent_logs[a].episode_return += REWARD_DEATH;
        // Kill credit: in FFA give +REWARD_KILL to all living opponents
        if (env->game_mode == MODE_FFA) {
            for (int k = 0; k < NUM_AGENTS; k++) {
                if (k != a && env->agent_alive[k]) {
                    env->rewards[k] += REWARD_KILL;
                    env->agent_logs[k].kills += 1.0f;
                    env->agent_logs[k].episode_return += REWARD_KILL;
                }
            }
        } else {
            for (int k = 0; k < NUM_AGENTS; k++) {
                if (k != a && env->agent_alive[k] && TEAM_OF[k] != TEAM_OF[a]) {
                    env->rewards[k] += REWARD_KILL;
                    env->agent_logs[k].kills += 1.0f;
                    env->agent_logs[k].episode_return += REWARD_KILL;
                }
            }
        }
    }

    // ── Phase 7: Collect power-ups ────────────────────────────────────────────

    pmm_collect_powerups(env);

    // ── Phase 8: Decay flames ─────────────────────────────────────────────────

    for (int i = 0; i < NUM_CELLS; i++) {
        if (env->flame_map[i] <= 0) continue;
        env->flame_map[i]--;
        if (env->flame_map[i] == 0 && env->board[i] == CELL_FLAMES)
            env->board[i] = CELL_PASSAGE;
    }

    // Restore alive agent cells that may have been overwritten by flames
    for (int a = 0; a < NUM_AGENTS; a++) {
        if (!env->agent_alive[a]) continue;
        int ci = pmm_idx(env->agent_row[a], env->agent_col[a]);
        if (env->board[ci] == CELL_PASSAGE || env->board[ci] == CELL_FLAMES)
            env->board[ci] = CELL_AGENT_0 + a;
    }

    // ── Phase 9: Check termination ────────────────────────────────────────────

    bool game_over = false;

    if (env->game_mode == MODE_FFA) {
        int alive = pmm_alive_count(env);
        if (alive <= 1 || env->step_count >= env->max_steps) {
            game_over = true;
            if (alive == 1) {
                for (int a = 0; a < NUM_AGENTS; a++) {
                    if (!env->agent_alive[a]) continue;
                    env->rewards[a] += REWARD_WIN;
                    env->agent_logs[a].wins += 1.0f;
                    env->agent_logs[a].episode_return += REWARD_WIN;
                }
            }
        }
    } else {
        int result = pmm_team_result(env);
        if (result != -1 || env->step_count >= env->max_steps) {
            game_over = true;
            if (result >= 0) {
                for (int k = 0; k < 2; k++) {
                    int a = TEAM_AGENTS[result][k];
                    if (!env->agent_alive[a]) continue;
                    env->rewards[a] += REWARD_WIN;
                    env->agent_logs[a].wins += 1.0f;
                    env->agent_logs[a].episode_return += REWARD_WIN;
                }
            }
        }
    }

    if (game_over) {
        for (int a = 0; a < NUM_AGENTS; a++) {
            env->terminals[a] = 1;
            env->agent_logs[a].episode_length = (float)env->step_count;
            env->agent_logs[a].score =
                env->agent_logs[a].wins - env->agent_logs[a].deaths;
            pmm_add_log(env, &env->agent_logs[a]);
            env->agent_logs[a] = (Log){0};
        }
    }

    // ── Phase 10: Compute observations ───────────────────────────────────────

    pmm_compute_obs(env);
}

// ── Render ────────────────────────────────────────────────────────────────────

static const Color PMM_PASSAGE   = { 50,  50,  50, 255};
static const Color PMM_RIGID     = {100,  80,  60, 255};
static const Color PMM_WOOD      = {180, 140,  80, 255};
static const Color PMM_BOMB_C    = { 30,  30,  30, 255};
static const Color PMM_FLAMES_C  = {230,  90,  20, 255};
static const Color PMM_XBOMB     = { 80, 170,  80, 255};
static const Color PMM_XRANGE    = { 80, 120, 210, 255};
static const Color PMM_XKICK     = {170,  80, 210, 255};
static const Color PMM_GRID_LINE = { 30,  30,  30, 255};
static const Color PMM_AGENTS[4] = {
    {220,  50,  50, 255},
    { 50, 130, 220, 255},
    { 50, 200,  80, 255},
    {220, 180,  50, 255},
};

Client* make_client(Pommerman* env) {
    Client* c = (Client*)calloc(1, sizeof(Client));
    InitWindow(env->width, env->height, "PufferLib Pommerman");
    SetTargetFPS(10);
    c->initialized = 1;
    return c;
}

void c_render(Pommerman* env) {
    if (!env->client) env->client = make_client(env);
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground((Color){15, 15, 15, 255});

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            int ci   = pmm_idx(r, c);
            int cell = env->board[ci];
            int x    = c * GRID_SQ;
            int y    = r * GRID_SQ;

            Color col;
            switch (cell) {
                case CELL_PASSAGE:    col = PMM_PASSAGE;  break;
                case CELL_RIGID:      col = PMM_RIGID;    break;
                case CELL_WOOD:       col = PMM_WOOD;     break;
                case CELL_BOMB:       col = PMM_BOMB_C;   break;
                case CELL_FLAMES:     col = PMM_FLAMES_C; break;
                case CELL_EXTRA_BOMB: col = PMM_XBOMB;    break;
                case CELL_INCR_RANGE: col = PMM_XRANGE;   break;
                case CELL_CAN_KICK:   col = PMM_XKICK;    break;
                default:
                    col = (cell >= CELL_AGENT_0 && cell <= CELL_AGENT_3)
                        ? PMM_AGENTS[cell - CELL_AGENT_0]
                        : PMM_PASSAGE;
                    break;
            }

            DrawRectangle(x, y, GRID_SQ, GRID_SQ, col);
            DrawRectangleLines(x, y, GRID_SQ, GRID_SQ, PMM_GRID_LINE);

            // Overlay: bomb life countdown
            if (cell == CELL_BOMB) {
                int bi = pmm_find_bomb(env, r, c);
                if (bi >= 0)
                    DrawText(TextFormat("%d", env->bombs[bi].life),
                             x + GRID_SQ/2 - 5, y + GRID_SQ/2 - 7, 14, WHITE);
            }

            // Overlay: agent index
            if (cell >= CELL_AGENT_0 && cell <= CELL_AGENT_3)
                DrawText(TextFormat("%d", cell - CELL_AGENT_0),
                         x + GRID_SQ/2 - 4, y + GRID_SQ/2 - 7, 14, WHITE);
        }
    }

    // HUD bar below the grid
    int hy = BOARD_SIZE * GRID_SQ + 6;
    for (int a = 0; a < NUM_AGENTS; a++) {
        int hx = 4 + a * (RENDER_W / NUM_AGENTS);
        if (!env->agent_alive[a]) {
            DrawText(TextFormat("A%d: DEAD", a), hx, hy, 10, GRAY);
        } else {
            DrawText(TextFormat("A%d ammo=%d rng=%d kick=%d",
                a, env->agent_ammo[a], env->agent_blast[a], env->agent_can_kick[a]),
                hx, hy, 10, PMM_AGENTS[a]);
        }
    }
    DrawText(TextFormat("step %d/%d", env->step_count, env->max_steps),
             4, hy + 14, 10, LIGHTGRAY);

    EndDrawing();
}
