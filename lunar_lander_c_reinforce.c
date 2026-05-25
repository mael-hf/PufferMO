/*
 * LunarLander - Agent REINFORCE en C (standalone, sans dépendances)
 * ==================================================================
 * Portage fidèle de l'environnement Gymnasium LunarLander-v2/v3
 * (version discrète : 4 actions)
 *
 * Physique  : Box2D simplifié (intégration Euler, mêmes paramètres Gym)
 * Réseau    : PolicyNet [8 → 64 → ReLU → 64 → ReLU → 4 → Softmax]
 * Algo      : REINFORCE Monte-Carlo + Adam + baseline (retour normalisé)
 *
 * Compilation :
 *   gcc -O2 -o lunar_c lunar_lander_c_reinforce.c -lm
 *
 * Lancement :
 *   ./lunar_c
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════
 * Constantes physiques (identiques à Gymnasium LunarLander-v2)
 * ═══════════════════════════════════════════════════════════════════ */
#define FPS                  50
#define DT                   (1.0 / FPS)

#define VIEWPORT_W           600.0
#define VIEWPORT_H           400.0
#define SCALE                30.0        /* pixels → mètres */
#define W                    (VIEWPORT_W / SCALE)   /* ≈ 20 m */
#define H                    (VIEWPORT_H / SCALE)   /* ≈ 13.3 m */

#define MAIN_ENGINE_POWER    13.0
#define SIDE_ENGINE_POWER     0.6
#define SIDE_ENGINE_HEIGHT    14.0       /* position latérale (unités lander) */
#define SIDE_ENGINE_AWAY       12.0
#define LANDER_DENSITY         5.0
#define LANDER_FRICTION        0.1
#define LEG_AWAY               0.2
#define LEG_DOWN               0.32

#define GRAVITY              -9.8        /* m/s² */

/* Terrain */
#define CHUNKS               11
#define HELIPAD_Y            (H / 4.0)

/* Espace obs/action */
#define OBS_DIM               8
#define ACT_DIM               4

/* ═══════════════════════════════════════════════════════════════════
 * Réseau [8 → 64 → 64 → 4]
 * ═══════════════════════════════════════════════════════════════════ */
#define H1   64
#define H2   64

typedef struct {
    double w1[H1][OBS_DIM]; double b1[H1];
    double w2[H2][H1];      double b2[H2];
    double w3[ACT_DIM][H2]; double b3[ACT_DIM];
} PolicyNet;

/* ═══════════════════════════════════════════════════════════════════
 * Adam optimizer
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    double mw1[H1][OBS_DIM], vw1[H1][OBS_DIM];
    double mb1[H1],          vb1[H1];
    double mw2[H2][H1],      vw2[H2][H1];
    double mb2[H2],          vb2[H2];
    double mw3[ACT_DIM][H2], vw3[ACT_DIM][H2];
    double mb3[ACT_DIM],     vb3[ACT_DIM];
    int    t;
} Adam;

/* ═══════════════════════════════════════════════════════════════════
 * Buffer d'épisode
 * ═══════════════════════════════════════════════════════════════════ */
#define MAX_EP  1000

typedef struct {
    double obs[MAX_EP][OBS_DIM];
    double h1[MAX_EP][H1];
    double h2[MAX_EP][H2];
    double probs[MAX_EP][ACT_DIM];
    int    action[MAX_EP];
    double reward[MAX_EP];
    int    len;
} EpBuf;

/* ═══════════════════════════════════════════════════════════════════
 * Environnement LunarLander
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Dynamique du lander */
    double x, y;               /* position [m] */
    double vx, vy;             /* vitesse [m/s] */
    double angle;              /* [rad] */
    double omega;              /* vitesse angulaire [rad/s] */
    /* Jambes */
    int    leg_contact[2];
    /* Terrain */
    double terrain_y[CHUNKS + 1];
    double helipad_y;
    /* Suivi */
    double prev_shaping;
    int    step_count;
} LunarEnv;

/* ─── Utilitaires ─── */

static double randf01(void) {
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

static double randn_bm(void) {
    /* Box-Muller */
    double u = randf01(), v = randf01();
    return sqrt(-2.0 * log(u + 1e-14)) * cos(2.0 * M_PI * v);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─── Terrain ─── */

static void terrain_gen(LunarEnv *e) {
    double raw[CHUNKS + 1];
    for (int i = 0; i <= CHUNKS; i++)
        raw[i] = HELIPAD_Y + (randf01() - 0.5) * H / 2.0;

    /* Plateau hélipad */
    int pl = CHUNKS / 2 - 1, pr = CHUNKS / 2 + 1;
    for (int i = pl; i <= pr; i++) raw[i] = HELIPAD_Y;

    /* Lissage */
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 1; i < CHUNKS; i++)
            raw[i] = (raw[i-1] + raw[i] + raw[i+1]) / 3.0;
        for (int i = pl; i <= pr; i++) raw[i] = HELIPAD_Y;
    }

    for (int i = 0; i <= CHUNKS; i++) e->terrain_y[i] = raw[i];
    e->helipad_y = HELIPAD_Y;
}

static double ground_at(const LunarEnv *e, double x) {
    double cw = W / CHUNKS;
    int idx = (int)(x / cw);
    if (idx < 0) idx = 0;
    if (idx >= CHUNKS) idx = CHUNKS - 1;
    double t = (x - idx * cw) / cw;
    return e->terrain_y[idx] * (1.0 - t) + e->terrain_y[idx+1] * t;
}

/* ─── Reset ─── */

static void env_reset(LunarEnv *e) {
    terrain_gen(e);

    e->x     = W / 2.0 + (randf01() - 0.5) * W / 3.0;
    e->y     = H * 0.75 + randf01() * H * 0.1;
    e->vx    = (randf01() - 0.5) * 2.0;
    e->vy    = -(randf01() * 0.5);
    e->angle = (randf01() - 0.5) * 0.4;
    e->omega = (randf01() - 0.5) * 0.4;

    e->leg_contact[0] = 0;
    e->leg_contact[1] = 0;
    e->prev_shaping   = 0.0;
    e->step_count     = 0;
}

/* ─── Observation ─── */

static void env_obs(const LunarEnv *e, double *obs) {
    /* Normalisation identique à Gymnasium */
    obs[0] = (e->x - W / 2.0)    / (W / 2.0);
    obs[1] = (e->y - HELIPAD_Y)  / (H / 2.0);
    obs[2] = e->vx * (VIEWPORT_W / SCALE / 2.0) / FPS;
    obs[3] = e->vy * (VIEWPORT_H / SCALE / 2.0) / FPS;
    obs[4] = e->angle;
    obs[5] = e->omega * 20.0 / FPS;
    obs[6] = (double)e->leg_contact[0];
    obs[7] = (double)e->leg_contact[1];
}

/* ─── Step ─── */

/*
 * Actions discrètes :
 *   0 → rien
 *   1 → moteur principal (pousse vers le haut dans le repère du lander)
 *   2 → propulseur latéral gauche
 *   3 → propulseur latéral droit
 *
 * Retourne la récompense ; *done = 1 si épisode terminé.
 */
static double env_step(LunarEnv *e, int action, int *done) {
    e->step_count++;

    double sa = sin(e->angle), ca = cos(e->angle);
    double Fx = 0.0, Fy = 0.0, torque = 0.0;

    if (action == 1) {
        /* Moteur principal : pousse dans la direction opposée à l'axe du lander */
        Fx += -sa * MAIN_ENGINE_POWER;
        Fy +=  ca * MAIN_ENGINE_POWER;
    }
    if (action == 2) {
        /* Propulseur gauche */
        Fx +=  ca * SIDE_ENGINE_POWER;
        Fy +=  sa * SIDE_ENGINE_POWER;
        torque -= SIDE_ENGINE_POWER * 0.12;
    }
    if (action == 3) {
        /* Propulseur droit */
        Fx += -ca * SIDE_ENGINE_POWER;
        Fy += -sa * SIDE_ENGINE_POWER;
        torque += SIDE_ENGINE_POWER * 0.12;
    }

    /* Intégration Euler — masse/inertie normalisées à 1 pour simplifier */
    e->vx    += Fx * DT;
    e->vy    += (Fy + GRAVITY) * DT;
    e->omega += torque * DT * 20.0;   /* facteur 20 pour un moment d'inertie réaliste */

    /* Drag aérien simplifié */
    e->vx    *= 0.999;
    e->vy    *= 0.999;
    e->omega *= 0.997;

    e->x     += e->vx * DT;
    e->y     += e->vy * DT;
    e->angle += e->omega * DT;

    /* Normalisation angle */
    while (e->angle >  M_PI) e->angle -= 2.0 * M_PI;
    while (e->angle < -M_PI) e->angle += 2.0 * M_PI;

    /* ── Contact jambes ── */
    double sa2 = sin(e->angle), ca2 = cos(e->angle);

    for (int leg = 0; leg < 2; leg++) {
        double sign = (leg == 0) ? -1.0 : 1.0;
        double lx = e->x + sign * LEG_AWAY * ca2 - LEG_DOWN * sa2;
        double ly = e->y + sign * LEG_AWAY * sa2 + LEG_DOWN * ca2;
        double gy = ground_at(e, lx);
        e->leg_contact[leg] = (ly <= gy + 0.05) ? 1 : 0;
    }

    /* ── Récompense (Gymnasium identique) ── */
    double dist_x = e->x - W / 2.0;
    double dist_y = e->y - e->helipad_y;

    double shaping =
        -100.0 * sqrt(dist_x * dist_x + dist_y * dist_y)
        -100.0 * sqrt(e->vx * e->vx + e->vy * e->vy)
        -100.0 * fabs(e->angle)
        + 10.0  * e->leg_contact[0]
        + 10.0  * e->leg_contact[1];

    double reward = shaping - e->prev_shaping;
    e->prev_shaping = shaping;

    /* Pénalité carburant */
    if (action == 1) reward -= 0.30;
    if (action == 2 || action == 3) reward -= 0.03;

    /* ── Terminaison ── */
    *done = 0;
    double gy_body = ground_at(e, e->x);
    int crashed   = (e->y <= gy_body + 0.05);
    int out_x     = (e->x < 0.0 || e->x > W);
    int out_top   = (e->y > H);
    int timeout   = (e->step_count >= 1000);

    if (crashed) {
        *done = 1;
        int good_land = e->leg_contact[0] && e->leg_contact[1]
                        && fabs(e->vx) < 0.5 && fabs(e->vy) < 0.5
                        && fabs(e->angle) < 0.25;
        reward += good_land ? 200.0 : -100.0;
    } else if (out_x || out_top || timeout) {
        *done = 1;
        reward -= 100.0;
    }

    return reward;
}

/* ═══════════════════════════════════════════════════════════════════
 * Réseau — initialisation & forward
 * ═══════════════════════════════════════════════════════════════════ */

static void net_init(PolicyNet *p) {
    double sc;
    sc = sqrt(2.0 / OBS_DIM);
    for (int i = 0; i < H1; i++) {
        p->b1[i] = 0.0;
        for (int j = 0; j < OBS_DIM; j++) p->w1[i][j] = randn_bm() * sc;
    }
    sc = sqrt(2.0 / H1);
    for (int i = 0; i < H2; i++) {
        p->b2[i] = 0.0;
        for (int j = 0; j < H1; j++) p->w2[i][j] = randn_bm() * sc;
    }
    sc = sqrt(2.0 / H2);
    for (int i = 0; i < ACT_DIM; i++) {
        p->b3[i] = 0.0;
        for (int j = 0; j < H2; j++) p->w3[i][j] = randn_bm() * sc;
    }
}

static void net_forward(const PolicyNet *p, const double *obs,
                        double *probs, double *fh1, double *fh2) {
    for (int i = 0; i < H1; i++) {
        double s = p->b1[i];
        for (int j = 0; j < OBS_DIM; j++) s += p->w1[i][j] * obs[j];
        fh1[i] = s > 0.0 ? s : 0.0;
    }
    for (int i = 0; i < H2; i++) {
        double s = p->b2[i];
        for (int j = 0; j < H1; j++) s += p->w2[i][j] * fh1[j];
        fh2[i] = s > 0.0 ? s : 0.0;
    }
    double logits[ACT_DIM], mx = -1e18, sum = 0.0;
    for (int i = 0; i < ACT_DIM; i++) {
        double s = p->b3[i];
        for (int j = 0; j < H2; j++) s += p->w3[i][j] * fh2[j];
        logits[i] = s;
        if (s > mx) mx = s;
    }
    for (int i = 0; i < ACT_DIM; i++) { probs[i] = exp(logits[i] - mx); sum += probs[i]; }
    for (int i = 0; i < ACT_DIM; i++) probs[i] /= sum;
}

static int sample_action(const double *probs) {
    double r = randf01(), cdf = 0.0;
    for (int i = 0; i < ACT_DIM - 1; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    return ACT_DIM - 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * Adam
 * ═══════════════════════════════════════════════════════════════════ */

static void adam_init(Adam *a) { memset(a, 0, sizeof(*a)); }

static inline void adam_upd(double *p, double *m, double *v,
                             double g, double lr, double b1, double b2,
                             double eps, int t) {
    *m = b1 * (*m) + (1.0 - b1) * g;
    *v = b2 * (*v) + (1.0 - b2) * g * g;
    double mh = *m / (1.0 - pow(b1, t));
    double vh = *v / (1.0 - pow(b2, t));
    *p -= lr * mh / (sqrt(vh) + eps);
}

/* ═══════════════════════════════════════════════════════════════════
 * REINFORCE : calcul des retours + backprop + Adam
 * ═══════════════════════════════════════════════════════════════════ */

static void reinforce_step(PolicyNet *pol, Adam *adam,
                            EpBuf *buf, double gamma, double lr) {
    int n = buf->len;
    if (n == 0) return;

    /* Retours cumulés normalisés */
    double G = 0.0;
    double ret[MAX_EP];
    for (int t = n - 1; t >= 0; t--) { G = buf->reward[t] + gamma * G; ret[t] = G; }
    double mean = 0.0, var = 0.0;
    for (int t = 0; t < n; t++) mean += ret[t];
    mean /= n;
    for (int t = 0; t < n; t++) var += (ret[t] - mean) * (ret[t] - mean);
    double std = sqrt(var / n + 1e-8);
    for (int t = 0; t < n; t++) ret[t] = (ret[t] - mean) / std;

    /* Gradients accumulés */
    double dw1[H1][OBS_DIM], db1[H1];
    double dw2[H2][H1],      db2[H2];
    double dw3[ACT_DIM][H2], db3[ACT_DIM];
    memset(dw1, 0, sizeof(dw1)); memset(db1, 0, sizeof(db1));
    memset(dw2, 0, sizeof(dw2)); memset(db2, 0, sizeof(db2));
    memset(dw3, 0, sizeof(dw3)); memset(db3, 0, sizeof(db3));

    for (int t = 0; t < n; t++) {
        double *fh1   = buf->h1[t];
        double *fh2   = buf->h2[t];
        double *fprob = buf->probs[t];
        double *fobs  = buf->obs[t];
        int     act   = buf->action[t];
        double  Gt    = ret[t];

        /* δ logits = -(G_t * ∇ log π) = -(G_t * (1_{i=a} - π_i)) */
        double dl[ACT_DIM];
        for (int i = 0; i < ACT_DIM; i++)
            dl[i] = -Gt * ((i == act ? 1.0 : 0.0) - fprob[i]);

        for (int i = 0; i < ACT_DIM; i++) {
            db3[i] += dl[i];
            for (int j = 0; j < H2; j++) dw3[i][j] += dl[i] * fh2[j];
        }

        double dh2[H2];
        for (int j = 0; j < H2; j++) {
            double s = 0.0;
            for (int i = 0; i < ACT_DIM; i++) s += pol->w3[i][j] * dl[i];
            dh2[j] = (fh2[j] > 0.0) ? s : 0.0;
        }

        for (int i = 0; i < H2; i++) {
            db2[i] += dh2[i];
            for (int j = 0; j < H1; j++) dw2[i][j] += dh2[i] * fh1[j];
        }

        double dh1[H1];
        for (int j = 0; j < H1; j++) {
            double s = 0.0;
            for (int i = 0; i < H2; i++) s += pol->w2[i][j] * dh2[i];
            dh1[j] = (fh1[j] > 0.0) ? s : 0.0;
        }

        for (int i = 0; i < H1; i++) {
            db1[i] += dh1[i];
            for (int j = 0; j < OBS_DIM; j++) dw1[i][j] += dh1[i] * fobs[j];
        }
    }

    /* Mise à jour Adam */
    adam->t++;
    const double B1 = 0.9, B2 = 0.999, EPS = 1e-8;

    for (int i = 0; i < H1; i++) {
        adam_upd(&pol->b1[i], &adam->mb1[i], &adam->vb1[i], db1[i], lr, B1, B2, EPS, adam->t);
        for (int j = 0; j < OBS_DIM; j++)
            adam_upd(&pol->w1[i][j], &adam->mw1[i][j], &adam->vw1[i][j], dw1[i][j], lr, B1, B2, EPS, adam->t);
    }
    for (int i = 0; i < H2; i++) {
        adam_upd(&pol->b2[i], &adam->mb2[i], &adam->vb2[i], db2[i], lr, B1, B2, EPS, adam->t);
        for (int j = 0; j < H1; j++)
            adam_upd(&pol->w2[i][j], &adam->mw2[i][j], &adam->vw2[i][j], dw2[i][j], lr, B1, B2, EPS, adam->t);
    }
    for (int i = 0; i < ACT_DIM; i++) {
        adam_upd(&pol->b3[i], &adam->mb3[i], &adam->vb3[i], db3[i], lr, B1, B2, EPS, adam->t);
        for (int j = 0; j < H2; j++)
            adam_upd(&pol->w3[i][j], &adam->mw3[i][j], &adam->vw3[i][j], dw3[i][j], lr, B1, B2, EPS, adam->t);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * BENCHMARK
 * ═══════════════════════════════════════════════════════════════════ */

static void benchmark_sps(long n_steps) {
    printf("\n%s\n", "═══════════════════════════════════════════════════════");
    printf("  BENCHMARK — Steps Par Seconde (SPS)\n");
    printf("%s\n", "═══════════════════════════════════════════════════════");

    LunarEnv env;
    env_reset(&env);

    double start = now_s();
    long   done_steps = 0;
    int    done;

    while (done_steps < n_steps) {
        int action = rand() % ACT_DIM;
        env_step(&env, action, &done);
        done_steps++;
        if (done) env_reset(&env);
    }

    double elapsed = now_s() - start;
    double sps     = done_steps / elapsed;

    printf("  Environnement  : C (standalone, LunarLander physique native)\n");
    printf("  Steps simulés  : %ld\n", done_steps);
    printf("  Temps total    : %.4f s\n", elapsed);
    printf("  SPS            : %.0f steps/seconde\n", sps);
    printf("%s\n\n", "═══════════════════════════════════════════════════════");
}

/* ═══════════════════════════════════════════════════════════════════
 * ENTRAÎNEMENT
 * ═══════════════════════════════════════════════════════════════════ */

static void train(int n_episodes, double gamma, double lr, int print_every) {
    printf("%s\n", "═══════════════════════════════════════════════════════");
    printf("  ENTRAÎNEMENT — REINFORCE sur LunarLander (C)\n");
    printf("%s\n", "═══════════════════════════════════════════════════════");

    LunarEnv  env;
    PolicyNet pol;
    Adam      adam;
    EpBuf     buf;

    net_init(&pol);
    adam_init(&adam);

    double *ep_rewards = malloc(n_episodes * sizeof(double));
    double obs[OBS_DIM], probs[ACT_DIM], fh1[H1], fh2[H2];

    double start = now_s();
    long   total_steps = 0;
    int    ep;

    for (ep = 1; ep <= n_episodes; ep++) {
        env_reset(&env);
        env_obs(&env, obs);
        buf.len = 0;

        double ep_reward = 0.0;
        int done = 0;

        while (!done && buf.len < MAX_EP) {
            int t = buf.len;
            net_forward(&pol, obs, probs, fh1, fh2);

            memcpy(buf.obs[t],   obs,   sizeof(double) * OBS_DIM);
            memcpy(buf.h1[t],    fh1,   sizeof(double) * H1);
            memcpy(buf.h2[t],    fh2,   sizeof(double) * H2);
            memcpy(buf.probs[t], probs, sizeof(double) * ACT_DIM);

            int act = sample_action(probs);
            buf.action[t] = act;
            buf.reward[t] = env_step(&env, act, &done);

            ep_reward += buf.reward[t];
            buf.len++;
            total_steps++;

            env_obs(&env, obs);
        }

        reinforce_step(&pol, &adam, &buf, gamma, lr);
        ep_rewards[ep - 1] = ep_reward;

        if (ep % print_every == 0) {
            int from = (ep > print_every) ? ep - print_every : 0;
            double m = 0.0;
            for (int i = from; i < ep; i++) m += ep_rewards[i];
            m /= (ep - from);

            double elapsed = now_s() - start;
            printf("  Ep %4d/%d | Moy(%d) : %8.1f | Steps/ep : %4d | SPS : %.0f\n",
                   ep, n_episodes, print_every, m, buf.len,
                   total_steps / elapsed);
        }

        /* Critère de résolution : moyenne >= 200 sur 100 épisodes */
        if (ep >= 100) {
            double m100 = 0.0;
            for (int i = ep - 100; i < ep; i++) m100 += ep_rewards[i];
            m100 /= 100.0;
            if (m100 >= 200.0) {
                double el = now_s() - start;
                printf("\n  ✓ LunarLander résolu à l'épisode %d !\n", ep);
                printf("    Temps : %.2f s | Steps totaux : %ld\n", el, total_steps);
                break;
            }
        }
    }

    double elapsed_total = now_s() - start;
    double sps_global    = total_steps / elapsed_total;

    /* Récompense finale */
    int from = (ep > 100) ? ep - 100 : 0;
    double final_mean = 0.0;
    for (int i = from; i < ep; i++) final_mean += ep_rewards[i];
    final_mean /= (ep - from);

    printf("\n  Résumé entraînement :\n");
    printf("    Épisodes     : %d\n", ep);
    printf("    Steps totaux : %ld\n", total_steps);
    printf("    Durée totale : %.2f s\n", elapsed_total);
    printf("    SPS moyen    : %.0f steps/seconde\n", sps_global);
    printf("\n  Récompense finale (moy 100 derniers épisodes) : %.1f\n", final_mean);
    printf("%s\n\n", "═══════════════════════════════════════════════════════");

    free(ep_rewards);
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(void) {
    srand((unsigned)time(NULL));

    printf("  LunarLander REINFORCE — Version C standalone\n");
    printf("  Compilation : gcc -O2 -o lunar_c lunar_lander_c_reinforce.c -lm\n\n");

    benchmark_sps(100000L);

    train(3000,   /* épisodes max   */
          0.99,   /* gamma          */
          3e-4,   /* lr Adam        */
          100);   /* print chaque N */

    printf("  Pour comparer avec la version Python :\n");
    printf("    python lunar_lander_python_reinforce.py\n\n");
    return 0;
}
