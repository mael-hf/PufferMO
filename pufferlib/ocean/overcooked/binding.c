#include "overcooked.h"
#include <stdio.h>
#include <stdlib.h>

#define Env Overcooked
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // 1. Valeurs par défaut au cas où
    env->layout_id = 0;
    env->num_agents = 2;
    env->grid_size = 32;
    env->max_ticks = 400;

    // 2. PufferLib force 5 arguments positionnels dans 'args'
    // args[0] = self (l'objet Python)
    // args[1] = layout
    // args[2] = num_agents
    // args[3] = grid_size
    // args[4] = max_ticks
    if (args != NULL && PyTuple_Size(args) >= 5) {
        env->layout_id = (LayoutType)PyLong_AsLong(PyTuple_GetItem(args, 1));
        env->num_agents = (int)PyLong_AsLong(PyTuple_GetItem(args, 2));
        env->grid_size = (int)PyLong_AsLong(PyTuple_GetItem(args, 3));
        env->max_ticks = (int)PyLong_AsLong(PyTuple_GetItem(args, 4));
    } 

    // 3. Initialisation du moteur de jeu
    init(env);
    return 0;
}

// Fonction log obligatoire pour la compilation PufferLib
static int my_log(PyObject* dict, Log* log) {
    return 0;
}