#include <allegro5/allegro5.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_primitives.h>
#include <stdbool.h>
#include <stdio.h>
#include "qlearning.h"
#include "graphics.h"

int main(){
    srand(time(NULL));
    start_graphics();

    ql_init(8, 4, 1, 6);
    ql_set_learning_rate(0.5);
    ql_set_discount_factor(0.9);
    ql_set_exploration_range(1.0, 0.01);
    ql_set_exploration_decay(0.95);

    //display_menu();
    //display_parameters();
    display_environment(0);

    interpreter();
    end_graphics();
    return 0;
}