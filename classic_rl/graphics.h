#include <allegro5/allegro5.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_primitives.h>
#include <stdbool.h>
#include <stdio.h>

void start_graphics();
void display_environment(int state);
void display_error(float TD_Error);

void display_menu();
void display_parameters();
void display_environment();

void interpreter();
void end_graphics();