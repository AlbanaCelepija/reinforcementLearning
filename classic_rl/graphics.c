#include <allegro5/allegro5.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_primitives.h>
#include <stdbool.h>
#include <stdio.h>
#include "qlearning.h"
enum KEYS{ UP, DOWN, LEFT, RIGHT};
#define width  1200
#define height 800

ALLEGRO_TIMER* timer;
ALLEGRO_EVENT_QUEUE* queue;
ALLEGRO_DISPLAY* disp;
ALLEGRO_FONT* font;
ALLEGRO_EVENT event;
ALLEGRO_COLOR color, black, red, blue, green, yellow;

int i;
float x = 8, y = 28;
bool done = false;
bool redraw = true;
int pos_x = width / 2;
int pos_y = height / 2;
int FPS = 60;
bool keys[4] = {false, false, false, false};

void start_graphics(){
    al_init();
    al_install_keyboard();
    al_init_primitives_addon();
    
    timer = al_create_timer(0.5);
    queue = al_create_event_queue();   // create the queue of the events generated from the mouse, keyboard etc...    
    disp = al_create_display(width, height);    // open a new window for the display of the environment and the RL agent
    font = al_create_builtin_font();

    black = al_map_rgb_f(0, 0, 0);
    red = al_map_rgb_f(1, 0, 0);
    blue = al_map_rgb_f(0, 0, 1);
    green = al_map_rgb_f(0, 1, 0);
    yellow = al_map_rgb(0xff, 0xff, 0xc0);

    // place new events into the queue 
    al_register_event_source(queue, al_get_keyboard_event_source());
    al_register_event_source(queue, al_get_display_event_source(disp));
    al_register_event_source(queue, al_get_timer_event_source(timer));
    //al_register_event_source(queue, al_get_mouse_event_source());

    al_set_window_title(disp, "Reinforcement Learning - NN and DL Course - 2023");
    al_clear_to_color(yellow);     // clear the display  
}
/*
void display_environment();
void display_error(float TD_Error);

void display_menu();
void display_parameters();
*/
void display_Qmatrix(){
    int y_left = 200;   
    int y_right = 230; 
    int x_left, x_right, red_color, green_color;
    al_draw_text(font, blue, 300 + 15, y_left - 15, 1, "L");
    al_draw_text(font, blue, 300 + 45, y_left - 15, 1, "R");
    al_draw_text(font, blue, 300 + 75, y_left - 15, 1, "U");
    al_draw_text(font, blue, 300 + 105, y_left - 15, 1, "D");
    for(int i=0;i<8;i++){
        x_left = 300;        
        x_right = 330;        
        al_draw_textf(font, blue, x_left - 30, y_left + 15, 1, "State %d", i);
        for(int j=0;j<4;j++){   
            float QValue = get_qlement_QMatrix(i, j);
            if (QValue < 4){
                green_color = 200;
                red_color = 200 + (QValue * 10);
            } else {
                red_color = 200;
                green_color = 200 - (QValue * 10);
            }            
            al_draw_filled_rectangle(x_left, y_left, x_right, y_right, al_map_rgb(red_color, green_color, 0));
            x_left += 30;
            x_right +=30;
        }
        y_left += 30;
        y_right += 30;
    }
}

void display_environment(int state){
    al_draw_line(600, 0, 600, 1200, black, 4);   // split the main panel
    al_draw_textf(font, blue, 750, 8, 1, "The environment representation %d", state);
    al_draw_textf(font, blue, 8, 8, 0, "Display Q-Matrix");
    // draw the evironment rctangle
    int y_left_rectangle = 200;
    al_draw_rectangle(700, y_left_rectangle, 700 + 400, y_left_rectangle + 200, black, 2);
    al_draw_line(800, y_left_rectangle, 800, y_left_rectangle + 200, black, 4);
    al_draw_line(900, y_left_rectangle, 900, y_left_rectangle + 200, black, 4);
    al_draw_line(1000, y_left_rectangle, 1000, y_left_rectangle + 200, black, 4);
    al_draw_line(700, y_left_rectangle + 100, 1100, y_left_rectangle + 100, black, 4);
    int y, state_id;
    for(int i = 0; i<8; i++){
        y = y_left_rectangle + 50;
        state_id = i;
        if (i > 3){
            y = y_left_rectangle + 150;
            state_id -= 4;
        }
        if(i==ql_get_bad_state()){
            al_draw_filled_rectangle(700 + (100 * state_id), y - 50, 700 + (100 * state_id) + 100, y + 50, red);
        } else if(i==ql_get_goal_state()){
            al_draw_filled_rectangle(700 + (100 * state_id), y - 50, 700 + (100 * state_id) + 100, y + 50, green);
        }
        al_draw_textf(font, blue, 700 + (100 * state_id) + 10, y - 40, 0, "%d", i);        
    }   
    y = y_left_rectangle + 50;
    state_id = state;
    if (state > 3){
        y = y_left_rectangle + 150;
        state_id -= 4;
    } 
    al_draw_filled_circle(700 + (100 * state_id) + 50, y, 10, blue);
    display_Qmatrix();
    al_flip_display(); // make the operations visible
    al_clear_to_color(al_map_rgb(0xff, 0xff, 0xc0));
}

// full learning loop
float qlearn(int episodes){
    float TD_Error;
    int state0;
    int episode_counter;
    episode_counter = 0;
    do {
        episode_counter++;
        state0 = get_initial_state();
        display_environment(state0);
        TD_Error = ql_learn_episode(state0);
        //display_error(TD_Error);        
        ql_reduce_exploration();
        printf("\nEpisode number:----------------------------------------------- %d \n", episode_counter);

    } while (episode_counter < MAX_EPISODES);
    return TD_Error;
}

void interpreter(){   
    al_start_timer(timer);    
    while(1)
    {
        al_wait_for_event(queue, &event);
        if (event.type == ALLEGRO_EVENT_TIMER) {
            int n = event.timer.count;
            printf("Got timer event %d\n", n);                 
        } else if(event.type == ALLEGRO_EVENT_KEY_DOWN){
            qlearn(MAX_EPISODES);
        } else if((event.type == ALLEGRO_EVENT_KEY_DOWN) || (event.type == ALLEGRO_EVENT_DISPLAY_CLOSE))
            break;
    }
}

void end_graphics(){
    al_destroy_font(font);
    al_destroy_display(disp);
    al_destroy_timer(timer);
    al_destroy_event_queue(queue);
}