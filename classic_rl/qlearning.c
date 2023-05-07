#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "qlearning.h"

#define MAX_STATES 100      // number of states
#define MAX_ACTIONS 100     // number of actions 
#define ALPHA 1.0           // the learning ratels
#define EPSILON_INI 0.9     // initial exploration factor
#define EPSILON_FIN 0.1     // final exploration factor
#define GAMMA 0.9           // default discount factor
#define DECAY 0.95          // default epsilon decay rate

//-------------------------------------------------------------
// QL matrix
//-------------------------------------------------------------
static int Q_matrix[MAX_STATES][MAX_ACTIONS];      // define the Q-matrix

//-------------------------------------------------------------
// Global variables
//-------------------------------------------------------------
static int number_states;     // actual number of states
static int number_actions;    // actual number of actions
static int goal_state;        // goal state
static int bad_state;        // bad state

static float alpha;           // actual learning rate
static float my_gamma;        // discount factor
static float decay;           // decay rate for epsilon
static float epsilon;         // actual exploration probability
static float eps_initial;     // initial exploration probability
static float eps_final;       // final exploration probability
static float eps_normal = 1.0;// normal exploration probability

//---------------------------------------------------
// SET GLOBAL VAR
//---------------------------------------------------

void ql_init_QMatrix(){
    int state, action;
    for(state=0; state<number_states; state++){
        for(action=0; action<number_actions; action++){
            Q_matrix[state][action] = 0;
        }
    }
}
void ql_set_learning_rate(float lear_rate){
    alpha = lear_rate;
}
void ql_set_discount_factor(float disc_fact){
    my_gamma = disc_fact;
}
void ql_set_exploration_range(float exp_ini, float exp_fin){
    eps_initial = exp_ini;
    eps_final = exp_fin;
}
void ql_set_exploration_decay(float dec){
    decay = dec;
}
void ql_set_goal_state(int goal_s){
    goal_state = goal_s;
}
void ql_set_bad_state(int bad_s){
    bad_state = bad_s;
}
void ql_init(int num_states, int num_actions, int goal_s, int bad_s){
    number_actions = num_actions;
    number_states = num_states;
    if (num_actions > MAX_ACTIONS){
        printf("Number of actions is too big\n");
        exit(1);
    }
    if (num_states > MAX_STATES){
        printf("Number of states is too big\n");
        exit(1);
    }
    ql_set_learning_rate(ALPHA);
    ql_set_discount_factor(GAMMA);
    ql_set_exploration_range(EPSILON_INI, EPSILON_FIN);
    ql_set_exploration_decay(DECAY);
    ql_init_QMatrix();
    ql_set_goal_state(goal_s);
    ql_set_bad_state(bad_s);
    epsilon = 1.0;
}
//----------------------------------------------------
// GET GLOBAL VAR
//----------------------------------------------------
float ql_get_learning_rate(){
    return alpha;
}
float ql_get_discount_factor(){
    return my_gamma;
}
float ql_get_epsilon(){
    return epsilon;
}
float ql_get_exploration_decay(){
    return decay;
}
int ql_get_goal_state(){
    return goal_state;
}
int ql_get_bad_state(){
    return bad_state;
}
//---------------------------------------------------------
// Learning 
//---------------------------------------------------------
void ql_reduce_exploration(){
    eps_normal = decay * eps_normal;
    epsilon = eps_final + eps_normal * (eps_initial - eps_final);
}
float ql_maxQ(int state){
    int action;
    float max_value = Q_matrix[state][0];
    for(action=1; action<number_actions; action++){
        if (Q_matrix[state][action] > max_value)
            max_value = Q_matrix[state][action];
    }
    return max_value;
}
int ql_best_action(int state){
    int action, best_action = 0;
    float max_value = Q_matrix[state][0];
    for(action=1; action<number_actions; action++){
        if (Q_matrix[state][action] > max_value){
            max_value = Q_matrix[state][action];
            best_action = action;
        }
    }
    return best_action;
}
int ql_egreedy_policy(int state){
    int best_action = ql_best_action(state);
    int random_action = rand()%number_actions;
    float random_epsilon = (float)rand()/(float)(RAND_MAX);
    if (random_epsilon < epsilon)
        return random_action;
    else
        return best_action;
}
float ql_update_QMatrix(int state, int action, int reward, int new_state){
    /*
    Q(s,a) = Q(s,a) + alpha*[r + gamma*Q(s',a') - Q(s,a)]
    */
   float Q_target = reward + my_gamma * ql_maxQ(new_state);
   float temp_diff_error = Q_target - Q_matrix[state][action];
   Q_matrix[state][action] = Q_matrix[state][action] + alpha * temp_diff_error;
   return fabs(temp_diff_error);
}

float get_qlement_QMatrix(int i, int j){
    return Q_matrix[i][j];
}

int next_state(int state, int action){
    /*
    */
   if(state == 0){
        switch(action){
            case 0:
            case 2:
                return state;
            case 1:
                return 1;
            case 3:
                return 4;
        }
    } else if(state == 1){
        switch(action){
            case 0:
                return 0;
            case 2:
                return state;
            case 1:
                return 2;
            case 3:
                return 5;
        }
    } else if(state == 2){
        switch(action){
            case 0:
                return 1;
            case 2:
                return state;
            case 1:
                return 3;
            case 3:
                return 6;
        }
    } else if(state == 3){
        switch(action){
            case 0:
                return 2;
            case 2:
            case 1:
                return state;
            case 3:
                return 7;
        }
    } else if(state == 4){
        switch(action){
            case 0:
            case 3:
            return state;
            case 2:
                return 0;
            case 1:
                return 5;
        }
    } else if(state == 5){
        switch(action){
            case 0:
                return 4;
            case 3:
                return state;
            case 2:
                return 1;
            case 1:
                return 6;
        }
    } else if(state == 6){
        switch(action){
            case 0:
                return 5;
            case 3:
                return state;
            case 2:
                return 2;
            case 1:
                return 7;
        }
    } else if(state == 7){
        switch(action){
            case 1:
            case 3:
                return state;
            case 2:
                return 3;
            case 0:
                return 6;
        }
    }
}

int get_reward(int from_state, int new_state){
    /*
       Reward -1  - move
       Reward -5  - hit wall 
       Reward -10 - bad state
       reward 20  - goal state
    */
   if(new_state == goal_state){
    return 20;
   } else if(new_state == from_state){
    return -5;
   } else if(new_state == bad_state){
    return -10;
   } else{
    return -1;
   }
}
//-------------------------------------------------
// 
//-------------------------------------------------
float ql_learn_episode(int state0){
    int state, action, new_state, reward, steps;
    float TD_Error = 0;
    steps = 0;
    state = state0;
    while (state != goal_state){
        action = ql_egreedy_policy(state);
        new_state = next_state(state, action); //T[state][action];
        reward = get_reward(state, new_state);//R[state][action];
        TD_Error += ql_update_QMatrix(state, action, reward, new_state); 
        state = new_state;
        steps++;
    }
    printf("\nQ-Matrix \n");
    for (int i=0; i<number_states; i++){
        for(int j=0; j<number_actions; j++){
            printf("%d \t", Q_matrix[i][j]);
        }
        printf("\n");
    }
    return TD_Error/steps;
}
int get_initial_state(){
    int random_state = rand()%number_states;
    return random_state;
}
