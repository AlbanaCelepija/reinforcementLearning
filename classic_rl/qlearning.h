
#define MAX_STATES 100      // number of states
#define MAX_ACTIONS 100     // number of actions 
#define ALPHA 1.0           // the learning ratels
#define EPSILON_INI 0.9     // initial exploration factor
#define EPSILON_FIN 0.1     // final exploration factor
#define GAMMA 0.9           // default discount factor
#define DECAY 0.95          // default epsilon decay rate
#define MAX_EPISODES 400    // max number of episodes

void ql_init_QMatrix();
void ql_set_learning_rate(float lear_rate);
void ql_set_discount_factor(float disc_fact);
void ql_set_exploration_range(float exp_ini, float exp_fin);
void ql_set_exploration_decay(float dec);
void ql_set_goal_state(int goal_s);
void ql_set_bad_state(int bad_s);
void ql_init(int num_states, int num_actions, int goal_state, int bad_state);
float ql_get_learning_rate();
float ql_get_discount_factor();
float ql_get_epsilon();
float ql_get_exploration_decay();
void ql_reduce_exploration();
float ql_maxQ(int state);
int ql_best_action(int state);
int ql_egreedy_policy(int state);
float ql_update_QMatrix(int state, int action, int reward, int new_state);
float ql_learn_episode(int state0);
int get_initial_state();
float get_qlement_QMatrix(int i, int j);
int ql_get_bad_state();
int ql_get_goal_state();