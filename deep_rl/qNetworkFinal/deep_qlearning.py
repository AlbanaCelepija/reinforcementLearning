import numpy as np
from collections import namedtuple
from crawlingrobot_discrete_env import CrawlingRobotDiscreteEnv
from config import Config
from DQNAgent import DQNAgent

configs = Config("config.ini")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# General settings
EPISODES = 200
batch_size = 32
init_replay_memory_size = 500

def fill_replay_memory(env, state, agent):
    '''
    Filling up the replay memory buffer
    '''
    for i in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])
        else:
            state = next_state
        #env.render()

if __name__ == '__main__':
    env = CrawlingRobotDiscreteEnv()
    agent = DQNAgent(env, configs)
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    fill_replay_memory(env, state, agent)

    total_rewards, losses = [], []
    for e in range(EPISODES):
        state = env.reset()
        if e % 10 == 0:
            env.render()
        state = np.reshape(state, [1, agent.state_size])
        for i in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(Transition(state, action, reward,
                                      next_state, done))
            state = next_state
            if e % 10 == 0:
                env.render()
            if done:
                total_rewards.append(i)
                print(f'Episode: {e}/{EPISODES}, Total reward: {i}')
                break
            loss = agent.replay(batch_size, e)
            losses.append(loss)
