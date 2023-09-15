import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque


np.random.seed(1)
torch.manual_seed(1)

class DQNAgent:
    def __init__(
            self, env, configs, discount_factor=0.95,
            epsilon_greedy=1.0, epsilon_min=0.01,
            epsilon_decay=0.995, learning_rate=1e-3,
            target_network_frequency = 50,
            tau = 0.99):
        self.env = env
        self.configs = configs
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.memory = deque(maxlen=int(self.configs.algorithm_parameters["buffer_size"]))

        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self._build_qNetwork()

    def _build_qNetwork(self):
        self.model = nn.Sequential(nn.Linear(self.state_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.action_size))

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        self.target_network = nn.Sequential(nn.Linear(self.state_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.action_size))
        self.target_network.load_state_dict(self.model.state_dict())

    def remember(self, transition):
        '''
        Fill in the replay memory
        '''
        self.memory.append(transition)

    def choose_action(self, state):
        '''
        Choose the action to take based on the value of the epsilon. 
        Exploration or Exploitation phases
        '''
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32))[0]
        return torch.argmax(q_values).item()

    def _learn(self, batch_samples, global_step):
        '''
        Algo Logic: Training
        '''
        batch_states, batch_targets = [], []
        for transition in batch_samples:
            state, action, reward, next_s, done = transition

            with torch.no_grad():
                if done:
                    target = reward
                else:
                    pred = self.model(torch.tensor(next_s, dtype=torch.float32))[0]
                    target = reward + self.gamma * pred.max()
                    #target_max = self.target_network(torch.tensor(next_s, dtype=torch.float32))[0]
                    #target = reward + self.gamma * target_max.max() # * (1 - done)                

                target_all = self.model(torch.tensor(state, dtype=torch.float32))[0]
                target_all[action] = target
                print(f"{action} {target}")

            batch_states.append(state.flatten())
            batch_targets.append(target_all)
            self._adjust_epsilon()

        self.optimizer.zero_grad()
        pred = self.model(torch.tensor(batch_states, dtype=torch.float32))

        loss = self.loss_fn(pred, torch.stack(batch_targets))
        loss.backward()
        self.optimizer.step()

        # update target network        
        if global_step % self.target_network_frequency == 0:
            print("updating target network")
            for target_network_param, q_network_param in zip(self.target_network.parameters(), self.model.parameters()):
                target_network_param.data.copy_(self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data)

        return loss.item()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size , global_step):
        '''
        Get random examples from the replay menory buffer
        '''
        samples = random.sample(self.memory, batch_size)
        return self._learn(samples, global_step)

