"""
We train a simple Q-Learning algorithm for fraud detection.
"""
import state_space
import action_space
import numpy as np
from authenticators.simple_authenticators import HeuristicAuthenticator


class SafeQLearn:
    def __init__(self, init='zero', thresh=5):
        # learning rate
        self.lr = 0.01
        # discount factor
        self.discount = 0.1
        # epsilon for eps-greedy policy
        self.epsilon = 0.1

        self.random_state = np.random.RandomState(13)

        # initialise a q-table based on the state and action space
        if init == 'zero':
            self.q_table = np.zeros((state_space.SIZE, action_space.SIZE))
        elif init == 'always second':
            self.q_table = np.zeros((state_space.SIZE, action_space.SIZE))
            self.q_table[:, 1] = 1
        elif init == 'random':
            self.q_table = self.random_state.uniform(0, 1, (state_space.SIZE, action_space.SIZE))
        else:
            raise NotImplementedError('Q-table initialisation', init, 'unknown.')

        # initialise counter for q-table
        self.counter = np.zeros((state_space.SIZE, action_space.SIZE))
        self.thresh = thresh

        # initialise heuristic agent
        self.heuristic_agent = HeuristicAuthenticator(50)

    def take_action(self, state, customer=None):

        if self.random_state.uniform(0, 1) > self.epsilon:

            if np.sum(self.counter[state, :]) < self.thresh:
                # Heuristic agent
                action = self.heuristic_agent.take_action(customer)

            else:
                # Q-table
                action_vals = self.q_table[state]
                action = np.argmax(action_vals)

        else:
            action = self.random_state.choice(action_space.ACTIONS)

        return action

    def update(self, state, action, reward, next_state):

        self.counter[state, action] += 1
        self.q_table[state, action] += self.lr * (reward + self.discount * np.max(self.q_table[next_state]) - self.q_table[state, action])

