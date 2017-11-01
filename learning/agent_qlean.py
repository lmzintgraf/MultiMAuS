"""
We train a simple Q-Learning algorithm for fraud detection.
"""
import state_space
import action_space
import numpy as np


# Q-TABLE
class QLearnAgent:
    def __init__(self, init='zero', lr=0.01, discount=0.1, epsilon=0.1, do_reward_shaping=False):

        # learning rate
        self.lr = lr
        # discount factor
        self.discount = discount
        # epsilon for eps-greedy policy
        self.epsilon = epsilon

        self.random_state = np.random.RandomState(42)

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

        self.q_counter = np.zeros((state_space.SIZE, action_space.SIZE))
        self.q_tracker = []

        self.do_reward_shaping = do_reward_shaping

    def take_action(self, state, customer=None):
        action_vals = self.q_table[state]
        if self.random_state.uniform(0, 1) > self.epsilon:
            action = np.argmax(action_vals)
        else:
            action = self.random_state.choice(action_space.ACTIONS)

        return action

    def update(self, state, action, reward, next_state):

        # do some reward shaping: reward success after one authentication
        if self.do_reward_shaping:
            if reward > 0:  # transaction was successful
                if action == 0:
                    reward += 0.2

        self.q_table[state, action] += self.lr * (reward + self.discount * np.max(self.q_table[next_state]) - self.q_table[state, action])
        self.q_counter[state, action] += 1
