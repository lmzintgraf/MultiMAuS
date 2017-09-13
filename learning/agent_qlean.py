"""
We train a simple Q-Learning algorithm for fraud detection.
"""
from environment import environment
import state_space
import action_space
import numpy as np

# Q-TABLE



class QLearnAgent:
    def __init__(self):
        # learning rate
        self.lr = 0.1
        # discount factor
        self.discount = 1
        # epsilon for eps-greedy policy
        self.epsilon = 0.1
        # initialise a q-table based on the state and action space
        self.q_table = np.zeros((state_space.SIZE, action_space.SIZE))

    def authorise_transaction(self, customer):
        state = state_space.get_state(customer)
        action_vals = self.q_table[state]
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(action_vals)
        else:
            action = np.random.choice(action_space.ACTIONS)

        reward, next_state = environment.take_action(action)

        self.update(state, action, reward, next_state)

    def update(self, state, action, reward, next_state):

        self.q_table[state, action] += self.lr * (reward + self.discount * np.max(self.q_table[next_state]) - self.q_table[state, action])