"""
We train a simple Q-Learning algorithm for fraud detection.
"""
import state_space
import action_space
import numpy as np


class BanditAgent:
    def __init__(self, init='zero', do_reward_shaping=False):

        self.frequency = np.zeros((state_space.SIZE, action_space.SIZE))
        self.avg_reward = np.zeros((state_space.SIZE, action_space.SIZE))

        self.do_reward_shaping = do_reward_shaping

    def take_action(self, state, customer=None):

        # get average reward for arms
        rew = self.avg_reward[state]
        # get frequency count
        freq = self.frequency[state]

        action = np.argmax(rew + np.sqrt(2*np.log(np.sum(freq))/freq))

        return action

    def update(self, state, action, reward, next_state):

        if self.do_reward_shaping:
            if reward > 0:  # transaction was successful
                if action == 0:
                    reward += 0.1
                else:
                    reward -= 0.01

        self.frequency[state, action] += 1
        self.avg_reward[state, action] = self.avg_reward[state, action] + (reward - self.avg_reward[state, action]) / self.frequency[state, action]
