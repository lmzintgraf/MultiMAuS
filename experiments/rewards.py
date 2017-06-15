import numpy as np


def get_rewards_per_timestep(agent_vars):
    num_steps = 366*24
    rewards = np.zeros(num_steps)
    for step in range(num_steps):
        try:
            success = agent_vars.loc[step]['Success']
            # print(success)
            fraud = agent_vars.loc[step]['Target']
            # print(fraud)
            amount = agent_vars.loc[step]['Amount']
            # print(amount)
            rewards[step] = np.sum(get_reward(success, fraud, amount))
        except KeyError:
            pass
    return rewards


def get_reward(success, fraud, amount):
    """
    The rewards are as follows
    - unsuccessful: 0
    - successful & fraudulent: -amount
    - successful & genuine: 0.25*amount + 0.25
    :param success:     boolean, whether transaction was successful
    :param fraud:       boolean, whether transaction was fraudulent
    :param amount:      amount of transaction
    :return: 
    """

    success = np.array(success, dtype=float)
    fraud = np.array(fraud, dtype=float)
    amount = np.array(amount, dtype=float)

    reward = fraud * (-amount)
    reward += (1 - fraud) * (0.025 * amount + 0.25)
    reward *= success

    return reward


def get_satisfaction_per_timestep(model_vars):
    num_steps = 366*24
    mean_satisfactions = np.zeros(num_steps)
    for step in range(num_steps):
        try:
            satisfaction = model_vars.loc[step]['Satisfaction']
            mean_satisfactions[step] = np.mean(satisfaction)
        except KeyError:
            pass
    return mean_satisfactions
