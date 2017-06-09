import numpy as np


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
    reward += (1 - fraud) * (0.025 * amount + 0.025)
    reward *= success

    return reward
