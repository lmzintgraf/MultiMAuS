

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

    success = int(success)
    fraud = int(fraud)

    reward = success
    reward *= fraud * (-amount)
    reward *= (1 - fraud) * (0.25 * amount + 0.25)

    return reward
