"""
In here we create a discrete state space representation
which can be used in learning algorithms such as Q-Learning.
The state space is intended to represent a single transaction sufficiently,
meaning it should include all relevant information that an agent could use
when making a decision; and it should not include any unnecessary information.
"""

# the number of different amount categories we use
NUM_AMOUNT_CAT = 5

# the actual size of the state space; the product of the above
SIZE = NUM_AMOUNT_CAT


def get_state(customer):
    """
    Given to the authenticator is the customer, which holds
    all the information about the current transaction.
    :param customer:
    :return:
    """
    amount_cat = get_amount_category(customer.curr_amount)
    return amount_cat


def get_amount_category(amount):
    if amount < 5:
        amount_category = 0
    elif amount < 25:
        amount_category = 1
    elif amount < 50:
        amount_category = 2
    elif amount < 100:
        amount_category = 3
    elif amount < 1000:
        amount_category = 4
    else:
        amount_category = 5
    return amount_category
