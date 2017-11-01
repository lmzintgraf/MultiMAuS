"""
In here we create a discrete state space representation
which can be used in learning algorithms such as Q-Learning.
The state space is intended to represent a single transaction sufficiently,
meaning it should include all relevant information that an agent could use
when making a decision; and it should not include any unnecessary information.
"""

import numpy as np

# the number of different amount categories we use
NUM_AMOUNT_CAT = 6
NUM_CURRENCY_CAT = 4

# the actual size of the state space; the product of the above
SIZE = NUM_AMOUNT_CAT * NUM_CURRENCY_CAT


def get_state(customer):
    """
    Given to the authenticator is the customer, which holds
    all the information about the current transaction.
    :param customer:
    :return:
    """
    amount_cat = get_amount_category(customer.curr_amount)
    # currency_cat = get_currency_category(customer.currency)

    # indices = np.array(range(SIZE)).reshape((NUM_AMOUNT_CAT, NUM_CURRENCY_CAT))
    # state_index = indices[amount_cat, currency_cat]

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


def get_currency_category(currency):
    if currency == 'EUR':
        currency_category = 0
    elif currency == 'USD':
        currency_category = 1
    # elif currency == 'GBP':
    #     currency_category = 2
    else:
        currency_category = 2
    return currency_category
