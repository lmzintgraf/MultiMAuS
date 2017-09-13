"""
We train a simple Q-Learning algorithm for fraud detection.
"""

import numpy as np

# STATE SPACE

# - transaction amount:
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


# ACTION SPACE
# the actions are simple: request a second authentication or not
action_space = [0, 1]


# Q-TABLE
# initialise a q-table based on the state and action space