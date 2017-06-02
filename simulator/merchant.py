from mesa import Agent
import numpy as np


def sigmoid(x, max_amount, x0, k):
    y = max_amount / (1 + np.exp(-k * (x - x0)))
    return y


class Merchant(Agent):
    """
    A merchant that sells products to customers.
    At the moment, merchants are just passive.
    """
    def __init__(self, merchant_id, transaction_model):
        super().__init__(merchant_id, transaction_model)

        self.sigmoid_params = self.model.parameters['merchant_amount_parameters'][:, self.unique_id, :]

    def get_amount(self, fraudster):
        x = self.model.random_state.uniform(0, 1, 1)[0]
        amount = sigmoid(x, *self.sigmoid_params[fraudster])
        return amount

