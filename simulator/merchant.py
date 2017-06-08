from mesa import Agent
import numpy as np


def sigmoid(x, min_amount, max_amount, x0, k):
    y = max_amount / (1 + np.exp(-k * (x - x0))) + min_amount
    return y


class Merchant(Agent):
    """
    A merchant that sells products to customers.
    """
    def __init__(self, merchant_id, transaction_model):
        super().__init__(merchant_id, transaction_model)

        # the parameters to obtain transaction amounts from this merchant
        self.random_state = self.model.random_state
        self.sigmoid_params = self.model.parameters['merchant_amount_parameters'][:, self.unique_id, :]

        # save the min/max amount in a seperate field in case customers want to choose the amount themselves
        self.min_amount = np.min([self.sigmoid_params[0][0], self.sigmoid_params[1][0]])
        self.max_amount = np.max([self.sigmoid_params[0][1], self.sigmoid_params[1][1]])

    def get_amount(self, customer):
        """
        Returns an amount for a customer that wants to buy something.
        We use the empirical distribution over amounts for the given merchant.
        :param customer:    The customer that wants to make a transaction,
                            child instance of AbstractCustomer
        :return: 
        """
        # get a random input to the sigmoid
        x = self.model.random_state.uniform(0, 1, 1)[0]

        # get an amount from the sigmoid function
        amount = sigmoid(x, *self.sigmoid_params[customer.fraudster])

        return amount

