from mesa import Agent
import numpy as np


class Merchant(Agent):
    """
    A merchant that sells products to customers.
    """
    def __init__(self, merchant_id, transaction_model):
        super().__init__(merchant_id, transaction_model)

        # the parameters to obtain transaction amounts from this merchant
        self.random_state = np.random.RandomState(self.model.random_state.randint(0, np.iinfo(np.int32).max))
        self.distr_params = self.model.parameters['merchant_amount_distr'][:, self.unique_id, :]

        # save the min/max amount in a seperate field in case customers want to choose the amount themselves
        num_bins = int(len(self.distr_params[0, :])/2)
        self.min_amount = np.min(self.distr_params[:, :num_bins])
        self.max_amount = np.max(self.distr_params[:, :num_bins])

    def get_amount(self, customer):
        """
        Returns an amount for a customer that wants to buy something.
        We use the empirical distribution over amounts for the given merchant.
        :param customer:    The customer that wants to make a transaction,
                            child instance of AbstractCustomer
        :return:
        """
        distr_params = self.distr_params[customer.fraudster]

        num_bins = int(len(distr_params)/2)
        bin_heights = distr_params[:num_bins]
        bin_edges = distr_params[num_bins:]

        # get a random input to the sigmoid
        bin_idx = self.random_state.choice(range(len(bin_heights)), p=bin_heights)

        amount = self.random_state.uniform(bin_edges[bin_idx], bin_edges[bin_idx+1])

        return amount
