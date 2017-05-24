from mesa import Agent
import numpy as np
from abc import abstractmethod
from simulator import utils_simulator

class AbstractCustomer(Agent):
    """ 
    Base class for customers, which can either be genuine or fraudulent.
    """
    def __init__(self, customer_id, transaction_model):
        super().__init__(customer_id, transaction_model)

        self.active = False

        # pick currency, country, card
        self.currency = self.pick_currency()
        self.country = self.pick_country()
        self.card = self.pick_creditcard_number()

        # fields for storing the current transaction properties
        self.curr_merchant = None
        self.curr_transaction_amount = None

    def step(self):
        """ 
        This is called in each simulation step (i.e., one hour).
        Each individual customer/fraudster decides whether to make a transaction or  not.
        """
        if self.get_transaction_prob() > self.model.random_state.uniform(0, 1, 1)[0]:
            self.active = True
            self.start_transaction()
        else:
            self.active = False
            self.curr_merchant = None
            self.curr_transaction_amount = None

    def get_transaction_prob(self):

        frac_trans_per_hour = self.model.parameters['frac trans per hour'][self.model.curr_datetime.hour, self.fraudster]
        frac_trans_per_hour += utils_simulator.rand_skew_norm(5, 0, self.model.random_state)
        frac_trans_per_hour = np.max([frac_trans_per_hour, 0])

        mean_trans_per_month = self.model.parameters['mean trans per month'][self.model.curr_datetime.month - 1, self.fraudster]
        mean_trans_per_month += utils_simulator.rand_skew_norm(5, 0, self.model.random_state)
        mean_trans_per_month = np.max([mean_trans_per_month, 0])

        # calculate total number of transaction at given moment
        num_transactions_now = mean_trans_per_month / 31 * frac_trans_per_hour
        num_transactions_now = int(round(num_transactions_now, 0))

        # estimate the probability of this customer/fraudster making a transaction
        if self.fraudster == 0:
            transaction_prob = num_transactions_now / len(self.model.customers)
        else:
            transaction_prob = num_transactions_now / len(self.model.fraudsters)

        return transaction_prob

    @abstractmethod
    def start_transaction(self):
        return

    @abstractmethod
    def pick_currency(self):
        return

    @abstractmethod
    def pick_country(self):
        return

    @abstractmethod
    def pick_creditcard_number(self):
        return

