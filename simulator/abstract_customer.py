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

        # variable for whether a transaction is currently being processed
        self.active = False

        # each customer has to say if it's a fraudster or not
        self.fraudster = None

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
        """
        The transaction probability at a given point in time (one time step is one hour).
        :return: 
        """
        # num_trans is the number of transactions the customer will make in this hour
        # we assume that we have enough customers to model that each customer can make max 1 transaction per hour
        curr_date = self.model.curr_datetime
        std_transactions = self.model.parameters['std_transactions'][self.fraudster]
        trans_prob = self.model.parameters['trans_per_year'][self.fraudster]
        trans_prob += self.model.random_state.normal(0, std_transactions, 1)[0]
        trans_prob /= (1-self.fraudster)*len(self.model.customers) + self.fraudster*len(self.model.fraudsters)
        trans_prob *= self.model.parameters['frac_month'][curr_date.month - 1, self.fraudster]
        trans_prob *= self.model.parameters['frac_monthday'][curr_date.day - 1, self.fraudster]
        trans_prob *= 7 * self.model.parameters['frac_weekday'][curr_date.weekday(), self.fraudster]
        trans_prob *= self.model.parameters['frac_hour'][curr_date.hour, self.fraudster]

        # this is to ensure that fraudsters also sometimes make transactions in Jan/Feb (it's 0 in the data)
        prior_prob = self.model.parameters['trans_per_year'][self.fraudster] / 366 / 24
        prior_prob /= (1-self.fraudster)*len(self.model.customers) + self.fraudster*len(self.model.fraudsters)
        # prior_prob += self.model.random_state.normal(0, prior_prob/10, 1)[0]

        return 0.9 * trans_prob + 0.1 * prior_prob

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

