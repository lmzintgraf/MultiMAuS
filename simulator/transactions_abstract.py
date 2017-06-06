import numpy as np
from datetime import timedelta
from abc import ABCMeta, abstractmethod
from mesa import Model
from mesa.time import RandomActivation
from simulator.log_collector import LogCollector


class AbstractTransactionModel(Model, metaclass=ABCMeta):
    """A model with some number of agents."""
    def __init__(self, model_parameters):
        super().__init__()

        # load parameters
        self.parameters = model_parameters
        self.random_state = np.random.RandomState(self.parameters["seed"])
        self.curr_datetime = self.parameters['start date']

        print("starting on ", self.curr_datetime.date())

        # set termination status
        self.terminated = False

        # create merchants, customers and fraudsters
        self.merchants = self.initialise_merchants()
        self.customers = self.initialise_customers()
        self.fraudsters = self.initialise_fraudsters()

        # set up a scheduler
        self.schedule = RandomActivation(self)

        # create data collector for the transaction logs
        self.log_collector = LogCollector(
            agent_reporters={"Date": lambda c: c.model.curr_datetime,
                             "CardID": lambda c: c.card,
                             "MerchantID": lambda c: c.curr_merchant.unique_id,
                             "Amount": lambda c: c.curr_transaction_amount,
                             "Currency": lambda c: c.currency,
                             "Country": lambda c: c.country,
                             "Target": lambda c: c.fraudster})

    def step(self):

        # this calls the step function of each agent in the schedule (customer, fraudster)
        self.schedule.agents = []
        self.schedule.agents.extend(self.customers)
        self.schedule.agents.extend(self.fraudsters)
        self.schedule.step()

        # migration of customers/fraudsters
        self.customer_migration()

        # write new transactions to log
        self.log_collector.collect(self)

        # update time
        new_datetime = self.curr_datetime + timedelta(hours=1)
        if new_datetime.month != self.curr_datetime.month:
            print(new_datetime.date())
        self.curr_datetime = new_datetime

        # check if termination criterion met
        if self.curr_datetime.date() > self.parameters['end date'].date():
            self.terminated = True

    def customer_migration(self):
        """
        Customers/Fraudsters can be added or removed after one time step of transactions.
        Not an abstract method because could just be left out.
        """
        pass

    @abstractmethod
    def initialise_merchants(self):
        """
        Initialise merchants from which customers can buy.
        :return:    a list of merchants (instances of Merchant)
        """
        pass

    @abstractmethod
    def initialise_customers(self):
        """
        Initialise the genuine customers.
        :return:    a list of customers (child instances of AbstractCustomer)
        """
        pass

    @abstractmethod
    def initialise_fraudsters(self):
        """
        Initialise the fraudulent customers.
        :return:    a list of fraudsters (child instances of AbstractCustomer)
        """
        pass
