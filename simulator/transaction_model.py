import numpy as np
from datetime import timedelta

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import Grid

from simulator.authenticator import Authenticator
from simulator.customer import Customer
from simulator.fraudster import Fraudster
from simulator.merchant import Merchant
from simulator.log_collector import LogCollector


class TransactionModel(Model):
    """A model with some number of agents."""
    def __init__(self, model_parameters):
        super().__init__()

        # load parameters
        self.parameters = model_parameters
        self.random_state = np.random.RandomState(self.parameters["seed"])
        self.curr_datetime = self.parameters['start date']
        self.terminated = False

        # create the payment processing platform via which all transactions go
        self.authenticator = Authenticator(self, self.random_state, self.parameters["max authentication steps"])

        # create merchants, customers and fraudsters
        self.merchants = self.instantiate_merchants()
        self.customers = [Customer(i, self) for i in range(self.parameters["start num customers"])]
        self.fraudsters = [Fraudster(i, self) for i in range(self.parameters["start num fraudsters"])]

        # TODO: social network between customers
        # grid = Grid(10, 10, 5)
        # grid.place_agent(self.customers[0], [0,1])
        # grid.move_to_empty(self.customers[0])

        # set up a schedule
        self.schedule = RandomActivation(self)
        a = [self.schedule.add(self.customers[i]) for i in range(len(self.customers))]
        a = [self.schedule.add(self.fraudsters[i]) for i in range(len(self.fraudsters))]

        # create data collector for the transaction logs
        self.log_collector = LogCollector(
            agent_reporters={"Date": lambda c: c.model.curr_datetime,
                             "CardID": lambda c: c.unique_id,
                             "MerchantID": lambda c: c.curr_merchant.unique_id,
                             "Amount": lambda c: c.curr_transaction_amount,
                             "Currency": lambda c: c.currency,
                             "Country": lambda c: c.country,
                             "Target": lambda c: c.fraudster})

    def step(self):

        # this calls the step function of each agent in the schedule (customer, fraudster)
        self.schedule.step()

        # write new transactions to log
        self.log_collector.collect(self)

        # update time
        self.curr_datetime += timedelta(hours=1)

        # check if termination criterion met
        if self.curr_datetime.date() > self.parameters['end date'].date():
            self.terminated = True

        # TODO: customer/fraudster migration

    def process_transaction(self, client, amount, merchant):
        authorise = self.authenticator.authorise_payment(client, amount, merchant)
        return authorise

    def instantiate_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num merchants"])]
