from datetime import timedelta

import numpy as np
from authenticator import Authenticator
from customer import Customer
from fraudster import Fraudster
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from simulator.merchant import Merchant


class TransactionModel(Model):
    """A model with some number of agents."""
    def __init__(self, model_parameters):
        super().__init__()

        # load parameters
        self.parameters = model_parameters

        # set current datetime and termination status
        self.curr_datetime = self.parameters['start date']
        self.terminated = False

        # set up a schedule
        self.schedule = RandomActivation(self)

        # random state to reproduce results
        self.random_state = np.random.RandomState(self.parameters["seed"])

        # create the payment processing platform via which all transactions go
        self.authenticator = Authenticator(self, self.random_state, self.parameters["max authentication steps"])

        # create merchants
        self.merchants = [Merchant(i, self) for i in range(self.parameters["num merchants"])]

        # create customers
        self.customers = []
        for i in range(self.parameters["num customers"]):
            self.customers.append(Customer(i, self, self.random_state))

        # create fraudsters
        self.fraudsters = []
        for i in range(self.parameters["num fraudsters"]):
            self.fraudsters.append(Fraudster(i, self, self.random_state))

            # # Add the agent to a random grid cell
            # x = random.randrange(self.grid.width)
            # y = random.randrange(self.grid.height)
            # self.grid.place_agent(a, (x, y))

        # self.datacollector = DataCollector(
        #     model_reporters={"Gini": compute_gini},
        #     agent_reporters={"Wealth": lambda a: a.wealth})

        # initialise a data collector
        self.datacollector = DataCollector(
            agent_reporters={"Date": lambda c: c.model.curr_datetime,
                             "CardID": lambda c: c.unique_id,
                             "MerchantID": lambda c: c.curr_merchant.unique_id,
                             "Amount": lambda c: c.curr_transaction_amount,
                             "Currency": lambda c: c.currency,
                             "Country": lambda c: c.country,
                             "Target": lambda c: c.fraudster})

    def step(self):

        # pick the customers and agents that will make a transaction
        for c in self.random_state.choice(self.customers, size=5):
            self.schedule.add(c)
        for f in self.random_state.choice(self.fraudsters, size=3):
            self.schedule.add(f)

        # this calls the step function of each agent in the schedule (customer, fraudster)
        self.schedule.step()

        # write new transactions to log
        self.datacollector.collect(self)

        # remove agents from scheduler
        self.schedule.agents = []

        # update time
        self.curr_datetime += timedelta(hours=1)

        # check if termination criterion met
        if self.curr_datetime.date() > self.parameters['end date'].date():
            self.terminated = True

        # TODO: customer/fraudster migration

    def process_transaction(self, client, amount, merchant):
        authorise = self.authenticator.authorise_payment(client, amount, merchant)
        return authorise

