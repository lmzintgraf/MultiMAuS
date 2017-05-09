import numpy as np

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import parameters

from customer import Customer
from fraudster import Fraudster
from merchant import Merchant
import random


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)


class TransactionModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        super().__init__()

        # get the parameters for the simulation
        model_parameters = parameters.get_default_parameters()

        # number of agents in our simulation
        self.num_customers = model_parameters["num customers"]
        self.num_fraudsters = model_parameters["num fraudsters"]
        self.num_merchants = model_parameters["num merchants"]

        # random state to reproduce results
        self.random_state = np.random.RandomState(model_parameters["seed"])

        # we use a random activation scheme for the customers
        self.schedule = RandomActivation(self)

        # self.grid = MultiGrid(width, height, True)
        # self.running = True

        # create merchants (these are not agents because they are passive)
        merchants = [Merchant(self) for _ in range(self.num_merchants)]

        # create customers
        for i in range(self.num_customers):
            c = Customer(i, self, merchants, self.random_state)
            self.schedule.add(c)

        # create fraudsters
        for i in range(self.num_fraudsters):
            f = Fraudster(i, self, merchants, self.random_state)
            self.schedule.add(f)

            # # Add the agent to a random grid cell
            # x = random.randrange(self.grid.width)
            # y = random.randrange(self.grid.height)
            # self.grid.place_agent(a, (x, y))

        # self.datacollector = DataCollector(
        #     model_reporters={"Gini": compute_gini},
        #     agent_reporters={"Wealth": lambda a: a.wealth})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
