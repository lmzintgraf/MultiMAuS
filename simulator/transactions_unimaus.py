from simulator.merchant import Merchant
from simulator.customer_unimaus import Customer, Fraudster
from mesa.time import RandomActivation
from simulator import parameters
from simulator.log_collector import LogCollector
from mesa import Model
from datetime import timedelta
import numpy as np


class UniMausTransactionModel(Model):
    def __init__(self, model_parameters):
        super().__init__()

        # make sure we didn't accidentally add new params anywhere
        assert len(model_parameters) == len(parameters.get_default_parameters())

        # load parameters
        self.parameters = model_parameters
        self.random_state = np.random.RandomState(self.parameters["seed"])
        self.curr_global_date = self.parameters['start_date']
        self.curr_month = -1

        # set termination status
        self.terminated = False

        # create merchants, customers and fraudsters
        self.merchants = self.initialise_merchants()

        self.next_customer_id = 0
        self.next_fraudster_id = 0
        self.next_card_id = 0
        self.customers = self.initialise_customers()
        self.fraudsters = self.initialise_fraudsters()

        # set up a scheduler
        self.schedule = RandomActivation(self)

        # create data collector for the transaction logs
        self.log_collector = LogCollector(
            agent_reporters={"Global_Date": lambda c: c.model.curr_global_date,
                             "Local_Date": lambda c: c.curr_local_date,
                             "CardID": lambda c: c.card_id,
                             "MerchantID": lambda c: c.curr_merchant.unique_id,
                             "Amount": lambda c: c.curr_amount,
                             "Currency": lambda c: c.currency,
                             "Country": lambda c: c.country,
                             "Target": lambda c: c.fraudster})

        self.rand_factor_today = self.random_state.normal(0, 0.1, 1)[0]

        # get the true fractions of transactions per month/hour/weekday/monthday
        self.t_frac_month = self.parameters['frac_month']
        self.t_frac_hour = self.parameters['frac_hour']
        self.t_frac_monthday = self.parameters['frac_monthday']
        self.t_frac_weekday = self.parameters['frac_weekday']

    def step(self):

        # add some noise to the transaction probabilities
        self.t_frac_month = self.parameters['frac_month'] + self.parameters['noise_level']*self.random_state.normal(0, 1./120, 1)[0]
        self.t_frac_hour = self.parameters['frac_hour'] + self.parameters['noise_level']*self.random_state.normal(0, 1./240, 1)[0]
        self.t_frac_monthday = self.parameters['frac_monthday'] + self.parameters['noise_level']*self.random_state.normal(0, 1./310, 1)[0]
        self.t_frac_weekday = self.parameters['frac_weekday'] + self.parameters['noise_level']*self.random_state.normal(0, 1./70, 1)[0]

        # this calls the step function of each agent in the schedule (customer, fraudster)
        self.schedule.agents = []
        self.schedule.agents.extend(self.customers)
        self.schedule.agents.extend(self.fraudsters)
        self.schedule.step()

        # write new transactions to log
        self.log_collector.collect(self)

        # migration of customers/fraudsters
        self.customer_migration()

        # update time
        if self.curr_month != self.curr_global_date.month:
            self.curr_month = self.curr_global_date.month
            print(self.curr_global_date.date())
        self.curr_global_date = self.curr_global_date + timedelta(hours=1)

        # check if termination criterion met
        if self.curr_global_date.date() > self.parameters['end_date'].date():
            self.terminated = True

    def customer_migration(self):

        num_customers_before = len(self.customers)
        num_fraudsters_before = len(self.fraudsters)

        # emigration
        self.customers = [c for c in self.customers if c.stay]
        self.fraudsters = [f for f in self.fraudsters if f.stay]

        # immigration
        self.immigration_customers(num_customers_before)
        self.immigration_fraudsters(num_fraudsters_before)

    def immigration_customers(self, num_customers_before):
        # add as many customers as we removed
        num_new_cust = num_customers_before - len(self.customers)
        self.customers.extend([Customer(self) for _ in range(num_new_cust)])

    def immigration_fraudsters(self, num_fraudsters_before):
        # estimate number of fraudsters we add
        num_new_frauds = num_fraudsters_before - len(self.fraudsters)
        self.fraudsters.extend([Fraudster(self) for _ in range(num_new_frauds)])

    def initialise_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num_merchants"])]

    def initialise_customers(self):
        return [Customer(self) for _ in range(self.parameters['num_customers'])]

    def initialise_fraudsters(self):
        return [Fraudster(self) for _ in range(self.parameters["num_fraudsters"])]

    def get_next_customer_id(self):
        next_id = self.next_customer_id
        self.next_customer_id += 1
        return next_id

    def get_next_fraudster_id(self):
        next_id = self.next_fraudster_id
        self.next_fraudster_id += 1
        return next_id

    def get_next_card_id(self):
        next_id = self.next_card_id
        self.next_card_id += 1
        return next_id
