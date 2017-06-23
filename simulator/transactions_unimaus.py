from simulator.merchant import Merchant
from mesa.time import RandomActivation
from simulator import parameters
from simulator.log_collector import LogCollector
from mesa import Model
from datetime import timedelta
import numpy as np


class UniMausTransactionModel(Model):
    def __init__(self, model_parameters, CustomerClass, FraudsterClass):
        super().__init__()

        # make sure we didn't accidentally add new params anywhere
        assert len(model_parameters) == len(parameters.get_default_parameters())

        # load parameters
        self.parameters = model_parameters
        self.CustomerClass = CustomerClass
        self.FraudsterClass = FraudsterClass

        # random internal state
        self.random_state = np.random.RandomState(self.parameters["seed"])

        # current date
        self.curr_global_date = self.parameters['start_date']

        # set termination status
        self.terminated = False

        # create merchants, customers and fraudsters
        self.next_customer_id = 0
        self.next_fraudster_id = 0
        self.next_card_id = 0
        self.merchants = self.initialise_merchants()
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

    def step(self):

        # print some logs every mont
        if self.curr_global_date.month != (self.curr_global_date - timedelta(hours=1)).month:
            print(self.curr_global_date.date())
            print('num customers:', len(self.customers))
            print('num fraudsters:', len(self.fraudsters))
            print('')

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
        self.curr_global_date = self.curr_global_date + timedelta(hours=1)

        # check if termination criterion met
        if self.curr_global_date.date() > self.parameters['end_date'].date():
            self.terminated = True

    def customer_migration(self):

        # emigration
        self.customers = [c for c in self.customers if c.stay]
        self.fraudsters = [f for f in self.fraudsters if f.stay]

        # immigration
        self.immigration_customers()
        self.immigration_fraudsters()

    def immigration_customers(self):
        fraudster = 0

        noise_level = self.parameters['noise_level']
        # estimate how many genuine transactions there were
        num_transactions = self.parameters['trans_per_year'][fraudster]
        num_transactions /= 356 * 24
        # scale by current month
        num_trans_month = num_transactions * 12 * self.parameters['frac_month'][self.curr_global_date.month - 1, fraudster]
        num_transactions = (1 - noise_level) * num_trans_month + noise_level * num_transactions

        # estimate how many customers on avg left
        num_customers_left = num_transactions * (1 - self.parameters['stay_prob'][fraudster])

        if num_customers_left > 1:
            num_customers_left += self.random_state.normal(0, 1, 1)[0]
            num_customers_left = int(np.round(num_customers_left, 0))
            num_customers_left = np.max([0, num_customers_left])
        else:
            if num_customers_left > np.random.uniform(0, 1, 1)[0]:
                num_customers_left = 1
            else:
                num_customers_left = 0

        # add as many customers as we think that left
        self.customers.extend([self.CustomerClass(self) for _ in range(num_customers_left)])

    def immigration_fraudsters(self):
        fraudster = 1
        noise_level = self.parameters['noise_level']
        # estimate how many fraudulent transactions there were
        num_transactions = self.parameters['trans_per_year'][fraudster]
        num_transactions /= 356 * 24
        # scale by current month
        num_trans_month = num_transactions * 12 * self.parameters['frac_month'][self.curr_global_date.month - 1, fraudster]
        num_transactions = (1 - noise_level) * num_trans_month + noise_level * num_transactions

        # estimate how many fraudsters on avg left
        num_fraudsters_left = num_transactions * (1 - self.parameters['stay_prob'][fraudster])

        if num_fraudsters_left > 1:
            num_fraudsters_left += self.random_state.normal(0, 1, 1)[0]
            num_fraudsters_left = int(np.round(num_fraudsters_left, 0))
            num_fraudsters_left = np.max([0, num_fraudsters_left])
        else:
            if num_fraudsters_left > np.random.uniform(0, 1, 1)[0]:
                num_fraudsters_left = 1
            else:
                num_fraudsters_left = 0

        # add as many fraudsters as we think that left
        self.fraudsters.extend([self.FraudsterClass(self) for _ in range(num_fraudsters_left)])

    def initialise_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num_merchants"])]

    def initialise_customers(self):
        return [self.CustomerClass(self) for _ in range(self.parameters['num_customers'])]

    def initialise_fraudsters(self):
        return [self.FraudsterClass(self) for _ in range(self.parameters["num_fraudsters"])]

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
