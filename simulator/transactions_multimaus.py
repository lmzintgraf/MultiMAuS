import numpy as np
from datetime import timedelta

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import Grid

from simulator.authenticator import Authenticator
from simulator.customer_unimaus import UniMausCustomer
from simulator.fraudster import Fraudster
from simulator.merchant import Merchant
from simulator.log_collector import LogCollector
from simulator import parameters


class TransactionModel(Model):
    """A model with some number of agents."""
    def __init__(self, model_parameters):
        super().__init__()

        # load parameters
        self.parameters = model_parameters
        self.random_state = np.random.RandomState(self.parameters["seed"])
        self.curr_datetime = self.parameters['start date']
        self.curr_month = self.curr_datetime.month  # TODO: remove after testing

        # set termination status
        self.terminated = False

        # create the payment processing platform via which all transactions go
        self.authenticator = Authenticator(self, self.random_state, self.parameters["max authentication steps"])

        # create merchants, customers and fraudsters
        self.merchants = self.initialise_merchants()
        self.customers = self.initialise_customers()
        self.fraudsters = self.initialise_fraudsters()

        self.customer_count = len(self.customers)
        self.fraudster_count = len(self.fraudsters)

        # TODO: social network between customers
        # grid = Grid(10, 10, 5)
        # grid.place_agent(self.customers[0], [0,1])
        # grid.move_to_empty(self.customers[0])

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

        # TODO: remove after testing
        if self.curr_datetime.month != self.curr_month:
            self.curr_month = self.curr_datetime.month
            print("month", self.curr_month)
            print("num customers", len(self.customers))
            print("num fraudsters", len(self.fraudsters))

        # this calls the step function of each agent in the schedule (customer, fraudster)
        self.schedule.agents = []
        self.schedule.agents.extend(self.customers)
        self.schedule.agents.extend(self.fraudsters)
        self.schedule.step()

        # migration of customers/fraudsters
        self.migration()

        # write new transactions to log
        self.log_collector.collect(self)

        # update time
        self.curr_datetime += timedelta(hours=1)

        # check if termination criterion met
        if self.curr_datetime.date() > self.parameters['end date'].date():
            self.terminated = True

    def migration(self):

        self.emigration_customers()
        self.emigration_fraudsters()

        self.immigration_customers()
        self.immigration_fraudsters()

    def immigration_customers(self):

        # the number of customers making a transaction at each step
        avg_num_cust = int(parameters.aggregated_data.loc['transactions']['non-fraud']) / 366. / 24.
        # weigh by the current motnh
        avg_num_cust *= 12 * self.parameters['frac_month'][self.curr_datetime.month - 1, 0]
        # average number of customers leaving at each step
        avg_cust_leave = avg_num_cust * (1 - self.parameters['stay_prob'][0])

        if avg_cust_leave >= 1:
            avg_cust_leave += self.parameters['noise_level'] * self.random_state.normal(0, 1, 1)[0]
            num_new_cust = int(np.round(avg_cust_leave, 0))
            self.add_customers(range(self.customer_count, self.customer_count + num_new_cust))
            self.customer_count += num_new_cust
        else:
            if avg_cust_leave > self.random_state.uniform(0, 1, 1)[0]:
                self.customers.append(MultiMausCustomer(self.customer_count, self))
                self.customer_count += 1

    def immigration_fraudsters(self):

        # the number of customers making a transaction at each step
        avg_num_fraudst = int(parameters.aggregated_data.loc['transactions']['fraud']) / 366. / 24.
        # weigh by the current motnh
        avg_num_fraudst *= 12 * self.parameters['frac_month'][self.curr_datetime.month - 1, 1]
        # average number of customers leaving at each step
        avg_fraudst_leave = avg_num_fraudst * (1 - self.parameters['stay_prob'][1])

        if avg_fraudst_leave >= 1:
            avg_fraudst_leave += self.parameters['noise_level'] * self.random_state.normal(0, 1, 1)[0]
            num_new_fraudst = int(np.round(avg_fraudst_leave, 0))
            self.add_fraudsters(
                range(self.fraudster_count, self.fraudster_count + num_new_fraudst))
            self.fraudster_count += num_new_fraudst
        else:
            avg_fraudst_leave += self.parameters['noise_level'] * self.random_state.normal(0, 0.1, 1)[0]
            if avg_fraudst_leave > self.random_state.uniform(0, 1, 1)[0]:
                self.fraudsters.append(Fraudster(self.fraudster_count, self))
                self.fraudster_count += 1

    def emigration_customers(self):
        self.customers = [c for c in self.customers if c.stay]

    def emigration_fraudsters(self):
        self.fraudsters = [f for f in self.fraudsters if f.stay]

    def process_transaction(self, client, amount, merchant):
        authorise = self.authenticator.authorise_payment(client, amount, merchant)
        return authorise

    def initialise_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num merchants"])]

    def initialise_customers(self):
        num_cust = self.parameters['num_customers']
        return [UniMausCustomer(i, self, fraudster=False) for i in range(num_cust)]

    def initialise_fraudsters(self):
        num_fraud = self.parameters["num_fraudsters"]
        return [UniMausCustomer(i, self, fraudster=True) for i in range(num_fraud)]

    def add_customers(self, customer_ids):
        self.customers.extend([UniMausCustomer(cid, self, fraudster=False) for cid in customer_ids])

    def add_fraudsters(self, fraudster_ids):
        self.fraudsters.extend([UniMausCustomer(fid, self, fraudster=True) for fid in fraudster_ids])
