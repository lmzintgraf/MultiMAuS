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
from simulator import parameters


class TransactionModel(Model):
    """A model with some number of agents."""
    def __init__(self, model_parameters):
        super().__init__()

        # load parameters
        self.parameters = model_parameters
        self.random_state = np.random.RandomState(self.parameters["seed"])
        self.curr_datetime = self.parameters['start date']
        self.curr_month = self.curr_datetime.month
        self.terminated = False

        # create the payment processing platform via which all transactions go
        self.authenticator = Authenticator(self, self.random_state, self.parameters["max authentication steps"])

        # create merchants, customers and fraudsters
        self.merchants = self.instantiate_merchants()

        self.customers = [Customer(i, self) for i in range(self.parameters["start num customers"])]
        self.customer_count = len(self.customers)
        self.customers_that_left = 0
        self.customers_that_came = 0

        self.fraudsters = [Fraudster(i, self) for i in range(self.parameters["start num fraudsters"])]
        self.fraudster_count = len(self.fraudsters)

        # TODO: social network between customers
        # grid = Grid(10, 10, 5)
        # grid.place_agent(self.customers[0], [0,1])
        # grid.move_to_empty(self.customers[0])

        # set up a schedule
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

        # print updates
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

        # write new transactions to log
        self.log_collector.collect(self)

        # update time
        self.curr_datetime += timedelta(hours=1)

        # check if termination criterion met
        if self.curr_datetime.date() > self.parameters['end date'].date():
            self.terminated = True

        self.migration()

    def migration(self):

        # LEAVING customers/fraudsters
        self.customers = [c for c in self.customers if c.stay]
        self.fraudsters = [f for f in self.fraudsters if f.stay]

        # COMING customers/fraudsters
        leave_prob = 1 - self.parameters['stay_prob']

        # the number of customers making a transaction at each step
        avg_num_cust = int(parameters.aggregated_data.loc['transactions']['non-fraud']) / 366. / 24.
        avg_cust_leave = avg_num_cust * leave_prob[0]
        avg_cust_leave += self.random_state.normal(0, 1, 1)[0]
        for i in range(int(np.round(avg_cust_leave,0))):
            self.customers.append(Customer(self.customer_count, self))
            self.customer_count += 1

        # the number of fraudsters making a transaction at each step
        avg_num_fraud = int(parameters.aggregated_data.loc['transactions']['fraud'])/366./24.
        avg_fraud_leave = avg_num_fraud * leave_prob[1]
        if avg_fraud_leave > self.random_state.uniform(0, 1, 1)[0]:
            self.fraudsters.append(Fraudster(self.fraudster_count, self))
            self.fraudster_count += 1

    def process_transaction(self, client, amount, merchant):
        authorise = self.authenticator.authorise_payment(client, amount, merchant)
        return authorise

    def instantiate_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num merchants"])]
