from simulator.merchant import Merchant
from mesa.time import RandomActivation
from simulator.log_collector import LogCollector
from simulator import parameters
from mesa import Model
from authenticators.simple_authenticators import NeverSecondAuthenticator
from simulator.customers import GenuineCustomer, FraudulentCustomer
from datetime import timedelta
import numpy as np


class TransactionModel(Model):
    def __init__(self, model_parameters, authenticator=NeverSecondAuthenticator(), scheduler=None):
        super().__init__(seed=123)

        # load parameters
        if model_parameters is None:
            model_parameters = parameters.get_default_parameters()
        self.parameters = model_parameters

        # calculate the intrinsic transaction motivation per customer (which is proportional to number of customers/fraudsters)
        # we keep this separate because then we can play around with the number of customers/fraudsters,
        # but individual behaviour doesn't change
        self.parameters['transaction motivation'] = np.array([1./3333, 1./55])

        # save authenticator for checking transactions
        self.authenticator = authenticator

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
        self.schedule = scheduler if scheduler is not None else RandomActivation(self)

        # we add to the log collector whether transaction was successful
        self.log_collector = self.initialise_log_collector()

    @staticmethod
    def initialise_log_collector():
        return LogCollector(
            agent_reporters={"Global_Date": lambda c: c.model.curr_global_date,
                             "Local_Date": lambda c: c.local_datetime,
                             "CardID": lambda c: c.card_id,
                             "MerchantID": lambda c: c.curr_merchant.unique_id,
                             "Amount": lambda c: c.curr_amount,
                             "Currency": lambda c: c.currency,
                             "Country": lambda c: c.country,
                             "Target": lambda c: c.fraudster,
                             "AuthSteps": lambda c: c.curr_auth_step,
                             "TransactionCancelled": lambda c: c.curr_trans_cancelled,
                             "TransactionSuccessful": lambda c: c.curr_trans_success},
            model_reporters={
                "Satisfaction": lambda m: np.sum([m.customers[i].satisfaction for i in range(len(m.customers))]) / len(m.customers)})

    def inform_attacked_customers(self):
        fraud_card_ids = [f.card_id for f in self.fraudsters if f.active and f.curr_trans_authorised]
        for customer in self.customers:
            if customer.card_id in fraud_card_ids:
                customer.card_got_corrupted()

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

        # inform the customers whose card got corrupted
        self.inform_attacked_customers()

        # write new transactions to log
        self.log_collector.collect(self)

        # migration of customers/fraudsters
        self.customer_migration()

        # update time
        self.curr_global_date = self.curr_global_date + timedelta(hours=1)

        # check if termination criterion met
        if self.curr_global_date.date() > self.parameters['end_date'].date():
            self.terminated = True

    def process_transaction(self, customer):
        return self.authenticator.authorise_transaction(customer)

    def customer_migration(self):

        # emigration
        self.customers = [c for c in self.customers if c.stay]
        self.fraudsters = [f for f in self.fraudsters if f.stay]

        # immigration
        self.immigration_customers()
        self.immigration_fraudsters()

    def immigration_customers(self):

        fraudster = 0

        # estimate how many genuine transactions there were
        num_transactions = self.parameters['trans_per_year'][fraudster] / 366 / 24

        # scale by current month
        num_trans_month = num_transactions * 12 * self.parameters['frac_month'][self.curr_global_date.month - 1, fraudster]
        num_transactions = (1 - self.parameters['noise_level']) * num_trans_month + \
                           self.parameters['noise_level'] * num_transactions

        # estimate how many customers on avg left; this many we will add
        num_new_customers = num_transactions * (1 - self.parameters['stay_prob'][fraudster])

        # weigh by mean satisfaction
        num_new_customers *= self.get_social_satisfaction()

        if num_new_customers > 1:
            num_new_customers += self.random_state.normal(0, 1, 1)[0]
            num_new_customers = int(np.round(num_new_customers, 0))
            num_new_customers = np.max([0, num_new_customers])
        else:
            if num_new_customers > self.random_state.uniform(0, 1, 1)[0]:
                num_new_customers = 1
            else:
                num_new_customers = 0

        # add as many customers as we think that left
        self.customers.extend([GenuineCustomer(self) for _ in range(num_new_customers)])

    def get_social_satisfaction(self):
        """
        Return the satisfaction of a customer's social network
        :return: 
        """
        return np.mean([c.satisfaction for c in self.customers])

    def immigration_fraudsters(self):

        fraudster = 1
        # estimate how many fraudulent transactions there were
        num_transactions = self.parameters['trans_per_year'][fraudster] / 366 / 24
        # scale by current month
        num_trans_month = num_transactions * 12 * self.parameters['frac_month'][self.curr_global_date.month - 1, fraudster]
        num_transactions = (1 - self.parameters['noise_level']) * num_trans_month + \
                           self.parameters['noise_level'] * num_transactions

        # estimate how many fraudsters on avg left
        num_fraudsters_left = num_transactions * (1 - self.parameters['stay_prob'][fraudster])

        if num_fraudsters_left > 1:
            num_fraudsters_left += self.random_state.normal(0, 1, 1)[0]
            num_fraudsters_left = int(np.round(num_fraudsters_left, 0))
            num_fraudsters_left = np.max([0, num_fraudsters_left])
        else:
            if num_fraudsters_left > self.random_state.uniform(0, 1, 1)[0]:
                num_fraudsters_left = 1
            else:
                num_fraudsters_left = 0

        # add as many fraudsters as we think that left
        self.add_fraudsters(num_fraudsters_left)

    def add_fraudsters(self, num_fraudsters):
        """
        Adds n new fraudsters to the simulation

        :param num_fraudsters:
            The number n of new fraudsters to add
        """
        self.fraudsters.extend([FraudulentCustomer(self) for _ in range(num_fraudsters)])

    def initialise_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num_merchants"])]

    def initialise_customers(self):
        return [GenuineCustomer(self) for _ in range(self.parameters['num_customers'])]

    def initialise_fraudsters(self):
        return [FraudulentCustomer(self) for _ in range(self.parameters["num_fraudsters"])]

    def get_next_customer_id(self, fraudster):
        if not fraudster:
            next_id = self.next_customer_id
            self.next_customer_id += 1
        else:
            next_id = self.next_fraudster_id
            self.next_fraudster_id += 1
        return next_id

    def get_next_card_id(self):
        next_id = self.next_card_id
        self.next_card_id += 1
        return next_id
