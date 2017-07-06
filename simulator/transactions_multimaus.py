from simulator.log_collector import LogCollector
from simulator.transactions_unimaus import UniMausTransactionModel
import numpy as np


class MultiMausTransactionModel(UniMausTransactionModel):
    def __init__(self, model_parameters, CustomerClass, FraudsterClass, authenticator):

        # an authenticator for processing the transactions
        self.authenticator = authenticator

        super().__init__(model_parameters, CustomerClass, FraudsterClass)

        # we add to the log collector whether transaction was successful
        self.log_collector = LogCollector(
            agent_reporters={"Global_Date": lambda c: c.model.curr_global_date,
                             "Local_Date": lambda c: c.curr_local_date,
                             "CardID": lambda c: c.card_id,
                             "MerchantID": lambda c: c.curr_merchant.unique_id,
                             "Amount": lambda c: c.curr_amount,
                             "Currency": lambda c: c.currency,
                             "Country": lambda c: c.country,
                             "Target": lambda c: c.fraudster,
                             "AuthSteps": lambda c: c.curr_auth_step,
                             "Success": lambda c: c.curr_trans_authorised},
            model_reporters={"Satisfaction": lambda m: np.sum([m.customers[i].satisfaction for i in range(len(m.customers))]) / len(m.customers)})

    def inform_attacked_customers(self):
        fraud_card_ids = [f.card_id for f in self.fraudsters if f.active and f.curr_trans_authorised]
        for customer in self.customers:
            if customer.card_id in fraud_card_ids:
                customer.card_got_corrupted()

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
        num_new_customers *= np.mean([c.satisfaction for c in self.customers])

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
        self.customers.extend([self.CustomerClass(self) for _ in range(num_new_customers)])

    def get_social_satisfaction(self):
        """
        Return the satisfaction of a customer's social network
        :return: 
        """
        return np.mean([c.satisfaction for c in self.customers])

    def initialise_customers(self):
        return [self.CustomerClass(self, satisfaction=np.copy(self.parameters['init_satisfaction'])) for _ in range(self.parameters['num_customers'])]
