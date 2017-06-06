from simulator.abstract_customer import AbstractCustomer
import numpy as np


class UniMausCustomer(AbstractCustomer):
    def __init__(self, customer_id, transaction_model, fraudster):
        """
        Base class for customers/fraudsters that support uni-modal authentication.
        :param customer_id: 
        :param transaction_model: 
        :param fraudster: 
        """
        super().__init__(customer_id, transaction_model, fraudster)

    def pick_country(self):
        country_frac = self.model.parameters['country_frac']
        return self.model.random_state.choice(country_frac.index.values, p=country_frac.iloc[:, self.fraudster].values)

    def pick_currency(self):
        currency_prob = self.model.parameters['currency_per_country'][self.fraudster]
        currency_prob = currency_prob.loc[self.country]
        return self.model.random_state.choice(currency_prob.index.values, p=currency_prob.values.flatten())

    def pick_creditcard_number(self):
        return self.unique_id

    def decide_making_transaction(self):
        return self.get_transaction_prob() > self.model.random_state.uniform(0, 1, 1)[0]

    def get_transaction_prob(self):
        """
        The transaction probability at a given point in time (one time step is one hour).
        :return: 
        """
        # num_trans is the number of transactions the customer will make in this hour
        # we assume that we have enough customers to model that each customer can make max 1 transaction per hour
        curr_date = self.model.curr_datetime

        # we start from the total number of transactions per year
        prior_prob = self.model.parameters['trans_per_year'][self.fraudster]
        # add some noise to this
        std_trans = self.model.parameters['noise_level'] * prior_prob
        prior_prob += self.model.random_state.normal(0, std_trans, 1)[0]

        # get the (uniform) transaction probability per hour
        prior_prob /= 365 * 24

        # now weigh this according to the customer's intrinsic transaction incentive
        # the incentive [0,1] to make a transaction (can change over time)
        self.trans_incentive = 1/10  # TODO
        prior_prob *= self.trans_incentive

        # now weigh by probabilities of transactions per month/week/...
        # (did it like this so we can easily comment stuff out)
        trans_prob = np.copy(prior_prob)
        trans_prob *= 12 * self.model.parameters['frac_month'][curr_date.month - 1, self.fraudster]
        trans_prob *= 30.5 * self.model.parameters['frac_monthday'][curr_date.day - 1, self.fraudster]
        trans_prob *= 7 * self.model.parameters['frac_weekday'][curr_date.weekday(), self.fraudster]
        trans_prob *= 24 * self.model.parameters['frac_hour'][curr_date.hour, self.fraudster]

        return (1-self.model.parameters['noise_level']) * trans_prob + self.model.parameters['noise_level'] * prior_prob

    def pick_merchant(self):
        """
        Can be called at each transaction; will select a merchant to buy from.
        :return:    merchant ID
        """
        merchant_prob = self.model.parameters['merchant_per_currency'][self.fraudster]
        merchant_prob = merchant_prob.loc[self.currency]
        merchant_ID = self.model.random_state.choice(merchant_prob.index.values, p=merchant_prob.values.flatten())
        return [m for m in self.model.merchants if m.unique_id == merchant_ID][0]

    def make_transaction(self):
        """
        Make a transaction.
        :return: 
        """
        # pick merchant and transaction amount
        self.curr_merchant = self.pick_merchant()
        self.curr_transaction_amount = self.curr_merchant.get_amount(self.fraudster)

    def stay_customer(self):
        stay_prob = self.model.parameters['stay_prob'][self.fraudster]
        if stay_prob < self.model.random_state.uniform(0, 1, 1)[0]:
            return True
        else:
            return False
