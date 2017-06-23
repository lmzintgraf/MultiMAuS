from simulator.customer_abstract import AbstractCustomer
import string
from datetime import timedelta
import numpy as np
from string import ascii_uppercase


class UniMausCustomer(AbstractCustomer):
    def __init__(self, unique_id, transaction_model, fraudster):
        """
        Base class for customers/fraudsters that support uni-modal authentication.
        :param unique_id: 
        :param transaction_model: 
        :param fraudster: 
        """
        super().__init__(unique_id, transaction_model, fraudster)

        # initialise probability of making a transaction per month/hour/...
        self.noise_level = self.model.parameters['noise_level']

        # average number of transaction per hour in general; varies per customer
        self.avg_trans_per_hour = self.model.parameters['trans_per_year'][self.fraudster]
        rand_addition = self.random_state.normal(0, self.noise_level * self.avg_trans_per_hour, 1)[0]
        if np.abs(rand_addition) < self.avg_trans_per_hour:
            self.avg_trans_per_hour += rand_addition
        self.avg_trans_per_hour /= 366 * 24

        # transaction probability per month
        self.trans_prob_month = self.model.parameters['frac_month'][:, self.fraudster]
        self.trans_prob_month += self.random_state.normal(0, self.noise_level / 120, 12)

        # transaction probability per hour
        self.trans_prob_hour = self.model.parameters['frac_hour'][:, self.fraudster]
        self.trans_prob_hour += self.random_state.normal(0, self.noise_level / 240, 24)

        # transaction probability per day in month
        self.trans_prob_monthday = self.model.parameters['frac_monthday'][:, self.fraudster]
        self.trans_prob_monthday += self.random_state.normal(0, self.noise_level / 305, 31)

        # transaction probability per weekday
        self.trans_prob_weekday = self.model.parameters['frac_weekday'][:, self.fraudster]
        self.trans_prob_weekday += self.random_state.normal(0, self.noise_level / 70, 7)

        # intrinsic motivation to make transaction
        self.transaction_motivation = self.model.parameters['transaction_motivation'][self.fraudster]

        # convert country to integer
        self.country_int = int(''.join([str(string.ascii_uppercase.index(c)) for c in self.country]))

    def initialise_country(self):
        country_frac = self.model.parameters['country_frac']
        return self.random_state.choice(country_frac.index.values, p=country_frac.iloc[:, self.fraudster].values)

    def initialise_currency(self):
        currency_prob = self.model.parameters['currency_per_country'][self.fraudster]
        currency_prob = currency_prob.loc[self.country]
        return self.random_state.choice(currency_prob.index.values, p=currency_prob.values.flatten())

    def initialise_card_id(self):
        return self.model.get_next_card_id()

    def decide_making_transaction(self):
        return self.get_transaction_prob() > self.random_state.uniform(0, 1, 1)[0]

    def get_transaction_prob(self):
        """
        The transaction probability at a given point in time (one time step is one hour).
        :return: 
        """

        trans_prob = self.avg_trans_per_hour

        # now weigh by probabilities of transactions per month/week/...
        trans_prob *= 12 * self.trans_prob_month[self.curr_local_date.month - 1]
        trans_prob *= 24 * self.trans_prob_hour[self.curr_local_date.hour]
        trans_prob *= 30.5 * self.trans_prob_monthday[self.curr_local_date.day - 1]
        trans_prob *= 7 * self.trans_prob_weekday[self.curr_local_date.weekday()]

        # weigh by transaction motivation
        trans_prob *= self.transaction_motivation

        # we use the 'prior' prob here because there was no fraud in jan/feb in the real data
        return trans_prob

    def pick_merchant(self):
        """
        Can be called at each transaction; will select a merchant to buy from.
        :return:    merchant ID
        """
        merchant_prob = self.model.parameters['merchant_per_currency'][self.fraudster]
        merchant_prob = merchant_prob.loc[self.currency]
        merchant_ID = self.random_state.choice(merchant_prob.index.values, p=merchant_prob.values.flatten())
        return [m for m in self.model.merchants if m.unique_id == merchant_ID][0]

    def make_transaction(self):
        """
        Make a transaction. Called after customer decided to purchase something.
        :return: 
        """
        # pick merchant and transaction amount
        self.curr_merchant = self.pick_merchant()
        self.curr_amount = self.curr_merchant.get_amount(self)
        self.num_transactions += 1

    def decide_staying(self):
        leave = (1-self.model.parameters['stay_prob'][self.fraudster]) > self.random_state.uniform(0, 1, 1)[0]
        if leave:
            self.stay = False


class GenuineCustomer(UniMausCustomer):
    def __init__(self, transaction_model):
        """
        Initialise a genuine customer for the uni-modal authentication model
        :param transaction_model: 
        """

        # initialise the base customer
        customer_id = transaction_model.get_next_customer_id()
        super().__init__(customer_id, transaction_model, fraudster=False)

        # add field for whether the credit card was corrupted by a fraudster
        self.card_corrupted = False

    def decide_making_transaction(self):
        """
        For a genuine customer, we add the option of leaving
        when the customer's card was subject to fraud
        :return: 
        """
        do_trans = super().decide_making_transaction()
        leave_after_fraud = (1-self.model.parameters['stay_after_fraud']) > self.random_state.uniform(0, 1, 1)[0]
        if do_trans and self.card_corrupted and leave_after_fraud:
                    do_trans = False
                    self.stay = False
        return do_trans


class FraudulentCustomer(UniMausCustomer):
    def __init__(self, transaction_model):
        """
        Initialise a fraudulent customer for the uni-model authentication model
        :param transaction_model: 
        """
        fraudster_id = transaction_model.get_next_fraudster_id()
        super().__init__(fraudster_id, transaction_model, fraudster=True)

    def initialise_card_id(self):
        """
        Pick a card either by using a card from an existing user,
        or a completely new one (i.e., from customers unnknown to the processing platform)
        :return: 
        """
        if self.model.parameters['fraud_cards_in_genuine'] > self.random_state.uniform(0, 1, 1)[0]:
            # the fraudster picks a customer...
            # ... (1) from a familiar country
            fraudster_countries = self.model.parameters['country_frac'].index[self.model.parameters['country_frac']['fraud'] !=0].values
            # ... (2) from a familiar currency
            fraudster_currencies = self.model.parameters['currency_per_country'][1].index.get_level_values(1).unique()
            # ... (3) that has already made a transaction
            customers_active_ids = [c.unique_id for c in self.model.customers if c.num_transactions > 0]
            # now pick the fraud target (if there are no targets get own credit card)
            try:
                customer = self.random_state.choice([c for c in self.model.customers if (c.country in fraudster_countries) and (c.currency in fraudster_currencies) and (c.unique_id in customers_active_ids)])
                # get the information from the target
                card = customer.card_id
                self.country = customer.country
                self.currency = customer.currency
                # tell customer card's been corrupted
                customer.card_corrupted = True
            except ValueError:
                card = super().initialise_card_id()
        else:
            card = super().initialise_card_id()
        return card
