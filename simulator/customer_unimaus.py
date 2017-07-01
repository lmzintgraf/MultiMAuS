from simulator.customer_abstract import AbstractCustomer
import numpy as np


class UniMausCustomer(AbstractCustomer):
    def __init__(self, unique_id, transaction_model, fraudster):
        """
        Base class for customers/fraudsters that support uni-modal authentication.
        :param unique_id: 
        :param transaction_model: 
        :param fraudster: 
        """

        # copy model parameters to class
        super().__init__(unique_id, transaction_model, fraudster)

        # initialise probability of making a transaction per month/hour/...
        self.noise_level = self.params['noise_level']

        # average number of transaction per hour in general; varies per customer
        trans_per_year = self.params['trans_per_year'][self.fraudster]
        rand_addition = self.random_state.normal(0, self.noise_level * trans_per_year, 1)[0]
        if trans_per_year + rand_addition > 0:
            trans_per_year += rand_addition
        self.avg_trans_per_hour = trans_per_year / 366 / 24

        # transaction probability per month
        self.trans_prob_month = self.params['frac_month'][:, self.fraudster]
        self.trans_prob_month = self.random_state.multivariate_normal(self.trans_prob_month, np.eye(12) * self.noise_level / 1200, 1)[0]
        self.trans_prob_month[self.trans_prob_month < 0] = 0

        # transaction probability per day in month
        self.trans_prob_monthday = self.params['frac_monthday'][:, self.fraudster]
        self.trans_prob_monthday = self.random_state.multivariate_normal(self.trans_prob_monthday, np.eye(31) * self.noise_level / 305, 1)[0]
        self.trans_prob_monthday[self.trans_prob_monthday < 0] = 0

        # transaction probability per weekday (we assume this differs per individual)
        self.trans_prob_weekday = self.params['frac_weekday'][:, self.fraudster]
        self.trans_prob_weekday = self.random_state.multivariate_normal(self.trans_prob_weekday, np.eye(7) * self.noise_level / 70, 1)[0]
        self.trans_prob_weekday[self.trans_prob_weekday < 0] = 0

        # transaction probability per hour (we assume this differs per individual)
        self.trans_prob_hour = self.params['frac_hour'][:, self.fraudster]
        self.trans_prob_hour = self.random_state.multivariate_normal(self.trans_prob_hour, np.eye(24) * self.noise_level / 240, 1)[0]
        self.trans_prob_hour[self.trans_prob_hour < 0] = 0

        # intrinsic motivation to make transaction
        self.transaction_motivation = self.params['transaction_motivation'][self.fraudster]

    def initialise_country(self):
        country_frac = self.params['country_frac']
        return self.random_state.choice(country_frac.index.values, p=country_frac.iloc[:, self.fraudster].values)

    def initialise_currency(self):
        currency_prob = self.params['currency_per_country'][self.fraudster]
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
        merchant_prob = self.params['merchant_per_currency'][self.fraudster]
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
        leave = (1-self.params['stay_prob'][self.fraudster]) > self.random_state.uniform(0, 1, 1)[0]
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

    def card_got_corrupted(self):
        self.card_corrupted = True

    def decide_making_transaction(self):
        """
        For a genuine customer, we add the option of leaving
        when the customer's card was subject to fraud
        :return:
        """
        do_trans = super().decide_making_transaction()
        leave_after_fraud = (1-self.params['stay_after_fraud']) > self.random_state.uniform(0, 1, 1)[0]
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
        if self.params['fraud_cards_in_genuine'] > self.random_state.uniform(0, 1, 1)[0]:
            # the fraudster picks a customer...
            # ... (1) from a familiar country
            fraudster_countries = self.params['country_frac'].index[self.params['country_frac']['fraud'] !=0].values
            # ... (2) from a familiar currency
            fraudster_currencies = self.params['currency_per_country'][1].index.get_level_values(1).unique()
            # ... (3) that has already made a transaction
            customers_active_ids = [c.unique_id for c in self.model.customers if c.num_transactions > 0]
            # now pick the fraud target (if there are no targets get own credit card)
            try:
                customer = self.random_state.choice([c for c in self.model.customers if (c.country in fraudster_countries) and (c.currency in fraudster_currencies) and (c.unique_id in customers_active_ids)])
                # get the information from the target
                card = customer.card_id
                self.country = customer.country
                self.currency = customer.currency
            except ValueError:
                card = super().initialise_card_id()
        else:
            card = super().initialise_card_id()
        return card
