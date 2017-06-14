from simulator.customer_abstract import AbstractCustomer
import numpy as np


class UniMausCustomer(AbstractCustomer):
    def __init__(self, unique_id, transaction_model, fraudster):
        """
        Base class for customers/fraudsters that support uni-modal authentication.
        :param customer_id: 
        :param transaction_model: 
        :param fraudster: 
        """
        super().__init__(unique_id, transaction_model, fraudster)

        # counter for number of transactions done
        self.num_transactions = 0

        # intrinsic motivation to make transaction
        self.transaction_motivation = self.model.parameters['transaction_motivation'][self.fraudster]

    def initialise_country(self):
        country_frac = self.model.parameters['country_frac']
        return self.model.random_state.choice(country_frac.index.values, p=country_frac.iloc[:, self.fraudster].values)

    def initialise_currency(self):
        currency_prob = self.model.parameters['currency_per_country'][self.fraudster]
        currency_prob = currency_prob.loc[self.country]
        return self.model.random_state.choice(currency_prob.index.values, p=currency_prob.values.flatten())

    def initialise_card_id(self):
        return self.model.get_next_card_id()

    def decide_making_transaction(self):
        return self.get_transaction_prob() > self.model.random_state.uniform(0, 1, 1)[0]

    def get_transaction_prob(self):
        """
        The transaction probability at a given point in time (one time step is one hour).
        :return: 
        """
        # noise level
        noise_level = self.model.parameters['noise_level']

        # we start from the total number of transactions per year
        prior_prob = self.model.parameters['trans_per_year'][self.fraudster]
        # # add some randomness
        # random_addition = prior_prob * self.model.rand_factor_today
        # if prior_prob + self.model.parameters['noise_level'] * random_addition > 0:
        #     prior_prob += self.model.parameters['noise_level'] * random_addition

        # get the (uniform) transaction probability per hour
        prior_prob /= 365 * 24

        # weigh by number of customer/fraudsters
        prior_prob *= self.transaction_motivation

        # now weigh by probabilities of transactions per month/week/...
        trans_prob = np.copy(prior_prob)
        trans_prob *= 12 * self.model.t_frac_month[self.curr_local_date.month - 1, self.fraudster]
        trans_prob *= 24 * self.model.t_frac_hour[self.curr_local_date.hour, self.fraudster]
        trans_prob *= 30.5 * self.model.t_frac_monthday[self.curr_local_date.day - 1, self.fraudster]
        trans_prob *= 7 * self.model.t_frac_weekday[self.curr_local_date.weekday(), self.fraudster]

        # we use the 'prior' prob here because there was no fraud in jan/feb in the real data
        return (1 - noise_level) * trans_prob + noise_level * prior_prob

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
        Make a transaction. Called after customer decided to purchase something.
        :return: 
        """
        # pick merchant and transaction amount
        self.curr_merchant = self.pick_merchant()
        self.curr_amount = self.curr_merchant.get_amount(self)
        self.num_transactions += 1

    def stay_customer(self):
        leave = (1-self.model.parameters['stay_prob'][self.fraudster]) > self.model.random_state.uniform(0, 1, 1)[0]
        if leave:
            self.stay = False


class Customer(UniMausCustomer):
    def __init__(self, transaction_model):
        customer_id = transaction_model.get_next_customer_id()
        super().__init__(customer_id, transaction_model, fraudster=False)
        self.card_corrupted = False

    def decide_making_transaction(self):
        do_trans = super().decide_making_transaction()
        leave_after_fraud = (1-self.model.parameters['stay_after_fraud']) > self.model.random_state.uniform(0, 1, 1)[0]
        if do_trans and self.card_corrupted and leave_after_fraud:
                    do_trans = False
                    self.stay = False
        return do_trans


class Fraudster(UniMausCustomer):
    def __init__(self, transaction_model):
        fraudster_id = transaction_model.get_next_fraudster_id()
        super().__init__(fraudster_id, transaction_model, fraudster=True)

    def initialise_card_id(self):
        if self.model.parameters['fraud_cards_in_genuine'] > self.model.random_state.uniform(0, 1, 1)[0]:
            # the fraudster picks a customer...
            # ... (1) from a familiar country
            fraudster_countries = self.model.parameters['country_frac'].index[self.model.parameters['country_frac']['fraud'] !=0].values
            # ... (2) from a familiar currency
            fraudster_currencies = self.model.parameters['currency_per_country'][1].index.get_level_values(1).unique()
            # ... (3) that has already made a transaction
            customers_active_ids = [c.unique_id for c in self.model.customers if c.num_transactions > 0]
            # now pick the fraud target (if there are no targets get own credit card)
            try:
                customer = self.model.random_state.choice([c for c in self.model.customers if (c.country in fraudster_countries) and (c.currency in fraudster_currencies) and (c.unique_id in customers_active_ids)])
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
