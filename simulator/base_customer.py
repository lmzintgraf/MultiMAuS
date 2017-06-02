from mesa import Agent


class BaseCustomer(Agent):
    """ 
    Base class for customers, which can either be genuine or fraudulent.
    """
    def __init__(self, customer_id, transaction_model, fraudster):
        super().__init__(customer_id, transaction_model)

        # variable for whether a transaction is currently being processed
        self.active = False

        # each customer has to say if it's a fraudster or not
        self.fraudster = fraudster

        # pick country, currency, card
        self.country = self.pick_country()
        self.currency = self.pick_currency()
        self.card = self.pick_creditcard_number()

        # fields for storing the current transaction properties
        self.curr_merchant = None
        self.curr_transaction_amount = None

        # fields to collect some statistics
        self.num_successful_transactions = 0
        self.num_cancelled_transactions = 0

        # current authentication step
        self.curr_auth_step = 1

        # variable tells us whether the customer wants to stay
        self.stay = True

    def step(self):
        """ 
        This is called in each simulation step (i.e., one hour).
        Each individual customer/fraudster decides whether to make a transaction or  not.
        """
        if self.get_transaction_prob() > self.model.random_state.uniform(0, 1, 1)[0]:
            self.active = True
            self.start_transaction()
            self.decide_staying()
        else:
            self.active = False
            self.curr_merchant = None
            self.curr_transaction_amount = None

    def get_transaction_prob(self):
        """
        The transaction probability at a given point in time (one time step is one hour).
        :return: 
        """
        # num_trans is the number of transactions the customer will make in this hour
        # we assume that we have enough customers to model that each customer can make max 1 transaction per hour
        curr_date = self.model.curr_datetime
        std_transactions = self.model.parameters['std_transactions'][self.fraudster]
        trans_prob = self.model.parameters['trans_per_year'][self.fraudster]
        trans_prob += self.model.random_state.normal(0, std_transactions, 1)[0]
        trans_prob /= (1-self.fraudster)*len(self.model.customers) + self.fraudster*len(self.model.fraudsters)
        trans_prob *= self.model.parameters['frac_month'][curr_date.month - 1, self.fraudster]
        trans_prob *= self.model.parameters['frac_monthday'][curr_date.day - 1, self.fraudster]
        trans_prob *= 7 * self.model.parameters['frac_weekday'][curr_date.weekday(), self.fraudster]
        trans_prob *= self.model.parameters['frac_hour'][curr_date.hour, self.fraudster]

        # this is to ensure that fraudsters also sometimes make transactions in Jan/Feb (it's 0 in the data)
        prior_prob = self.model.parameters['trans_per_year'][self.fraudster] / 366 / 24
        prior_prob /= (1-self.fraudster)*len(self.model.customers) + self.fraudster*len(self.model.fraudsters)
        # prior_prob += self.model.random_state.normal(0, prior_prob/10, 1)[0]

        return 0.8 * trans_prob + 0.2 * prior_prob

    def pick_country(self):
        country_frac = self.model.parameters['country_frac']
        return self.model.random_state.choice(country_frac.index.values, p=country_frac.iloc[:, self.fraudster].values)

    def pick_currency(self):
        currency_prob = self.model.parameters['currency_per_country'][self.fraudster]
        currency_prob = currency_prob.loc[self.country]
        return self.model.random_state.choice(currency_prob.index.values, p=currency_prob.values.flatten())

    def pick_merchant(self):
        """
        Can be called at each transaction; will select a merchant to buy from.
        :return:    merchant ID
        """
        merchant_prob = self.model.parameters['merchant_per_currency'][self.fraudster]
        merchant_prob = merchant_prob.loc[self.currency]
        merchant_ID = self.model.random_state.choice(merchant_prob.index.values, p=merchant_prob.values.flatten())
        return [m for m in self.model.merchants if m.unique_id == merchant_ID][0]

    def start_transaction(self):
        """
        Make a transaction.
        :return: 
        """
        # randomly pick a merchant
        self.curr_merchant = self.pick_merchant()

        # randomly pick a transaction amount
        self.curr_transaction_amount = self.curr_merchant.get_amount(self.fraudster)

        # make the transaction
        success = self.model.process_transaction(client=self, amount=self.curr_transaction_amount, merchant=self.curr_merchant)
        if success:
            self.num_successful_transactions += 1
        else:
            self.num_cancelled_transactions += 1

    def pick_creditcard_number(self):
        return self.unique_id

    def decide_staying(self):
        leave_prob = 1 - self.model.parameters['stay_prob'][self.fraudster]
        if leave_prob > self.model.random_state.uniform(0, 1, 1)[0]:
            self.stay = False
