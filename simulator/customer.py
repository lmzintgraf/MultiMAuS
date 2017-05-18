from mesa import Agent


class Customer(Agent):
    """ A customer that can make transactions """
    def __init__(self, customer_id, transaction_model, random_state):
        super().__init__(customer_id, transaction_model)
        self.random_state = random_state

        # I am not a fraudster
        self.fraudster = 0

        # decide which currency to use
        self.currency = self.pick_currency()
        self.country = 'india'

        # the customer's credit card
        self.card = None

        # transaction frequency (avg. purchases per day)
        self.min_transaction_frequency = 1./365
        self.transaction_frequency = self._initialise_transaction_frequency()

        # the average amount the user spends
        self.min_average_amount = 0.0005  # equals ten cents
        self.average_amount = self._initialise_average_amount()

        # values of the current transaction
        self.curr_merchant = None
        self.curr_transaction_amount = 0
        self.curr_auth_step = 1

        # fields to collect some statistics
        self.num_successful_transactions = 0
        self.num_cancelled_transactions = 0

    def step(self):
        """ This is called in each simulation step """
        # make a transaction
        self.make_transaction()

    def make_transaction(self):
        self.curr_merchant = self.random_state.choice(self.model.merchants)
        self.curr_transaction_amount = self.random_state.uniform(0, 1, 1)[0]
        success = self.model.process_transaction(client=self, amount=self.curr_transaction_amount, merchant=self.curr_merchant)
        if success:
            self.num_successful_transactions += 1
        else:
            self.num_cancelled_transactions += 1
        self.curr_auth_step = 1

    def get_authentication(self):
        """
        Authenticate self; this can be called several times per transaction.
        Returns the authentication quality.
        :return:
        """
        self.curr_auth_step += 1
        if self.random_state.uniform(0, 1, 1)[0] < 0.5:
            # authentication quality is max
            auth_quality = 1
        else:
            # cancel the transaction
            auth_quality = None
        return auth_quality

    def _initialise_transaction_frequency(self):
        """ Transaction frequency: average purchases per day """
        # get a sample from a beta distribution with median 1/70
        freq_sample = self.random_state.beta(1.5, 7, 1)[0]
        # cut off at 1/365
        if freq_sample < self.min_transaction_frequency:
            freq_sample = self.min_transaction_frequency
        return freq_sample

    def _initialise_average_amount(self):
        # sample from left-skewed beta distribution
        # this is the average amount per transaction
        avg_amount = self.random_state.beta(10, 1, 1)[0]
        avg_amount *= self.random_state.beta(10, 1, 1)[0]
        # weigh by transaction frequency (higher frequency => smaller amount)
        avg_amount /= self.transaction_frequency
        # strech to [0, 1] interval again
        avg_amount *= self.min_transaction_frequency
        # multiply with maximal possible amount
        if avg_amount < self.min_average_amount:
            avg_amount = self.min_average_amount
        return avg_amount

    def pick_currency(self):
        """ Pick a currency, using the probabilities given in the model parameters """
        # currencies = self.model.parameters["currencies"]
        # currency_fracts = self.model.parameters["currency prob non-fraud"]
        # return self.random_state.choice(currencies, p=currency_fracts)
        return 'EUR'