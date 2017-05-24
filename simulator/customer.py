from simulator.abstract_customer import AbstractCustomer


class Customer(AbstractCustomer):
    """ A customer that can make transactions """
    def __init__(self, customer_id, transaction_model):
        super().__init__(customer_id, transaction_model)

        # I am not a fraudster
        self.fraudster = 0

        # transaction frequency (avg. purchases per day)
        self.min_transaction_frequency = 1./365
        self.transaction_frequency = self._initialise_transaction_frequency()

        # the average amount the user spends
        self.min_average_amount = 0.0005  # equals ten cents
        self.average_amount = self._initialise_average_amount()

        self.curr_auth_step = 1

        # fields to collect some statistics
        self.num_successful_transactions = 0
        self.num_cancelled_transactions = 0

    def start_transaction(self):
        """
        Make a transaction.
        :return: 
        """
        # randomly pick a merchant
        self.curr_merchant = self.model.random_state.choice(self.model.merchants)

        # randomly pick a transaction amount
        self.curr_transaction_amount = self.model.random_state.uniform(0, 1, 1)[0]

        # make the transaction
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
        if self.model.random_state.uniform(0, 1, 1)[0] < 0.5:
            # authentication quality is max
            auth_quality = 1
        else:
            # cancel the transaction
            auth_quality = None
        return auth_quality

    def _initialise_transaction_frequency(self):
        """ Transaction frequency: average purchases per day """
        # get a sample from a beta distribution with median 1/70
        freq_sample = self.model.random_state.beta(1.5, 7, 1)[0]
        # cut off at 1/365
        if freq_sample < self.min_transaction_frequency:
            freq_sample = self.min_transaction_frequency
        return freq_sample

    def _initialise_average_amount(self):
        # sample from left-skewed beta distribution
        # this is the average amount per transaction
        avg_amount = self.model.random_state.beta(10, 1, 1)[0]
        avg_amount *= self.model.random_state.beta(10, 1, 1)[0]
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

    def pick_country(self):
        return 'India'

    def pick_creditcard_number(self):
        return self.unique_id
