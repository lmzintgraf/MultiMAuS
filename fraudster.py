from mesa import Agent


class Fraudster(Agent):
    """ A customer that can make transactions """
    def __init__(self, fraudster_id, transaction_model, random_state):
        super().__init__(fraudster_id, transaction_model)
        self.random_state = random_state

        # I am a fraudster
        self.fraudster = 1

        # decide which currency to use
        self.currency = self.pick_currency()
        self.country = 'india'

        # there's always at least one authentication step
        self.curr_auth_step = 1

        # standard currency for this customer
        self.currency = 'EUR'

        # the customer's credit card
        self.card = None

        # values of the current transaction
        self.curr_merchant = None
        self.curr_transaction_amount = 0

        # gather some statistics
        self.num_successful_transactions = 0
        self.num_cancelled_transactions = 0

    def step(self):
        """ This is called in each simulation step """
        self.make_transaction()

    def make_transaction(self):
        self.curr_merchant = self.random_state.choice(self.model.merchants)
        self.curr_transaction_amount = self.random_state.uniform(0, 1, 1)[0]
        success = self.model.process_transaction(client=self, amount=self.curr_transaction_amount, merchant=self.curr_merchant)
        if success:
            self.num_successful_transactions += 1
        else:
            self.num_cancelled_transactions += 1
        self.curr_auth_step = 0

    def get_authentication(self):
        """
        Authenticate self; this can be called several times per transaction.
        Returns the authentication quality.
        :return:
        """
        self.curr_auth_step += 1
        if self.curr_auth_step == 1:
            auth_quality = self.random_state.uniform(0, 1, 1)[0]
        else:
            # cancel the transaction
            auth_quality = None
        return auth_quality

    def pick_currency(self):
        """ Pick a currency, using the probabilities given in the model parameters """
        # currencies = self.model.parameters["currencies"]
        # currency_fracts = self.model.parameters["currency prob fraud"]
        # return self.random_state.choice(currencies, p=currency_fracts)
        return 'USD'