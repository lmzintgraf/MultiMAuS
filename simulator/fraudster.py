from simulator.abstract_customer import AbstractCustomer


class Fraudster(AbstractCustomer):
    """ A customer that can make transactions """
    def __init__(self, fraudster_id, transaction_model):
        super().__init__(fraudster_id, transaction_model, fraudster=1)

        # Fraudster's country (i.e., where card was issued)
        self.country = self.pick_country()

        # there's always at least one authentication step
        self.curr_auth_step = 1

        # fields for some statistics
        self.num_successful_transactions = 0
        self.num_cancelled_transactions = 0

    def start_transaction(self):
        """
        Make a fraudulent transaction.
        :return: 
        """
        # randomly pick a merchant
        self.curr_merchant = self.model.random_state.choice(self.model.merchants)

        # randomly pick a transaction amount
        self.curr_transaction_amount = self.model.random_state.uniform(0, 1, 1)[0]

        # do the transaction
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
            auth_quality = self.model.random_state.uniform(0, 1, 1)[0]
        else:
            # cancel the transaction
            auth_quality = None
        return auth_quality

    def pick_creditcard_number(self):
        return self.unique_id
