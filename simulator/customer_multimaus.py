from simulator.unimaus_customer import UniMausCustomer


class MultiMausCustomer(UniMausCustomer):
    """ 
    A customer that supports multi-modal authentication. 
    """
    def __init__(self, customer_id, transaction_model):
        super().__init__(customer_id, transaction_model, fraudster=0)

        # fields to collect some statistics
        self.num_successful_transactions = 0
        self.num_cancelled_transactions = 0

        # current authentication step
        self.curr_auth_step = 1

        # how satisfied the customer is with the service in general
        self.satisfaction = 1.

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

    def make_transaction(self):
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

    def stay_customer(self):
        stay_prob = self.model.parameters['stay_prob'][self.fraudster]
        if stay_prob < self.model.random_state.uniform(0, 1, 1)[0]:
            return True
        else:
            return False
