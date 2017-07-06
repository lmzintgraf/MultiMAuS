from simulator.customer_unimaus import UniMausGenuineCustomer, UniMausFraudulentCustomer


class MultiMausGenuineCustomer(UniMausGenuineCustomer):
    def __init__(self, transaction_model, satisfaction=None):

        # call super init
        super().__init__(transaction_model)

        # field whether current transaction was authorised or not
        self.curr_trans_authorised = False

        # initialise the customer's patience
        self.patience = self.random_state.beta(10, 2, 1)[0]

        self.satisfaction = satisfaction
        if self.satisfaction is None:
            # how satisfied the customer is with the service in general
            self.satisfaction = 0.5 * (1 + self.model.get_social_satisfaction())

    def perform_transaction(self):

        # the super method will pick a merchant and amount
        super().perform_transaction()

        # request to make the transaction
        self.curr_trans_authorised = self.model.authenticator.authorise_transaction(customer=self)

    def card_got_corrupted(self):
        self.card_corrupted = True
        self.satisfaction *= 0.9

    def stay_after_transaction(self):
        stay_prob = self.satisfaction * self.params['stay_prob'][self.fraudster]
        leave = (1-stay_prob) > self.random_state.uniform(0, 1, 1)[0]
        if leave:
            self.stay = False

    def get_transaction_prob(self):
        return self.satisfaction * super().get_curr_transaction_prob()

    def make_transaction(self):
        super().make_transaction()
        self.update_satisfaction()

    def update_satisfaction(self):
        """
        Adjust the satisfaction of the user after a transaction was made.
        :return: 
        """
        # if no authentication was done, the satisfaction goes up by 1%
        if self.curr_auth_step == 0:
            self.satisfaction *= 1.001
        else:
            # if a second authentication was done, the satisfaction goes down by 1%
            if self.curr_trans_authorised:
                self.satisfaction *= 0.995
            # if second authentication as asked but the customer cancelled the transaction, the satisfaction goes down by 10%
            else:
                self.satisfaction *= 0.9

    def get_authentication(self):
        """
        Authenticate self; this can be called several times per transaction.
        Returns the authentication quality.
        :return:
        """
        curr_patience = 0.5 * (self.patience + self.curr_amount/self.curr_merchant.max_amount)
        if curr_patience > self.random_state.uniform(0, 1, 1)[0]:
            auth_quality = 1
        else:
            # cancel the transaction
            auth_quality = None
        return auth_quality


class MultiMausFraudulentCustomer(UniMausFraudulentCustomer):

    def get_authentication(self):
        """
        Authenticate self; this can be called several times per transaction.
        Returns the authentication quality.
        :return:
        """
        # we assume that the fraudster cannot provide a second authentication
        return None
