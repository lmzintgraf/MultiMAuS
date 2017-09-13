from authenticators.abstract_authenticator import AbstractAuthenticator


class OracleAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):
        if customer.fraudster:
            customer.give_authentication()


class NeverSecondAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):
        pass


class AlwaysSecondAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):
        customer.give_authentication()


class HeuristicAuthenticator(AbstractAuthenticator):
    def __init__(self, thresh=50):
        super().__init__()
        self.thresh = thresh

    def authorise_transaction(self, customer):
        if customer.curr_amount > self.thresh:
            customer.give_authentication()


class RandomAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):

        # ask for second authentication in 50% of the cases
        if customer.model.random_state.uniform(0, 1, 1)[0] < 0.5:
            customer.give_authentication()
