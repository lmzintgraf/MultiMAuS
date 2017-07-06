from authenticators.abstract_authenticator import AbstractAuthenticator


class OracleAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, customer):
        if customer.fraudster:
            return False
        else:
            return True


class NeverSecondAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, customer):
        return True


class AlwaysSecondAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, customer):
        if customer.get_authentication() is not None:
            return True
        else:
            return False


class HeuristicAuthenticator(AbstractAuthenticator):
    def __init__(self, thresh):g
        self.thresh = thresh

    def authorise_transaction(self, customer):

        authorise = True
        if customer.curr_amount > self.thresh:
            auth_quality = customer.get_authentication()
            if auth_quality is None:
                authorise = False

        return authorise


class RandomAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, customer):

        # ask for second authentication in 50% of the cases
        if customer.model.random_state.uniform(0, 1, 1)[0] < 0.5:
            auth_quality = customer.give_authentication()
            if auth_quality is None:
                authorise = False
            else:
                authorise = True
        else:
            authorise = True

        return authorise
