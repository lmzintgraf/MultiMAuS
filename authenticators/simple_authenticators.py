from authenticators.abstract_authenticator import AbstractAuthenticator


class OracleAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, model, customer):
        if customer.fraudster:
            return False
        else:
            return True


class PositiveAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, model, customer):
        return True


class HeuristicAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, model, customer):

        authorise = False

        if customer.curr_amount > 500:
            for a in range(model.parameters['num_auth_steps']):
                auth_quality = customer.get_authentication()
                if auth_quality is None:
                    authorise = False
                    break
                # decide if good enough or else continue
                elif auth_quality > model.random_state.uniform(0, 1, 1)[0]:
                    authorise = True
                    break
        else:
            authorise = True

        return authorise


class RandomAuthenticator(AbstractAuthenticator):

    def authorise_transaction(self, model, customer):

        authorise = False

        for a in range(model.parameters['num_auth_steps']):
            # ask for authentication in 50% of the cases
            if model.random_state.uniform(0, 1, 1)[0] < 0.5:
                auth_quality = customer.get_authentication()
                if auth_quality is None:
                    authorise = False
                    break
                # decide if good enough or else continue
                elif auth_quality > model.random_state.uniform(0, 1, 1)[0]:
                    authorise = True
                    break
            else:
                authorise = True

        return authorise
