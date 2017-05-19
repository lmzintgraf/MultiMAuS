from mesa import Agent


class Authenticator(Agent):
    def __init__(self, transaction_model, random_state, max_authentication_steps):
        super().__init__(0, transaction_model)
        self.random_state = random_state
        self.max_authentication_steps = max_authentication_steps

    def authorise_payment(self, customer, amount, merchant):

        authorise = False

        if self.random_state.uniform(0, 1, 1)[0] < 0.5:
            second_auth_quality = customer.get_authentication()
            if second_auth_quality is None:
                authorise = False
            elif second_auth_quality > 0.5:
                authorise = True
        else:
            authorise = True

        return authorise
