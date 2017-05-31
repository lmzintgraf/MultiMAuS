from mesa import Agent


class Merchant(Agent):
    """
    A merchant that sells products to customers.
    At the moment, merchants are just passive.
    """
    def __init__(self, merchant_id, transaction_model):
        super().__init__(merchant_id, transaction_model)

    def get_amount(self):
        return self.model.random_state.uniform(0, 1, 1)[0]