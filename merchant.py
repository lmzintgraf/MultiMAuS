from mesa import Agent


class Merchant(Agent):
    """ A merchant that sells producs to customers """
    def __init__(self, merchant_id, transaction_model):
        super().__init__(merchant_id, transaction_model)

    def step(self):