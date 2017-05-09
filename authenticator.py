from mesa import Agent


class Authenticater(Agent):
    """A payment processing platform"""
    def __init__(self, transaction_model, random_state):
        super().__init__(0, transaction_model)

    def process_transaction(self, transaction):
        pass