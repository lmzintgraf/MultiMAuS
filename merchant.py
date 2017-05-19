from mesa import Agent


class Merchant(Agent):
    """
    A merchant that sells products to customers.
    At the moment, merchants are just passive.
    """
    def __init__(self, merchant_id, transaction_model):
        super().__init__(merchant_id, transaction_model)


# TO DO
# - for each merchant, plot the average amount
# - for each client with more than X transactions, plot the average amount
# - plot country distribution