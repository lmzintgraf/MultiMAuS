from simulator.merchant import Merchant
from simulator.customer_unimaus import Customer, Fraudster
from simulator.transactions_abstract import AbstractTransactionModel


class UniMausTransactionModel(AbstractTransactionModel):
    """A model with some number of agents."""
    def __init__(self, model_parameters):
        super().__init__(model_parameters)

        # count the total number of customers (for the new unique ids)
        self.next_customer_id = len(self.customers)
        self.next_fraudster_id = len(self.fraudsters)

    def customer_migration(self):

        num_customers_before = len(self.customers)
        num_fraudsters_before = len(self.fraudsters)

        self.emigration_customers()
        self.emigration_fraudsters()

        self.immigration_customers(num_customers_before)
        self.immigration_fraudsters(num_fraudsters_before)

    def emigration_customers(self):
        self.customers = [c for c in self.customers if c.stay]

    def emigration_fraudsters(self):
        self.fraudsters = [f for f in self.fraudsters if f.stay]

    def immigration_customers(self, customer_count_before):
        """
        We will just add (about) as many customers as left.
        The number of customers is just the ones we have in our pool,
        not the actual number of customers making transactions.
        All it influences is how often customers make transactions again
        :param customer_count_before: 
        :return: 
        """
        # estimate number of customers we add
        num_new_cust = customer_count_before - len(self.customers)

        # add new customers and increase ID count
        new_cust_ids = range(self.next_customer_id, self.next_customer_id + num_new_cust)
        self.customers.extend([Customer(cid, self) for cid in new_cust_ids])
        self.next_customer_id += num_new_cust

    def immigration_fraudsters(self, fraudster_count_before):
        # estimate number of fraudsters we add
        num_new_frauds = fraudster_count_before - len(self.fraudsters)

        # add new customers and increase ID count
        new_frauds_ids = range(self.next_fraudster_id, self.next_fraudster_id + num_new_frauds)
        self.fraudsters.extend([Fraudster(cid, self) for cid in new_frauds_ids])
        self.next_fraudster_id += num_new_frauds

    def initialise_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num merchants"])]

    def initialise_customers(self):
        return [Customer(i, self) for i in range(self.parameters['num_customers'])]

    def initialise_fraudsters(self):
        return [Fraudster(i, self) for i in range(self.parameters["num_fraudsters"])]
