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

        # estimate number of customers we add
        num_new_cust = customer_count_before - len(self.customers)
        # TODO: add noise

        # add new customers and increase ID count
        new_cust_ids = range(self.next_customer_id, self.next_customer_id + num_new_cust)
        self.customers.extend([Customer(cid, self) for cid in new_cust_ids])
        self.next_customer_id += num_new_cust

        # # the number of customers making a transaction at each step
        # avg_num_cust = int(parameters.aggregated_data.loc['transactions']['non-fraud']) / 366. / 24.
        # # weigh by the current month
        # avg_num_cust *= 12 * self.parameters['frac_month'][self.curr_datetime.month - 1, 0]
        # # average number of customers leaving at each step
        # avg_cust_leave = avg_num_cust * (1 - self.parameters['stay_prob'][0])
        #
        # if avg_cust_leave >= 1:
        #     avg_cust_leave += self.parameters['noise_level'] * self.random_state.normal(0, 1, 1)[0]
        #     num_new_cust = int(np.round(avg_cust_leave, 0))
        #     new_cust_ids = range(self.customer_count, self.customer_count + num_new_cust)
        #     self.customers.extend([Customer(cid, self) for cid in new_cust_ids])
        #     self.customer_count += num_new_cust
        # else:
        #     if avg_cust_leave > self.random_state.uniform(0, 1, 1)[0]:
        #         self.customers.append(Customer(self.customer_count, self))
        #         self.customer_count += 1

    def immigration_fraudsters(self, fraudster_count_before):

        # estimate number of fraudsters we add
        num_new_frauds = fraudster_count_before - len(self.fraudsters)
        # TODO: add noise

        # add new customers and increase ID count
        new_frauds_ids = range(self.next_fraudster_id, self.next_fraudster_id + num_new_frauds)
        self.fraudsters.extend([Fraudster(cid, self) for cid in new_frauds_ids])
        self.next_fraudster_id += num_new_frauds

        # # the number of customers making a transaction at each step
        # avg_num_fraudst = int(parameters.aggregated_data.loc['transactions']['fraud']) / 366. / 24.
        # # weigh by the current motnh
        # avg_num_fraudst *= 12 * self.parameters['frac_month'][self.curr_datetime.month - 1, 1]
        # # average number of customers leaving at each step
        # avg_fraudst_leave = avg_num_fraudst * (1 - self.parameters['stay_prob'][1])
        #
        # if avg_fraudst_leave >= 1:
        #     avg_fraudst_leave += self.parameters['noise_level'] * self.random_state.normal(0, 1, 1)[0]
        #     num_new_fraudst = int(np.round(avg_fraudst_leave, 0))
        #     new_fraudster_ids = range(self.fraudster_count, self.fraudster_count + num_new_fraudst)
        #     self.fraudsters.extend([Customer(fid, self) for fid in new_fraudster_ids])
        #     self.fraudster_count += num_new_fraudst
        # else:
        #     avg_fraudst_leave += self.parameters['noise_level'] * self.random_state.normal(0, 0.1, 1)[0]
        #     if avg_fraudst_leave > self.random_state.uniform(0, 1, 1)[0]:
        #         self.fraudsters.append(Fraudster(self.fraudster_count, self))
        #         self.fraudster_count += 1

    def initialise_merchants(self):
        return [Merchant(i, self) for i in range(self.parameters["num merchants"])]

    def initialise_customers(self):
        return [Customer(i, self) for i in range(self.parameters['num_customers'])]

    def initialise_fraudsters(self):
        return [Fraudster(i, self) for i in range(self.parameters["num_fraudsters"])]
