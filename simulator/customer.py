from simulator.base_customer import BaseCustomer


class Customer(BaseCustomer):
    """ A customer that can make transactions """
    def __init__(self, customer_id, transaction_model):
        super().__init__(customer_id, transaction_model, fraudster=0)

    def get_authentication(self):
        """
        Authenticate self; this can be called several times per transaction.
        Returns the authentication quality.
        :return:
        """
        self.curr_auth_step += 1
        if self.model.random_state.uniform(0, 1, 1)[0] < 0.5:
            # authentication quality is max
            auth_quality = 1
        else:
            # cancel the transaction
            auth_quality = None
        return auth_quality
