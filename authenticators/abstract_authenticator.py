from abc import ABCMeta, abstractmethod


class AbstractAuthenticator(metaclass=ABCMeta):
    def __init__(self):
        """
        Every authenticator has to have a name
        :param name: 
        """
        super().__init__()

    @abstractmethod
    def authorise_transaction(self, customer):
        """
        Decide whether to authorise transaction.
        Note that all relevant information can be obtained from the customer.
        :param customer:    the customer making a transaction
        :return:            boolean, whether or not to authorise the transaction
        """
