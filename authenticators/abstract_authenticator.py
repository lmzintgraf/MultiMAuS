from abc import ABCMeta, abstractmethod


class AbstractAuthenticator(metaclass=ABCMeta):
    def __init__(self, name):
        """
        Every authenticator has to have a name
        :param name: 
        """
        super().__init__()
        self.name = name

    @abstractmethod
    def authorise_transaction(self, customer):
        """
        Decide whether to authorise transaction.
        Note that all relevant information can be obtained from the customer.
        :param customer:    the customer making a transaction
        :return:            boolean, whether or not to authorise the transaction
        """


class AbstractRLAuthenticator(AbstractAuthenticator, metaclass=ABCMeta):

    @abstractmethod
    def receive_reward(self):
        """
        After each transaction, get a reward signal from the environment.
        :return: 
        """