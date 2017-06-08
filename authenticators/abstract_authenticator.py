from abc import ABCMeta, abstractmethod


class AbstractAuthenticator(metaclass=ABCMeta):

    @abstractmethod
    def authorise_transaction(self, model, customer):
        """
        Decide whether to authorise transaction.
        Note that all relevant information can be obtained from the customer.
        :param customer:    the customer making a transaction
        :return:            boolean
        """


class AbstractRLAuthenticator(AbstractAuthenticator, metaclass=ABCMeta):

    @abstractmethod
    def receive_reward(self):
        """
        After each transaction, get a reward signal from the environment.
        :return: 
        """