from mesa.mesa import Agent
from abc import ABCMeta, abstractmethod
import numpy as np
import copy


class AbstractCustomer(Agent,  metaclass=ABCMeta):
    def __init__(self, unique_id, transaction_model, fraudster):
        """
        Abstract class for customers, which can either be genuine or fraudulent.
        :param unique_id:           the (unique) customer ID
        :param transaction_model:   the transaction model that is used, instance of mesa.Model
        :param fraudster:           boolean whether customer is genuine or fraudulent
        """
        # call super init from mesa agent
        super().__init__(unique_id, transaction_model)

        # copy parameters from model
        self.params = self.model.parameters

        # internal random state (different for every customer0
        self.random_state = np.random.RandomState(self.model.random_state.randint(0, np.iinfo(np.int32).max))

        # each customer has to say if it's a fraudster or not
        self.fraudster = int(fraudster)

        # pick country, currency, card
        self.country = self.initialise_country()
        self.currency = self.initialise_currency()
        self.card_id = None  # picked with first transaction

        # variable for whether a transaction is currently being processed
        self.active = False

        # fields for storing the current transaction properties
        self.curr_merchant = None
        self.curr_amount = None
        self.local_datetime = None
        self.curr_auth_step = 0
        self.curr_trans_cancelled = False
        self.curr_trans_success = False

        # variable tells us whether the customer wants to stay after current transaction
        self.stay = True

    def step(self):
        """ 
        This is called in each simulation step (i.e., one hour).
        Each individual customer/fraudster decides whether to make a transaction or  not.
        """

        # decide whether to make transaction or not
        make_transaction = self.decide_making_transaction()

        if make_transaction:

            # if this is the first transaction, we assign a card ID
            if self.card_id is None:
                self.card_id = self.initialise_card_id()

            # set the agent to active
            self.active = True

            # pick a current merchant
            self.curr_merchant = self.get_curr_merchant()

            # pick a current amount
            self.curr_amount = self.get_curr_amount()

            # process current transaction
            self.curr_trans_success = self.model.process_transaction(self)

            # if necessary post-process the transaction
            self.post_process_transaction()

        else:

            # set to inactive (important for our transaction logs)
            self.active = False
            self.curr_merchant = None
            self.curr_amount = None
            self.local_datetime = None

    def request_transaction(self):
        self.model.authorise_transaction(self)

    def post_process_transaction(self):
        """
        Optional updates after transaction;
        e.g. decide whether to stay or update satisfaction
        :return: 
        """
        pass

    @abstractmethod
    def give_authentication(self):
        """
        Authenticate self if requested by the payment processing platform.
        Return can e.g. be quality of authentication or boolean.
        If no authentication is given, this usually returns None.
        :return:
        """
        pass

    @abstractmethod
    def get_curr_merchant(self):
        pass

    @abstractmethod
    def get_curr_amount(self):
        pass

    @abstractmethod
    def decide_making_transaction(self):
        """
        Decide whether to make transaction or not, given the current time step
        :return:    Boolean indicating whether to make transaction or not
        """
        pass

    @abstractmethod
    def initialise_country(self):
        """
        Select country where customer's card was issued
        :return:    country
        """
        pass

    @abstractmethod
    def initialise_currency(self):
        """
        Select currency in which customer makes transactions
        :return:    string
        """
        pass

    @abstractmethod
    def initialise_card_id(self):
        """ 
        Select creditcard number (unique ID) for customer
        :return:    credit card number
        """
        pass
