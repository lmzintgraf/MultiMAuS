"""
This module provides an online API for the Unimaus Simulator. This can be useful in cases where
we want to have other code interacting with the simulator online, and don't necessarily need to
store the generated data in a file.

For a simple example of usage, see __main__ code at the bottom of this module.

@author Dennis Soemers (only the online API: Luisa Zintgraf developed the original simulator)
"""

from data.features.aggregate_features import AggregateFeatures
from data.features.apate_graph_features import ApateGraphFeatures
from simulator import parameters
from simulator.transactions_unimaus import UniMausTransactionModel
from simulator.customer_unimaus import GenuineCustomer, FraudulentCustomer


class OnlineUnimaus:

    def __init__(self, params=None):
        """
        Creates an object that can be used to run the simulator online / interactively. This means
        that we can have it generate a bit of data, do something with the data, generate a bit more
        data, do something again, etc. (as opposed to, generating one large batch of data, storing it
        in a file, and then using it in a different program).

        :param params:
            Parameters passed on to the UniMausTransactionModel. Will use the default parameters if None
        """
        if params is None:
            params = parameters.get_default_parameters()

        self.model = UniMausTransactionModel(params, GenuineCustomer, FraudulentCustomer)
        self.aggregate_feature_constructor = None
        self.apate_graph_feature_constructor = None

    def block_cards(self, card_ids):
        """
        Blocks the given list of Card IDs (removing all genuine and fraudulent customers with matching
        Card IDs from the simulation).

        NOTE: This function is only intended to be called using Card IDs that are 100% known to have
        been involved in fraudulent transactions. If the list contains more than a single Card ID,
        and the Card ID has not been used in any fraudulent transactions, the function may not be able
        to find the matching customer (due to an optimization in the implementation)

        :param card_ids:
            List of one or more Card IDs to block
        """
        n = len(card_ids)

        if n == 0:
            # nothing to do
            return

        if n == 1:
            # most efficient implementation in this case is simply to loop once through all customers (fraudulent
            # as well as genuine) and compare to our single blocked card ID
            blocked_card_id = card_ids[0]

            for customer in self.model.customers:
                if customer.card_id == blocked_card_id:
                    customer.stay = False

                    # should not be any more customers with same card ID, so can break
                    break

            for fraudster in self.model.fraudsters:
                if fraudster.card_id == blocked_card_id:
                    fraudster.stay = False

                    # should not be any more fraudsters with same card ID, so can break
                    break
        else:
            # with naive implementation, we'd have n loops through the entire list of customers, which may be expensive
            # instead, we loop through it once to collect only those customers with corrupted cards. Then, we follow
            # up with n loops through that much smaller list of customers with corrupted cards
            compromised_customers = [c for c in self.model.customers if c.card_corrupted]

            for blocked_card_id in card_ids:
                for customer in compromised_customers:
                    if customer.card_id == blocked_card_id:
                        customer.stay = False

                        # should not be any more customers with same card ID, so can break
                        break

                for fraudster in self.model.fraudsters:
                    if fraudster.card_id == blocked_card_id:
                        fraudster.stay = False

                        # should not be any more fraudsters with same card ID, so can break
                        break

    def clear_log(self):
        """
        Clears all transactions generated so far from memory
        """
        agent_vars = self.model.log_collector.agent_vars
        for reporter_name, records in agent_vars.items():
            records.clear()

    def get_log(self, clear_after=True):
        """
        Returns a log (in the form of a pandas dataframe) of the transactions generated so far.

        :param clear_after:
            If True, will clear the transactions from memory. This means that subsequent calls to get_log()
            will no longer include the transactions that have already been returned in a previous call.
        :return:
            The logged transactions. Returns None if no transactions were logged
        """
        log = self.model.log_collector.get_agent_vars_dataframe()

        if log is None:
            return None

        log.index = log.index.droplevel(1)

        if clear_after:
            self.clear_log()

        return log

    def step_simulator(self, num_steps=1):
        """
        Runs num_steps steps of the simulator (simulates num_steps hours of transactions)

        :param num_steps:
            The number of steps to run. 1 by default.
        :return:
            True if we successfully simulated a step, false otherwise
        """
        for step in range(num_steps):
            if self.model.terminated:
                print("WARNING: cannot step simulator because model is already terminated. ",
                      "Specify a later end_date in params to allow for a longer simulation.")
                return False

            self.model.step()
            return True

    def prepare_feature_constructors(self, data):
        """
        Prepares feature constructors (objects which can compute new features for us) using
        a given set of ''training data''. The training data passed into this function should
        NOT be re-used when training predictive models using the new features, because the new
        features will likely be unrealistically accurate on this data (and therefore models
        trained on this data would learn to rely on the new features too much)

        :param data:
            Data used to ''learn'' features
        """
        self.aggregate_feature_constructor = AggregateFeatures(data)
        self.apate_graph_feature_constructor = ApateGraphFeatures(data)

    def process_data(self, data):
        """
        Processes the given data, so that it will be ready for use in Machine Learning models. New features
        are added by the feature constructors, features which are no longer necessary afterwards are removed,
        and the Target feature is moved to the back of the dataframe

        NOTE: processing is done in-place

        :param data:
            Data to process
        :return:
            Processed dataframe
        """
        self.apate_graph_feature_constructor.add_graph_features(data)
        self.aggregate_feature_constructor.add_aggregate_features(data)

        # remove non-numeric columns / columns we don't need after adding features
        data.drop(["Global_Date", "Local_Date", "MerchantID", "Currency", "Country"], inplace=True, axis=1)

        # move Target column to the end
        data = data[[col for col in data if col != "Target" and col != "CardID"] + ["CardID", "Target"]]

        return data

    def update_feature_constructors_unlabeled(self, data):
        """
        Performs an update of existing feature constructors, treating the given new data
        as being unlabeled.

        :param data:
            (unlabeled) new data (should NOT have been passed into prepare_feature_constructors() previously)
        """
        self.aggregate_feature_constructor.update_unlabeled(data)

class DataLogWrapper:

    def __init__(self, dataframe):
        """
        Constructs a wrapper for a data log (in a dataframe). Provides some useful functions to make
        it easier to access this data from Java through jpy. This class is probably not very useful in
        pure Python.

        :param dataframe:
            The dataframe to wrap in an object
        """
        self.dataframe = dataframe

    def get_column_names(self):
        """
        Returns a list of column names

        :return:
            List of column names
        """
        return self.dataframe.columns

    def get_data_list(self):
        """
        Returns a flat list representation of the dataframe

        :return:
        """
        return [item for sublist in self.dataframe.as_matrix().tolist() for item in sublist]

    def get_num_cols(self):
        """
        Returns the number of columns in the dataframe

        :return:
            The number of columns in the dataframe
        """
        return self.dataframe.shape[1]

    def get_num_rows(self):
        """
        Returns the number of rows in the dataframe

        :return:
            The number of rows in the dataframe
        """
        return self.dataframe.shape[0]

if __name__ == '__main__':
    # construct our online simulator
    simulator = OnlineUnimaus()

    # change this value to change how often we run code inside the loop.
    # with n_steps = 1, we run code after every hour of transactions.
    # with n_steps = 2 for example, we would only run code every 2 steps
    n_steps = 1

    # if this is set to False, our simulator will not clear logged transactions after returning them from get_log.
    # This would mean that subsequent get_log calls would also return transactions that we've already seen earlier
    clear_logs_after_return = True

    # if this is set to True, we block card IDs as soon as we observe them being involved in fraudulent transactions
    # (we cheat a bit here by simply observing all true labels, this is just an example usage of API)
    block_fraudsters = True

    # keep running until we fail (which will be after 1 year due to end_date in default parameters)
    while simulator.step_simulator(n_steps):
        # get all transactions generated by the last n_steps (or all steps if clear_logs_after_return == False)
        data_log = simulator.get_log(clear_after=clear_logs_after_return)

        if data_log is not None:
            #print(data_log)

            if block_fraudsters:
                simulator.block_cards(
                    [transaction.CardID for transaction in data_log.itertuples() if transaction.Target == 1])
