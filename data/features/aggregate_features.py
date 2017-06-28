"""
This file provides a class that can be ''trained'' on training data, and then produce new
aggregate features for both training and test data. ''Training'' this class simply consists
of it memorizing various statistics.

Features constructed by this class are primarily inspired by:

"A data mining based system for credit-card fraud detection in e-tail", by
Nuno Carneiro, Gonçalo Figueira, Miguel Costa [1]

"Feature engineering strategies for credit card fraud detection", by
Alejandro Correa Bahnsen, Djamila Aouada, Aleksandar Stojanovic, Björn Ottersten [2]

Implementation is partially based on https://github.com/kb211/Fraud_Detection2017 [3]

@author Dennis Soemers
"""

from datetime import datetime
from datetime import timedelta
import pandas as pd

class AggregateFeatures:

    def __init__(self, training_data):
        """
        Constructs an object based on the given training data. This will cause it to memorize
        various statistics from the training data. The object can subsequently be used to generate
        new features for any dataset (can also add features to the same dataset if desired)

        :param training_data:
        """
        print(str(datetime.now()), ": Init Aggregate Features...")

        self.country_all_dict, self.country_fraud_dict = self.compute_fraud_ratio_dicts(training_data, "Country")
        print(str(datetime.now()), ": Finished computing country dicts...")
        self.first_order_times_dict = self.compute_first_order_times_dict(training_data)
        print(str(datetime.now()), ": Finished computing First Order Times dict...")

        # unfortunately we kinda actually have to store the entire training dataset for some of the aggregate
        # features as described in [2]
        self.training_data = training_data

        # compute and store a mapping from card IDs to lists of transactions
        # this is a bit expensive memory-wise, but will very significantly speed up feature construction
        self.transactions_by_card_ids = {}
        self.add_transactions_by_card_ids(training_data)
        print(str(datetime.now()), ": Finished computing Transactions by Card IDs dict from training data...")

        # TODO
        '''
        It may be useful to add support for updating aggregate statistics with individual new transactions after
        first ''training'' it on a large training set. This could be used to update aggregate statistics in real-
        time using transactions for which we obtain labels in real-time (transactions which are investigated
        by human experts for example)
        '''

    def add_aggregate_features(self, data, include_test_data_in_history=True):
        """
        Adds all aggregate features to the given dataset.

        Currently supported features:
            - CountryFraudRatio: The ratio of transactions with same country that were fraudulent in
            training data.
            - CountrySufficientSampleSize: Binary feature, 1 if and only if we have observed a sufficiently
            large sample size of transactions from the same country (>= 30)
            - TimeSinceFirstOrder: The time (in hours) since the first transaction was observed with the same
            card ID.


        :param data:
            Data to augment with aggregate features
        :param include_test_data_in_history:
            If true, the given data itself will also be treated as potential ''historic'' data (early
            rows in the data can be used as history for rows with same Card IDs later in the data). If
            false, only the training data passed into constructor will be used as history.
        :return:
            Augmented version of the dataset (features added in-place, so no need to capture return value)
        """

        '''
        The two features below are inspired by [1]. The paper describes clustering countries in four groups
        based on fraud ratio, and assigning countries with a small sample size to an ''intermediate level of
        risk'' group regardless of the actual ratio within that small sample size. No further information is
        provided in the paper about exactly how the clustering is done.

        Instead, we'll simply use a numeric feature for the ratio, and a binary feature indicating whether or
        not we consider the sample size to be sufficient. Completely linear Machine Learning models (such as
        pure Logistic Regression) may struggle to combine these two features in an intelligent manner, but
        more hierarchical models (like Neural Networks or Decision Trees) might be able to combine them a bit better.
        (based on my intuition at least, no fancy citations for this :( )
        '''
        data["CountryFraudRatio"] = data.apply(
            lambda row: self.get_country_fraud_ratio(row=row), axis=1)
        data["CountrySufficientSampleSize"] = data.apply(
            lambda row: self.is_country_sample_size_sufficient(row=row), axis=1)
        print(str(datetime.now()), ": Finished adding country-related features...")

        '''
        The following feature appears in Table 1 in [1], but has no explanation otherwise in the paper. Intuitively,
        I suppose it can be an indication of how trustworthy a Card is, in that one that has been in use for
        a very long time may be more trustworthy than a brand new card.
        '''
        data["TimeSinceFirstOrder"] = data.apply(
            lambda row: self.get_time_since_first_order(row=row), axis=1)
        print(str(datetime.now()), ": Finished adding Time Since First Order feature...")

        data = self.add_historical_features(data, include_test_data_in_history)
        print(str(datetime.now()), ": Finished adding historical features")

        return data

    def add_historical_features(self, data, include_test_data_in_history=True,
                                time_frames=[1, 3, 6, 12, 18, 24, 72, 168],
                                conditions=((), ('MerchantID',), ("Country",))):
        """
        Adds multiple historical features to the given dataset. Explanation:

        For every row in data:
            For every time-frame specified in time_frames:
                For every tuple of column names specified in conditions:
                    We collect all historical transactions in training data (and also the given new dataset itself
                    if include_test_data_in_history=True) that still fit within the timeframe (in hours), have
                    the same Card ID as row, and have an equal value for all column names. Based on this set
                    of recent, related transactions (related through Card ID and optionally additional conditions),
                    we construct two new features for the row:
                        1) the number of transactions in this set
                        2) the sum of transactions amounts in this set

        The total number of features added to every row by this function is
            2 * |time_frames| * |conditions|

        This is all based on Section 3.1 of [2]

        :param data:
            Dataset to augment with extra features
        :param include_test_data_in_history:
            If True, we also use early rows in the given data as potential ''historical data'' for subsequent rows
        :param time_frames:
            List of all the time-frames for which we want to compute features. Default selection of time-frames
            based on [2].
        :param conditions:
            A tuple of tuples of column names. Every tuple represents a condition. Historical transactions are only
            included in the set that features are computed from if they satisfy the condition. A condition is satisfied
            if and only if a transaction has the same values for all column names as the transaction we're computing
            features for. By default, we use an empty tuple (= compute features with no extra conditions other than
            Card ID and time-frame), ("MerchantID") (= compute features only from transactions with the same Merchant
            ID), and ("Country") (= compute features only from transactions with the same Country).

            Note that it's also possible to specify tuples with more than a single column name, to create even
            more specific conditions where multiple columns must match.
        :return:
            The dataset, augmented with new features (features added in-place)
        """

        # make sure time-frames are sorted
        time_frames = sorted(time_frames)

        if include_test_data_in_history:
            # add all new transactions to our mapping from Card IDs to transactions
            self.add_transactions_by_card_ids(data)
            print(str(datetime.now()), ": Finished computing Transactions by Card IDs dict from test data...")

        # add our new columns, with all 0s by default
        for feature_type in ("Num", "Amt_Sum"):
            for time_frame in time_frames:
                for cond in conditions:
                    new_col_name = feature_type + "_" + str(time_frame)

                    for cond_part in cond:
                        new_col_name += "_" + cond_part

                    if feature_type == "Num":
                        data[new_col_name] = 0
                    else:
                        data[new_col_name] = 0.0

        print(str(datetime.now()), ": Added all-zero columns")

        # now we have all the columns ready, and we can loop through rows, handling all features per row at once
        for row in data.itertuples():
            # date of the row we're adding features for
            row_date = row.Global_Date

            # the Card ID of the row we're adding features for
            row_card_id = row.CardID

            # select all training data with correct Card ID, and with a date earlier than row
            card_transactions = self.transactions_by_card_ids[row_card_id]
            matching_data = self.extract_transactions_before(card_transactions, row_date)

            if matching_data is None:
                continue

            if include_test_data_in_history:
                # also need to include parts of test data with earlier dates than the row we're adding features to
                matching_test_data = self.extract_transactions_before(data, row_date)

                if matching_test_data is not None:
                    matching_test_data = matching_test_data.loc[matching_test_data["CardID"] == row_card_id]
                    matching_data = pd.concat([matching_data, matching_test_data], axis=0, ignore_index=True)

            # loop over our time-frames in reverse order, so that we can gradually cut out more and more data
            for time_frame_idx in range(len(time_frames) - 1, -1, -1):
                time_frame = time_frames[time_frame_idx]

                # reduce matching data to part that fits within this time frame
                earliest_allowed_date = row_date - timedelta(hours=time_frame)
                matching_data = self.extract_transactions_after(matching_data, earliest_allowed_date)

                if matching_data is None:
                    break

                # loop through our conditions
                for condition in conditions:
                    conditional_matching_data = matching_data

                    col_name_num = "Num_" + str(time_frame)
                    col_name_amt = "Amt_Sum_" + str(time_frame)

                    # loop through individual parts of the condition
                    for condition_term in condition:
                        row_condition_value = getattr(row, condition_term)
                        conditional_matching_data = conditional_matching_data.loc[
                            conditional_matching_data[condition_term] == row_condition_value]

                        col_name_num += "_" + condition_term
                        col_name_amt += "_" + condition_term

                    # now the conditional_matching_data is all we want for two new features
                    data.set_value(row.Index, col_name_num, conditional_matching_data.shape[0])
                    data.set_value(row.Index, col_name_amt, conditional_matching_data["Amount"].sum())

        return data

    def add_transactions_by_card_ids(self, data):
        """
        Computes a dictionary, mapping from Card IDs to dataframes. For every unique card ID in the data,
        we store a small dataframe of all transactions with that Card ID.

        :param training_data:
            Labelled training data
        """
        for card_id in data.CardID.unique():
            if card_id not in self.transactions_by_card_ids:
                # card ID not in map yet
                self.transactions_by_card_ids[card_id] = data.loc[data["CardID"] == card_id]
            else:
                # card ID already in map, so should append
                self.transactions_by_card_ids[card_id] = self.transactions_by_card_ids[card_id]\
                    .append(data.loc[data["CardID"] == card_id], ignore_index=True)

    def compute_first_order_times_dict(self, training_data):
        """
        Computes a dictionary, mapping from Card IDs to timestamps (dates). For every unique card ID
        in the training data, we store the first point in time where that card was used for a transaction.

        :param training_data:
            Labelled training data
        :return:
            Dictionary, with card IDs as keys, and dates of first transactions as values
        """
        first_order_times_dict = {}
        for row in training_data.itertuples():
            card = row.CardID

            if card not in first_order_times_dict:
                first_order_times_dict[card] = row.Global_Date

        return first_order_times_dict

    def compute_fraud_ratio_dicts(self, training_data, column):
        """
        Computes two dictionaries, with all values of a given column as keys. The given column
        should correspond to a discrete feature, otherwise this is going to return fairly large
        and fairly useless dictionaries. One dictionary will contain, for every feature value, the
        total number of transactions, and the other will contain the number of fraudulent transactions.

        :param training_data:
            Labelled training data
        :param column:
            Column to compute dictionary for
        :return:
            Dictionary with counts of all transactions, and dictionary with counts of fraudulent transactions
        """
        all_transactions_dict = {}
        fraud_transactions_dict = {}

        # Thanks Kasper for implementation :D [3]
        fraud_list = training_data.loc[training_data["Target"] == 1]
        fraud_dict = fraud_list[column].value_counts()
        all_dict = training_data[column].value_counts()
        for key, item in all_dict.iteritems():
            all_transactions_dict[key] = all_dict[key]

            if key in fraud_dict:
                fraud_transactions_dict[key] = fraud_dict[key]
            else:
                fraud_transactions_dict[key] = 0

        return all_transactions_dict, fraud_transactions_dict

    def extract_transactions_before(self, data, date, hint=-1):
        """
        Helper function which extracts all transactions from the given data which took place before
        the given point in time. It assumes that the data is sorted by date (this assumption allows
        for a much more efficient implementation)

        :param data:
            Data to extract transactions from
        :param date:
            We'll extract transactions that took place before this point in time
        :param hint:
            If >= 0, we'll inspect the date of the transaction at this index first. Can be used to
            speed up the binary search. For example, if data = training data, and date = the date of
            a transaction from later test data, we can set hint to (the size of the training data - 1)
            in order to instantly see that the entire training data occurred before the given date
        :return:
            The extracted transactions
        """
        #print("")
        #print("Want all transactions before ", str(date))

        # we'll use binary search to find where a transaction at the given date should be inserted; then
        # we can simply return all transactions up to that index
        low = 0
        high = data.shape[0] - 1

        if hint >= 0:
            # we were given a hint, should investigate there first
            hint_date = data.iloc[hint].Global_Date

            if hint_date < date:
                low = hint + 1

        while low <= high:
            mid = (low + high) // 2
            mid_date = data.iloc[mid].Global_Date

            #print("Time at ", mid, " = ", str(mid_date))

            if mid_date >= date:
                high = mid - 1
            else:
                low = mid + 1

        # ''low'' is now the leftmost index where we could insert the transaction at the given date without
        # messing up the ordering.
        if low > 0:
            '''
            if data.iloc[low].Date < date:
                print("extract_transactions_before ERROR: should also have included ", low)

            if data.iloc[low - 1].Date >= date:
                print("extract_transactions_before ERROR: should not have included ", low)
            '''

            # return all data up to the low index (excluding low itself)
            #print("Returning everything up to ", low)
            return data.iloc[:low]
        else:
            # no data, so just return None
            #print("Returning None")
            return None

    def extract_transactions_after(self, data, date):
        """
        Helper function which extracts all transactions from the given data which took place
        after (or exactly at) the given point in time. It assumes that the data is sorted by
        date (this assumption allows for a much more efficient implementation)

        :param data:
            Data to extract transactions from
        :param date:
            We'll extract transactions that took place after or at this point in time
        :return:
            The extracted transactions
        """
        # we'll use binary search to find where a transaction at the given date should be inserted; then
        # we can simply return all transactions starting from that index
        low = 0
        high = data.shape[0] - 1

        while low <= high:
            mid = (low + high) // 2
            mid_date = data.iloc[mid].Global_Date

            if mid_date >= date:
                high = mid - 1
            else:
                low = mid + 1

        # ''low'' is now the leftmost index where we could insert the transaction at the given date without
        # messing up the ordering.
        if low < data.shape[0]:
            '''
            if data.iloc[low].Date < date:
                print("extract_transactions_after ERROR: should not have included ", low)

            if low > 0 and data.iloc[low - 1].Date >= date:
                print("extract_transactions_after ERROR: should also have included ", low - 1)
            '''

            # return all data starting from low
            return data.iloc[low:]
        else:
            # no data, so just return None
            return None

    def get_country_fraud_ratio(self, country="", row=None):
        """
        Computes the ratio of fraudulent transactions for a country

        :param country:
            Country (string) to get the fraud ratio for
        :param row:
            If not None, Country will be extracted from this row
        :return:
            Ratio of transactions corresponding to given country which are fraudulent
        """
        if row is not None:
            country = row["Country"]

        if country not in self.country_all_dict:
            # TODO may be interesting to try average of all countries? Or max, to motivate exploration?
            return 0.0
        else:
            return float(self.country_fraud_dict[country]) / float(self.country_all_dict[country])

    def get_time_since_first_order(self, row):
        """
        Computes the time since the first order (= transaction) with the same Card ID

        :param row:
            Data row representing a new transaction
        :return:
            Time (in hours) since first order with the same card (or 0 if never seen before)
        """
        cardID = row["CardID"]
        date = row["Global_Date"]

        if cardID in self.first_order_times_dict:
            time_delta = date - self.first_order_times_dict[cardID]
            return max(0, float(time_delta.days * 24) + (float(time_delta.seconds) / (60 * 60)))

        # first time we see this card, so simply return 0
        return 0

    def is_country_sample_size_sufficient(self, country="", row=None):
        """
        Returns 1 if and only if the number of observations for a given country >= 30
        (returns 0 otherwise)

        :param country:
            Country (string) to check the sample size for
        :param row:
            If not None, Country will be extracted from this row
        :return:
            1 if and only if the number of observations >= 30, 0 otherwise
        """
        if row is not None:
            country = row["Country"]

        if country not in self.country_all_dict:
            return 0
        else:
            if self.country_all_dict[country] >= 30:
                return 1
            else:
                return 0
