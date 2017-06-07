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


class Aggregate_Features:

    def __init__(self, training_data):
        """
        Constructs an object based on the given training data. This will cause it to memorize
        various statistics from the training data. The object can subsequently be used to generate
        new features for any dataset (can also add features to the same dataset if desired)

        :param training_data:
        """
        self.country_all_dict, self.country_fraud_dict = self.compute_fraud_ratio_dicts(training_data, "Country")
        self.first_order_times_dict = self.compute_first_order_times_dict(training_data)

        # TODO
        '''
        A bunch of interesting ideas in the Feature engineering strategies for credit card fraud detection paper
        '''

        # TODO
        '''
        It may be useful to add support for updating aggregate statistics with individual new transactions after
        first ''training'' it on a large training set. This could be used to update aggregate statistics in real-
        time using transactions for which we obtain labels in real-time (transactions which are investigated
        by human experts for example)
        '''

    def add_aggregate_features(self, data):
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

        '''
        The following feature appears in Table 1 in [1], but has no explanation otherwise in the paper. Intuitively,
        I suppose it can be an indication of how trustworthy a Card is, in that one that has been in use for
        a very long time may be more trustworthy than a brand new card.
        '''
        data["TimeSinceFirstOrder"] = data.apply(
            lambda row: self.get_time_since_first_order(row=row), axis=1)

        return data

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
            card = row["CardID"]

            if card not in first_order_times_dict:
                first_order_times_dict[card] = row["Date"]

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
        date = row["Date"]

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
            return self.country_all_dict[country]
