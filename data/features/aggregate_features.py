"""
This file provides a class that can be ''trained'' on training data, and then produce new
aggregate features for both training and test data. ''Training'' this class simply consists
of it memorizing various statistics.

Features constructed by this class are primarily inspired by:
"A data mining based system for credit-card fraud detection in e-tail", by
Nuno Carneiro, Gonçalo Figueira, Miguel Costa

Implementation is partially based on https://github.com/kb211/Fraud_Detection2017

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

        # TODO
        '''
        Paper describes clustering countries in four groups based on ratio of fraudulent transactions. Countries with
        less than 30 observations placed in ''intermediate level of risk'' group. No other details provided by paper.
        '''

        # TODO
        '''
        Paper has a ''TimeSinceFirstOrder'' feature in Table 1, with no further explanation in text. I suppose that
        this is an indication of how ''new'' the card is. Intuitively, a card which has been in use for a long time
        may be more trustworthy than a brand new card. So, I guess it could be a useful feature to add.
        '''

        # TODO
        '''
        A bunch of interesting ideas in the Feature engineering strategies for credit card fraud detection paper
        '''

        # TODO
        '''
        It may be useful to add support for updating aggregate statistics with individual new transactions after
        first ''training'' it on a large training set. This could be used to update aggregate statitistics in real-
        time using transactions for which we obtain labels in real-time (transactions which are investigated
        by human experts for example)
        '''

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

        # Thanks Kasper for implementation :D
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

    def get_country_fraud_ratio(self, country):
        """
        Computes the ratio of fraudulent transactions for the given country

        :param country:
            Country (string)
        :return:
            Ratio of transactions corresponding to given country which are fraudulent
        """

        if country not in self.country_all_dict:
            # TODO may be interesting to try average of all countries? Or max, to motivate exploration?
            return 0.0
        else:
            return float(self.country_fraud_dict[country]) / float(self.country_all_dict[country])
