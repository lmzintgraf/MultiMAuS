"""
Example script for how to use the AggregateFeatures class

@author Dennis Soemers
"""

from data.features.aggregate_features import AggregateFeatures
from data import utils_data
from os.path import join
import pandas as pd

def run_test():
    # specify filepath here to load complete dataset from (training + test data)
    DATA_FILEPATH = '../../real_data/transaction_log.csv'

    # number of entries that will be used for training
    NUM_TRAINING_INSTANCES = 50000

    # load dataframe
    df = pd.read_csv(DATA_FILEPATH)

    # convert date columns to proper type
    df["Global_Date"] = pd.to_datetime(df["Global_Date"])
    df["Local_Date"] = pd.to_datetime(df["Local_Date"])

    # extract part of data to use for training
    df_training = df.iloc[:NUM_TRAINING_INSTANCES]

    # extract part of data for testing (evaluating performance of trained model)
    df_test = df.iloc[NUM_TRAINING_INSTANCES:]

    # set the complete dataset to None so we don't accidentally use it anywhere below
    df = None

    # construct object that can compute features for us, based on training data
    aggregate_features = AggregateFeatures(df_training)

    # augment the training data with extra features.
    aggregate_features.add_aggregate_features(df_training)

    # augment the test data with extra features. In this case, also allow it to be used as history
    # IMPORTANT: first we let our aggregate_features object do an unlabeled-data update from this test data
    # this unlabeled update won't ''cheat'' and use the labels, but early transactions in the test data can
    # be used in feature engineering for later transactions in the same test data
    aggregate_features.update_unlabeled(df_test)
    aggregate_features.add_aggregate_features(df_test)

    #print(df_training.head())
    #print(df_test.head())

    df_training.to_csv(join(utils_data.FOLDER_REAL_DATA, 'aggregate_features_training_data.csv'), index_label=False)
    df_test.to_csv(join(utils_data.FOLDER_REAL_DATA, 'aggregate_features_test_data.csv'), index_label=False)

if __name__ == '__main__':
    #import cProfile

    #pr = cProfile.Profile()
    #pr.enable()
    run_test()
    #pr.disable()
    #pr.print_stats(sort='cumtime')
