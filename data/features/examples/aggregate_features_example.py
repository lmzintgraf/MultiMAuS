"""
Example script for how to use the AggregateFeatures class

@author Dennis Soemers
"""

from data.features.aggregate_features import AggregateFeatures
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

    # augment the training data with extra features. IMPORTANT: set include_test_data_in_history=False,
    # because the called function already automatically uses training data (and in this case, our training
    # data is simultaneously ''test data'')
    aggregate_features.add_aggregate_features(df_training, include_test_data_in_history=False)

    # augment the test data with extra features. In this case, also allow it to be used as history
    aggregate_features.add_aggregate_features(df_test, include_test_data_in_history=True)

    print(df_training.head())
    print(df_test.head())

if __name__ == '__main__':
    #import cProfile

    #cProfile.run('run_test()')
    run_test()
