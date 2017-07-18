"""
Example script of how to use the ApateGraphFeatures + AggregateFeatures class

@author Dennis Soemers
"""

from data.features.aggregate_features import AggregateFeatures
from data.features.apate_graph_features import ApateGraphFeatures
from data import utils_data
from os.path import join
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model

def run_test():
    # specify filepath here to load complete dataset from (training + test data)
    DATA_FILEPATH = '../../real_data/transaction_log.csv'

    # number of entries that will be used for feature learning (convergence procedure on network)
    NUM_FEATURE_LEARNING_INSTANCES = 25000

    # load dataframe
    df = pd.read_csv(DATA_FILEPATH)

    # convert date column to proper type
    df["Global_Date"] = pd.to_datetime(df["Global_Date"])
    df["Local_Date"] = pd.to_datetime(df["Local_Date"])

    # extract part of data to use for learning features
    df_feature_learning = df.iloc[:NUM_FEATURE_LEARNING_INSTANCES]

    # extract rest of the data
    df_rest = df.iloc[NUM_FEATURE_LEARNING_INSTANCES:]

    # aggregate features can be initialized with feature learning data
    aggregate_features = AggregateFeatures(df_feature_learning)

    # set the complete dataset to None so we don't accidentally use it anywhere below
    df = None

    # construct network and run convergence procedure on the feature learning dataset
    graph_features = ApateGraphFeatures(df_feature_learning)

    # augment our remaining dataset with extra features
    print(str(datetime.now()), ": Starting computation of APATE graph features...")
    graph_features.add_graph_features(df_rest)

    print(str(datetime.now()), ": Starting computation of aggregate features...")
    aggregate_features.update_unlabeled(df_rest)
    aggregate_features.add_aggregate_features(df_rest)

    # remove features which we no longer want to use in machine learning
    df_rest.drop(["Global_Date", "Local_Date", "MerchantID", "Currency", "Country"],
                 inplace=True, axis=1)

    df_rest = df_rest[[col for col in df_rest if col != "Target" and col != "CardID"] + ["CardID", "Target"]]

    # save the data with extra features
    df_rest.to_csv(join(utils_data.FOLDER_REAL_DATA, 'extra_features_data.csv'), index_label=False)

    print(df_rest.head())

if __name__ == '__main__':
    run_test()
