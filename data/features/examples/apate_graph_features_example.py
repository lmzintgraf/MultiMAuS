"""
Example script of how to use the ApateGraphFeatures class

@author Dennis Soemers
"""

from data.features.apate_graph_features import ApateGraphFeatures
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

    # number of entries that will be used for training Machine Learning model (augmented with extra features)
    NUM_MODEL_LEARNING_INSTANCES = 25000

    # load dataframe
    df = pd.read_csv(DATA_FILEPATH)

    # convert date column to proper type
    df["Global_Date"] = pd.to_datetime(df["Global_Date"])
    df["Local_Date"] = pd.to_datetime(df["Local_Date"])

    # extract part of data to use for learning features
    df_feature_learning = df.iloc[:NUM_FEATURE_LEARNING_INSTANCES]

    # extract part of data to use for learning ML model
    df_model_learning = df.iloc[
                        NUM_FEATURE_LEARNING_INSTANCES:(NUM_FEATURE_LEARNING_INSTANCES+NUM_MODEL_LEARNING_INSTANCES)]

    # extract part of data for testing (evaluating performance of trained model)
    df_test = df.iloc[(NUM_FEATURE_LEARNING_INSTANCES+NUM_MODEL_LEARNING_INSTANCES):]

    # set the complete dataset to None so we don't accidentally use it anywhere below
    df = None

    # construct network and run convergence procedure on the feature learning dataset
    graph_features = ApateGraphFeatures(df_feature_learning)

    # augment our model learning dataset with extra features
    print(str(datetime.now()), ": Starting computation of APATE graph features for model learning data...")
    graph_features.add_graph_features(df_model_learning)

    # augment our test dataset with extra features
    print(str(datetime.now()), ": Starting computation of APATE graph features for test data...")
    graph_features.add_graph_features(df_test)

    # remove features which we no longer want to use in machine learning
    df_model_learning = df_model_learning.drop(
        ["Global_Date", "Local_Date", "CardID", "MerchantID", "Currency", "Country"], 1)
    df_test = df_test.drop(
        ["Global_Date", "Local_Date", "CardID", "MerchantID", "Currency", "Country"], 1)

    # extract the ground truth labels
    training_labels = df_model_learning["Target"].values
    test_labels = df_test["Target"].values

    # remove ground truth labels from datasets, don't want to allow ML models to cheat by using them
    df_model_learning = df_model_learning.drop(["Target"], 1)
    df_test = df_test.drop(["Target"], 1)

    # construct a simple ML model for testing
    logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')

    # train the model
    logreg.fit(df_model_learning, training_labels)

    accuracy = logreg.score(df_test, test_labels)
    print("accuracy = ", accuracy)

    '''
    sns.pairplot(df_test, vars=["CHScore", "MerScore", "TrxScore"], hue="Target", plot_kws={"alpha": 0.5})
    sns.pairplot(df_test, vars=["CHScore_ST", "MerScore_ST", "TrxScore_ST"], hue="Target", plot_kws={"alpha": 0.5})
    sns.pairplot(df_test, vars=["CHScore_MT", "MerScore_MT", "TrxScore_MT"], hue="Target", plot_kws={"alpha": 0.5})
    sns.pairplot(df_test, vars=["CHScore_LT", "MerScore_LT", "TrxScore_LT"], hue="Target", plot_kws={"alpha": 0.5})

    plt.show()
    '''

if __name__ == '__main__':
    run_test()
