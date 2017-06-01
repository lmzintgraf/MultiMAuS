"""
Example script of how to use the ApateGraphFeatures class
"""

from data.features.apate_graph_features import ApateGraphFeatures
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

NUM_TRAINING_INSTANCES = 50000        # 50K for training should be fine

def run_test():
    # load dataframe
    df = pd.read_csv('../../real_data/transaction_log.csv')
    df["Date"] = pd.to_datetime(df["Date"])

    df_train = df.iloc[:NUM_TRAINING_INSTANCES]
    df_test = df.iloc[NUM_TRAINING_INSTANCES:]
    df = None

    graph_features = ApateGraphFeatures(df_train)
    graph_features.add_graph_features(df_train)
    graph_features.add_graph_features(df_test)

    df_train = df_train.drop(["Date", "CardID", "MerchantID", "Currency", "Country"], 1)
    df_test = df_test.drop(["Date", "CardID", "MerchantID", "Currency", "Country"], 1)

    training_labels = df_train["Target"].values
    test_labels = df_test["Target"].values
    df_train = df_train.drop(["Target"], 1)
    df_test = df_test.drop(["Target"], 1)

    logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
    logreg.fit(df_train, training_labels)

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
