"""
Simple example script for how to use functions from feature_combinations.py to add
extra features.

@author Dennis Soemers
"""

from data.features import feature_combinations
import pandas as pd

def run_example():
    # filepath to load dataframe from
    # NOTE: currently using dataset which was not preprocessed yet here. This is because
    # we need the GeoCode feature for our example, which is removed by preprocess_data_raw.py
    DATA_FILEPATH = '../../real_data/anonymized_dataset.csv'

    # load dataframe
    df = pd.read_csv(DATA_FILEPATH)

    # add a binary feature to our dataset, which tells us whether or not the Country and GeoCode values of
    # a transaction are equal
    df = feature_combinations.pair_equality(df, "Country", "GeoCode", "Country_GeoCode_Eq")

    '''
    Note: the above may be a useful feature according to "A data mining based system for credit-card fraud
    detection in e-tail", by Nuno Carneiro, Gonçalo Figueira, Miguel Costa      (Section 4.2)
    '''

    print(df.head())

if __name__ == '__main__':
    run_example()
