import numpy as np
import pandas
from os.path import join
from datetime import datetime

aggregated_data = pandas.read_csv('./data/real_agg/aggregated_data.csv')
path_agg_data = './data/real_agg/'


def get_default_parameters():

    params = {

        # the seed for the current simulation
        "seed": 666,

        # start and end date of simulation
        'start date': datetime(2016, 1, 1),
        'end date': datetime(2016, 1, 31),

        # max number of authentication steps (at least 1)
        "max authentication steps": 1,

        # number of customers and fraudsters
        "num customers": 100,
        "num fraudsters": 8,

        # merchants
        "num merchants": 28,

        # amount range (same currency)
        "min amount": 0.01,
        "max amount": 10500,

        # currencies
        'currencies': ['EUR', 'GBP', 'USD'],

        # transactions per hour of day
        'transactions per hour': np.load(join(path_agg_data, 'trans_per_hour.npy')),
        # transactions per month in year
        'transactions per month': np.load(join(path_agg_data, 'trans_per_month.npy')),

        # # countries
        # "country prob": pandas.read_csv('./data/aggregated/country_trans_prob.csv'),
        # # currencies per country
        # "currency prob per country": pandas.read_csv('./data/aggregated/currency_trans_prob_per_country.csv')

        # date

        # countries
    }

    return params


def get_path(parameters):
    """ given the parameters, get a unique path to store the outputs """
    # TODO
    path = './results/test'

    return path
