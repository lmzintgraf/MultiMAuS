from datetime import datetime
from os.path import join
import numpy as np
from data import utils_data
import pandas as pd

aggregated_data = pd.read_csv(join(utils_data.FOLDER_SIMULATOR_INPUT, 'aggregated_data.csv'), index_col=0)
trans_per_year = np.array(aggregated_data.loc['transactions'].values, dtype=np.float)[1:]


def get_default_parameters():

    params = {

        # seed for random number generator of current simulation
        "seed": 666,

        # start and end date of simulation
        'start date': datetime(2016, 1, 1),
        'end date': datetime(2016, 12, 31),

        # max number of authentication steps (at least 1)
        "max authentication steps": 1,

        # number of customers and fraudsters at beginning of simulation
        "start num customers": 200,
        "start num fraudsters": 20,

        # number of merchants at the beginning of simulation
        "num merchants": 28,

        # amount range (in Euros)
        "min amount": 0.01,
        "max amount": 10500,

        # currencies
        'currencies': ['EUR', 'GBP', 'USD'],

        # total number of transactions we want in one year
        'trans_per_year': trans_per_year,

        # standard deviation for total num transactions
        'std_transactions': [trans_per_year[0]/10, trans_per_year[1]/10],

        # transactions per day in a month
        'frac_monthday': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'monthday_frac.npy')),
        # transactions per day in a week
        'frac_weekday': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'weekday_frac.npy')),
        # transactions per month in a year
        'frac_month': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'month_frac.npy')),
        # transactions hour in a day
        'frac_hour': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'hour_frac.npy')),

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
