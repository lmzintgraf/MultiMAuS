import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, exists
from os import makedirs

FOLDER_REAL_DATA = join(dirname(__file__), 'real_data')
FOLDER_SIMULATOR_INPUT = join(dirname(__file__), 'simulator_input')
FOLDER_SIMULATOR_LOG = join(dirname(__file__), 'simulator_log')
FOLDER_REAL_DATA_ANALYSIS = join(FOLDER_REAL_DATA, 'analysis')

# create the above folders if they don't exist yet
for folder in [FOLDER_REAL_DATA, FOLDER_SIMULATOR_INPUT, FOLDER_SIMULATOR_LOG, FOLDER_REAL_DATA_ANALYSIS]:
    if not exists(folder):
        makedirs(folder)

FILE_ANONYMIZED_DATASET = join(FOLDER_REAL_DATA, 'anonymized_dataset.csv')
FILE_REAL_LOG = join(FOLDER_REAL_DATA, 'transaction_log.csv')
FILE_SIMULATOR_LOG = join(FOLDER_SIMULATOR_LOG, 'transaction_log.csv')


def get_dataset(data_source):
    """
    Returns the dataset (full), and subsets for non-fraud and fraud only.
    :param data_source:    where data comes from, type: str, value: 'real' or 'simulator'
    :return: 
    """

    if data_source == 'real':
        logs_folder = FOLDER_REAL_DATA
    elif data_source == 'simulator':
        logs_folder = FOLDER_SIMULATOR_LOG
    else:
        raise KeyError('dataset_type not known; must be input or output')

    # get dataset from file
    dataset01 = pd.read_csv(join(logs_folder, 'transaction_log.csv'))
    # make the "date" column actually dates
    dataset01["Date"] = pd.to_datetime(dataset01["Date"])

    # for convenience split the dataset into non-fraud(0)/fraud(1)
    dataset0 = dataset01[dataset01["Target"] == 0]
    dataset1 = dataset01[dataset01["Target"] == 1]

    # give the datasets names
    dataset01.name = 'all'
    dataset0.name = 'non-fraud'
    dataset1.name = 'fraud'

    return dataset01, dataset0, dataset1


def get_data_stats(data_source):

    datasets = get_dataset(data_source)

    data_stats_cols = ['all', 'non-fraud', 'fraud']
    data_stats = pd.DataFrame(columns=data_stats_cols)

    data_stats.loc['transactions'] = [d.shape[0] for d in datasets]

    data_stats.loc['transactions/hour'] = [round(d['Date'].apply(lambda x: x.hour).value_counts().sum()/24/365,
                                                 2) for d in datasets]
    data_stats.loc['transactions/day'] = [round(d['Date'].apply(lambda x: x.day).value_counts().sum() / 365,
                                                 2) for d in datasets]
    data_stats.loc['transactions/week'] = [round(d['Date'].apply(lambda x: x.week).value_counts().sum() / 52,
                                                2) for d in datasets]
    data_stats.loc['transactions/month'] = [round(d['Date'].apply(lambda x: x.month).value_counts().sum() / 12,
                                                2) for d in datasets]

    data_stats.loc['cards'] = [len(d["CardID"].unique()) for d in datasets]
    data_stats.loc['cards, single use'] = [sum(d["CardID"].value_counts() == 1) for d in datasets]
    data_stats.loc['cards, multi use'] = [sum(d["CardID"].value_counts() > 1) for d in datasets]

    data_stats.loc['first transaction'] = [min(d["Date"]).date() for d in datasets]
    data_stats.loc['last transaction'] = [max(d["Date"]).date() for d in datasets]

    data_stats.loc['min amount'] = [min(d["Amount"]) for d in datasets]
    data_stats.loc['max amount'] = [max(d["Amount"]) for d in datasets]
    data_stats.loc['avg amount'] = [np.average(d["Amount"]) for d in datasets]

    data_stats.loc['num merchants'] = [len(d["MerchantID"].unique()) for d in datasets]

    data_stats.loc['countries'] = [len(d["Country"].unique()) for d in datasets]
    data_stats.loc['currencies'] = [len(d["Currency"].unique()) for d in datasets]

    data_stats.loc['min trans/card'] = [min(d["CardID"].value_counts()) for d in datasets]
    data_stats.loc['max trans/card'] = [max(d["CardID"].value_counts()) for d in datasets]
    data_stats.loc['avg trans/card'] = [np.average(d["CardID"].value_counts()) for d in datasets]

    return data_stats


def get_grouped_prob(group_by, col_name):
    grouped_prob = get_dataset()[0].groupby([group_by, col_name]).size()
    grouped_prob = grouped_prob.groupby(level=0).apply(lambda x: x / sum(x))
    return grouped_prob


def get_transaction_dist(col_name):
    """ calculate fractions of transactions for given column """
    possible_vals = get_dataset()[0][col_name].value_counts().unique()
    trans_count = pd.DataFrame(0, index=possible_vals, columns=['all', 'non-fraud', 'fraud'])
    trans_count['all'] = get_dataset()[0][col_name].value_counts().value_counts()
    trans_count['non-fraud'] = get_dataset()[1][col_name].value_counts().value_counts()
    trans_count['fraud'] = get_dataset()[1][col_name].value_counts().value_counts()
    trans_count = trans_count.fillna(0)
    trans_count /= np.sum(trans_count.values, axis=0)

    # save
    trans_count.to_csv(join(FOLDER_SIMULATOR_INPUT, 'fract-dist.csv'.format(col_name)), index_label=False)

    # print
    print(col_name)
    print(trans_count)
    print("")

    return trans_count


def plot_hist_num_transactions(trans_frac, col_name):
    """ method to plot histogram of number of transactions for a column """
    plt.figure(figsize=(10, 7))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.bar(range(trans_frac.shape[0]), trans_frac.values[:, i], label=trans_frac.index[i])
        plt.ylabel('num transactions')
        if i == 2:
            plt.xlabel(col_name)
    plt.savefig(join(FOLDER_SIMULATOR_INPUT, '{}_num-trans_hist'.format(col_name)))
    plt.close()


def plot_bar_trans_prob(trans_frac, col_name, file_name=None):
    """ method to plot bar plot of number of transactions for a column """
    plt.figure()
    bottoms = np.vstack((np.zeros(3), np.cumsum(trans_frac, axis=0)))
    for i in range(trans_frac.shape[0]):
        plt.bar((0, 1, 2), trans_frac.values[i], label=trans_frac.index[i], bottom=bottoms[i])
    plt.xticks([0, 1, 2], ['all', 'non-fraud', 'fraud'])
    h = plt.ylabel('%')
    h.set_rotation(0)
    plt.title("{} Distribution".format(col_name))
    plt.legend()
    if not file_name:
        file_name = col_name
    plt.savefig(join(FOLDER_SIMULATOR_INPUT, '{}_num-trans_bar'.format(file_name)))
    plt.close()


if __name__ == '__main__':
    print(get_data_stats('simulator'))
