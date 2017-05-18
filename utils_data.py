import pandas
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

FOLDER_INPUT_RAW = 'data/input_raw/'
FOLDER_INPUT_AGG = 'data/input_agg/'
FOLDER_OUTPUT_RAW = 'data/output_raw/'
FOLDER_OUTPUT_AGG = 'data/input_raw'

FILE_INPUT_RAW = join(FOLDER_INPUT_RAW, 'anonymized_dataset.csv')
FILE_INPUT_PREPROCESSED = join(FOLDER_INPUT_RAW, 'dataset_preprocessed.csv')
FILE_OUTPUT_RAW = join(FOLDER_OUTPUT_RAW, 'transaction_log.csv')

def get_folder_input_raw():
    return FOLDER_INPUT_RAW


def get_folder_input_agg():
    return FOLDER_INPUT_AGG


def get_path_input_raw():
    return FILE_INPUT_RAW


def get_path_input_preprocessed():
    return FILE_INPUT_PREPROCESSED


def get_path_output_raw():
    return FILE_OUTPUT_RAW


def get_dataset():
    """ Returns the dataset (full), and subsets for non-fraud and fraud only """

    # get dataset from file
    dataset01 = pandas.read_csv(get_path_input_preprocessed())
    # make the "date" column actually dates
    dataset01["Date"] = pandas.to_datetime(dataset01["Date"])

    # for convenience split the dataset into non-fraud(0)/fraud(1)
    dataset0 = dataset01[dataset01["Target"] == 0]
    dataset1 = dataset01[dataset01["Target"] == 1]

    # give the datasets names
    dataset01.name = 'datasetAll'
    dataset0.name = 'datasetGenuine'
    dataset1.name = 'datasetFraud'

    return dataset01, dataset0, dataset1


def get_transaction_prob(col_name, d01=get_dataset()[0], d0=get_dataset()[1], d1=get_dataset()[2]):
    """ calculate fractions of transactions for given column """
    num_trans = pandas.DataFrame(0, index=d01[col_name].value_counts().index, columns=['all', 'non-fraud', 'fraud'])
    num_trans['all'] = d01[col_name].value_counts()
    num_trans['non-fraud'] = d0[col_name].value_counts()
    num_trans['fraud'] = d1[col_name].value_counts()
    num_trans = num_trans.fillna(0)
    num_trans /= np.sum(num_trans, axis=0)
    return num_trans


def get_grouped_prob(group_by, col_name):
    grouped_prob = get_dataset()[0].groupby([group_by, col_name]).size()
    grouped_prob = grouped_prob.groupby(level=0).apply(lambda x: x / sum(x))
    return grouped_prob


def get_transaction_dist(col_name):
    """ calculate fractions of transactions for given column """
    possible_vals = get_dataset()[0][col_name].value_counts().unique()
    trans_count = pandas.DataFrame(0, index=possible_vals, columns=['all', 'non-fraud', 'fraud'])
    trans_count['all'] = get_dataset()[0][col_name].value_counts().value_counts()
    trans_count['non-fraud'] = get_dataset()[1][col_name].value_counts().value_counts()
    trans_count['fraud'] = get_dataset()[1][col_name].value_counts().value_counts()
    trans_count = trans_count.fillna(0)
    trans_count /= np.sum(trans_count.values, axis=0)

    # save
    trans_count.to_csv(join(get_folder_input_agg(), 'fract-dist.csv'.format(col_name)), index_label=False)

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
    plt.savefig(join(get_folder_input_agg(), '{}_num-trans_hist'.format(col_name)))
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
    plt.savefig(join(get_folder_input_agg(), '{}_num-trans_bar'.format(file_name)))
    plt.close()