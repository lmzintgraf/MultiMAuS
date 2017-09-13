from os.path import isdir, join, dirname, exists
from os import mkdir
import pickle
import numpy as np
import datetime
from simulator import parameters
import pandas as pd


FOLDER_RESULTS = join(dirname(__file__), 'results')
FILE_RESULTS_IDX = join(FOLDER_RESULTS, 'curr_idx.txt')


def get_result_idx():
    for line in open(FILE_RESULTS_IDX):
        if line.strip():  # line contains eol character(s)
            return int(line)


def update_result_idx(old_result_idx):
    # increase result counter by one
    f = open(FILE_RESULTS_IDX, 'w+')
    f.write(str(old_result_idx + 1))
    f.close()


def get_params_path(result_idx):
    return join(FOLDER_RESULTS, '{}_parameters.pkl'.format(result_idx))


def get_transaction_log_path(result_idx):
    return join(FOLDER_RESULTS, '{}_transaction_log.csv'.format(result_idx))


def get_satisfaction_log_path(result_idx):
    return join(FOLDER_RESULTS, '{}_satisfaction_log.csv'.format(result_idx))


def save_results(model):

    # create a folder to save results in
    if not isdir(FOLDER_RESULTS):
        mkdir(FOLDER_RESULTS)

    if not exists(FILE_RESULTS_IDX):
        f = open(FILE_RESULTS_IDX, 'w+')
        f.write(str(0))
        f.close()

    result_idx = get_result_idx()

    # retrieve parameters for current experiment
    parameters = model.parameters
    # add the name of the authenticator to the parameters
    parameters['authenticator'] = model.authenticator.__class__.__name__
    # save the parameters
    pickle.dump(parameters, open(get_params_path(result_idx), 'wb'), pickle.HIGHEST_PROTOCOL)

    # save the transaction logs
    agent_vars = model.log_collector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)
    path_transaction_log = get_transaction_log_path(result_idx)
    agent_vars.to_csv(path_transaction_log, index_label=False)

    # save the satisfaction per timestep
    model_vars = model.log_collector.get_model_vars_dataframe()
    path_satisfaction_log = get_satisfaction_log_path(result_idx)
    model_vars.to_csv(path_satisfaction_log, index_label=False)

    # save some customer properties
    FOLDER_CUST_PROPS = join(FOLDER_RESULTS, '{}_cust_props'.format(result_idx))
    mkdir(FOLDER_CUST_PROPS)

    for i in range(5):
        # for customers
        np.save(join(FOLDER_CUST_PROPS, 'cust{}_trans_prob_monthday'.format(i)), model.customers[i].trans_prob_monthday)
        np.save(join(FOLDER_CUST_PROPS, 'cust{}_trans_prob_month'.format(i)), model.customers[i].trans_prob_month)
        np.save(join(FOLDER_CUST_PROPS, 'cust{}_trans_prob_hour'.format(i)), model.customers[i].trans_prob_hour)
        np.save(join(FOLDER_CUST_PROPS, 'cust{}_trans_prob_weekday'.format(i)), model.customers[i].trans_prob_weekday)

        # for fraudsters
        np.save(join(FOLDER_CUST_PROPS, 'fraud{}_trans_prob_monthday'.format(i)), model.fraudsters[i].trans_prob_monthday)
        np.save(join(FOLDER_CUST_PROPS, 'fraud{}_trans_prob_month'.format(i)), model.fraudsters[i].trans_prob_month)
        np.save(join(FOLDER_CUST_PROPS, 'fraud{}_trans_prob_hour'.format(i)), model.fraudsters[i].trans_prob_hour)
        np.save(join(FOLDER_CUST_PROPS, 'fraud{}_trans_prob_weekday'.format(i)), model.fraudsters[i].trans_prob_weekday)

    print("saved results under result index {}".format(result_idx))

    update_result_idx(result_idx)


def get_parameters(result_idx):
    return pickle.load(open(get_params_path(result_idx), 'rb'))


def check_parameter_consistency(params1):

    params2 = parameters.get_default_parameters()

    # make sure we didn't accidentally change the input parameters
    for key in params1.keys():
        try:
            if isinstance(params1[key], np.ndarray):
                assert np.sum(params1[key] - params2[key]) == 0
            elif isinstance(params1[key], float) or isinstance(params1[key], int):
                assert params1[key] - params2[key] == 0
            elif isinstance(params1[key], datetime.date):
                pass
            elif isinstance(params1[key], pd.DataFrame):
                assert np.sum(params1[key].values - params2[key].values) == 0
            elif isinstance(params1[key], list):
                for i in range(len(params1[key])):
                    assert np.sum(params1[key][i].values - params2[key][i].values) == 0
            else:
                print("unknown type", key, type(params1[key]))
        except AssertionError:
            print("!! params changed:", key)
