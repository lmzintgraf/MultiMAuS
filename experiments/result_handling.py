from os.path import isdir, join, dirname, exists
from os import mkdir
import pickle

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


def save_results(model):

    # create a folder to save results in
    if not isdir(FOLDER_RESULTS):
        mkdir(FOLDER_RESULTS)

    if not exists(FILE_RESULTS_IDX):
        f = open(FILE_RESULTS_IDX, 'w+')
        f.write(str(0))
        f.close()

    result_idx = get_result_idx()

    # save the parameters
    parameters = model.parameters
    pickle.dump(parameters, open(get_params_path(result_idx), 'wb'), pickle.HIGHEST_PROTOCOL)

    # save the transaction logs
    agent_vars = model.log_collector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)
    path_transaction_log = join(FOLDER_RESULTS, '{}_transaction_log.csv'.format(result_idx))
    agent_vars.to_csv(path_transaction_log, index_label=False)

    print("saved results under result index {}".format(result_idx))

    update_result_idx(result_idx)


def get_parameters(result_idx):
    return pickle.load(open(get_params_path(result_idx), 'rb'))
