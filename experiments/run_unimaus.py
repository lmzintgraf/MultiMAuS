from simulator import parameters
from simulator.transactions_unimaus import UniMausTransactionModel
import time
from simulator.customer_unimaus import GenuineCustomer, FraudulentCustomer
from experiments import result_handling
import numpy as np
import datetime
import pandas as pd


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


def run_single():

    start_time = time.time()

    # initialise the model with some parameters
    params = parameters.get_default_parameters().copy()
    model = UniMausTransactionModel(params, GenuineCustomer, FraudulentCustomer)

    # run the simulation until termination
    while not model.terminated:
        model.step()

    # get the collected data
    agent_vars = model.log_collector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)

    print('customers left:', len(model.customers))
    print('fraudsters left:', len(model.fraudsters))
    print('simulation took ', round((time.time() - start_time)/60., 2), ' minutes')

    # make sure we didn't accidentally changed the default parameters
    check_parameter_consistency(params)

    # save the results
    result_handling.save_results(model)

if __name__ == '__main__':

    run_single()
