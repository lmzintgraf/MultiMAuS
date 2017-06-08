from data import utils_data
from simulator import parameters
from simulator.transactions_unimaus import UniMausTransactionModel
import time


def run_single():

    start_time = time.time()

    # initialise the model with some parameters
    params = parameters.get_default_parameters()
    model = UniMausTransactionModel(params)

    # run the simulation until termination
    while not model.terminated:
        model.step()

    # get the collected data
    agent_vars = model.log_collector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)
    print(agent_vars.head())
    print(agent_vars.shape)

    agent_vars.to_csv(utils_data.FILE_SIMULATOR_LOG, index_label=False)

    print('simulation took ', round((time.time() - start_time)/60., 2), ' minutes')

if __name__ == '__main__':

    run_single()
