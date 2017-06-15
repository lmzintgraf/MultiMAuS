from data import utils_data
from simulator import parameters
from simulator.transactions_unimaus import UniMausTransactionModel
import time
from simulator.customer_unimaus import GenuineCustomer, FraudulentCustomer


def run_single():

    start_time = time.time()

    # initialise the model with some parameters
    params = parameters.get_default_parameters()
    model = UniMausTransactionModel(params, GenuineCustomer, FraudulentCustomer)

    # run the simulation until termination
    while not model.terminated:
        model.step()

    # get the collected data
    agent_vars = model.log_collector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)

    agent_vars.to_csv(utils_data.FILE_SIMULATOR_LOG, index_label=False)

    print('customers left:', len(model.customers))
    print('fraudsters left:', len(model.fraudsters))
    print(utils_data.get_data_stats('simulator'))
    print('simulation took ', round((time.time() - start_time)/60., 2), ' minutes')

if __name__ == '__main__':

    run_single()
