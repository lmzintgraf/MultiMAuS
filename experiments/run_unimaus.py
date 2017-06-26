from simulator import parameters
from simulator.transactions_unimaus import UniMausTransactionModel
import time
from simulator.customer_unimaus import GenuineCustomer, FraudulentCustomer
from experiments.results import result_handling


def run_single():

    start_time = time.time()

    # initialise the model with some parameters
    params = parameters.get_default_parameters()
    model = UniMausTransactionModel(params, GeenuineCustomer, FraudulentCustomer)

    # run the simulation until termination
    while not model.terminated:
        model.step()

    # get the collected data
    agent_vars = model.log_collector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)

    print('customers left:', len(model.customers))
    print('fraudsters left:', len(model.fraudsters))
    print('simulation took ', round((time.time() - start_time)/60., 2), ' minutes')

    result_handling(model)

if __name__ == '__main__':

    run_single()
