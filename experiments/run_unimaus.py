from simulator import parameters
from simulator.transaction_model import TransactionModel
import time
from experiments import result_handling


def run_single():

    start_time = time.time()

    # initialise the model with some parameters
    params = parameters.get_default_parameters().copy()
    model = TransactionModel(params)

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
    result_handling.check_parameter_consistency(params)

    # save the results
    result_handling.save_results(model)

if __name__ == '__main__':
    
    run_single()
