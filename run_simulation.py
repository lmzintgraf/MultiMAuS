from transaction_model import TransactionModel
import parameters
import pandas as pd
import utils_data

def run_single():

    # initialise the model with some parameters
    params = parameters.get_default_parameters()
    model = TransactionModel(params)

    # run the simulation until termination
    while not model.terminated:
        model.step()

    # get the collected data
    agent_vars = model.datacollector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)
    print("\nagent vars:")
    print(agent_vars.head())
    print(agent_vars.shape)

    agent_vars.to_csv(utils_data.FILE_SIMULATOR_LOG, index_label=False)


if __name__ == '__main__':

    run_single()
