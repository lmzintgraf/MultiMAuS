from transaction_model import TransactionModel
import parameters


def run_single():

    # initialise the model with some parameters
    params = parameters.get_default_parameters()
    model = TransactionModel(params)

    # run the simulation until termination
    while not model.terminated:
        model.step()

    # get the collected data
    agent_vars = model.datacollector.get_agent_vars_dataframe()
    print(agent_vars.head())


if __name__ == '__main__':

    run_single()
