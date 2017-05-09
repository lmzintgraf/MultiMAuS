from transaction_model import TransactionModel
# from mesa.batchrunner import BatchRunner
# import matplotlib.pyplot as plt
import parameters


def run_single():

    # initialise the model with some parameters
    sim_parameters = parameters.get_default_parameters()
    model = TransactionModel(sim_parameters)

    # run the simulation for num_sim_steps
    for i in range(sim_parameters["num simulation steps"]):
        model.step()

    # get the collected data
    agent_vars = model.datacollector.get_agent_vars_dataframe()
    print(agent_vars.head())


# def run_batch():
#
#     random.seed(666)
#
#     parameters = {"width": 10,
#                   "height": 10,
#                   "N": range(10, 500, 10)}
#
#     batch_run = BatchRunner(TransactionModel,
#                             parameters,
#                             iterations=5,
#                             max_steps=100,
#                             model_reporters={"Gini": compute_gini})
#     batch_run.run_all()


if __name__ == '__main__':

    run_single()
