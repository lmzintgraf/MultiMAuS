from model import TransactionModel
import random
from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)


def run_single():

    model = TransactionModel(50, 10, 10)
    for i in range(20):
        model.step()


def run_batch():

    random.seed(666)

    parameters = {"width": 10,
                  "height": 10,
                  "N": range(10, 500, 10)}

    batch_run = BatchRunner(TransactionModel,
                            parameters,
                            iterations=5,
                            max_steps=100,
                            model_reporters={"Gini": compute_gini})
    batch_run.run_all()


if __name__ == '__main__':

    run_single()
