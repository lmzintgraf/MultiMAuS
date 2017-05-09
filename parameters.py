import numpy as np


def get_default_parameters():

    params = {

        # the seed for the current simulation
        "seed": 666,

        # number of steps for the simulation
        "num simulation steps": 10,

        # max number of authentication steps (at least 1)
        "max authentication steps": 1,

        # number of customers and fraudsters
        "num customers": 10,
        "num fraudsters": 2,
        "num merchants": 5
    }

    return params


def get_path(parameters):
    """ given the parameters, get a unique path to store the outputs """
    path = './results/test'

    return path