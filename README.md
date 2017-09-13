# Multi-MAuS: A Multi-Modal Authentication Simulator

Simulator for online credit card transactions that supports 
multi-modal authentication. 

Please see the corresponding paper (MultiMAuS.pdf) for a
detailed description of the simulator.

### RUNNING THE SIMULATOR

The main script that should be executed for a simulation is
`simulator/transaction_model.py`. It needs a dictionary of
parameters as input. The default parameters that 
are used as input for the simulator can be found in 
`simulator.parameters.py`.

The scripts we used for the experiments in the paper are:
`experiments/run_unimaus.py` and 
`experiments/run_multimaus.py`.

### DATA

The simulator takes aggregated data as input which is obtained
from a real (private) dataset of credit card transactions.
This aggregated data is taken from ./data/simulator_input.

See readme file in data folder for more information.

### SIMULATOR

We use the agent modelling framework mesa (https://github.com/projectmesa/mesa).

### FEATURE EXTRACTION

We are currently working on creating features that can be used
for learning (e.g., classification) from the transaction logs.
Most of this can be found in `data/features`. Note however that
this is under development.
