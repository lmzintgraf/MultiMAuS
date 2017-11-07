# Multi-MAuS: A Multi-Modal Authentication Simulator

Simulator for online credit card transactions that supports 
multi-modal authentication. 

Please see the corresponding paper (MultiMAuS.pdf) for a
detailed description of the simulator. Citation:

> Zintgraf, L. M.; Lopez-Rojas, E. A.; Roijers, D. M.; and Nowé, A. 2017. MultiMAuS: A Multi-Modal Authentication Simulator for Fraud Detection Research. In Affenzeller, M.; Bruzzone, A. G.; Jimenez, E.; Longo, F.; and Piera, M. A., eds., *29th European Modeling and Simulation Symp. (EMSS 2017)*, 360–370. Curran Associates, Inc.

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

### FEATURE EXTRACTION

We also provide features that can be used for learning (e.g., classification) 
from the transaction logs. For more information, see `data/features`.