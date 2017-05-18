# Multi-MAuS: A Multi-Modal Authentication Simulator

Simulator for online credit card transactions with multi-modal authentication.


### DATA

#### Data handling

- The original (private!) data is in  the folder data/real_log/transaction_log.csv. It can be produced from the raw data (anonymized_dataset.csv) with preprocess_data_raw.py.

- Aggregated information from the original data is stored in the folder data/real_agg. Some of this is used as direct input to the simulator.

- The simulator saves the logs in data/simulator_log/transaction_log.csv.

- In the end, the simulation will come only with some 