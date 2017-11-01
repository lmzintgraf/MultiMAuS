import numpy as np
from datetime import datetime
from pytz import timezone
import matplotlib.pyplot as plt
from agent_qlean import QLearnAgent
from agent_safe import SafeAgent
from environment import Environment
from simulator import parameters
from simulator.transaction_model import TransactionModel
from experiments import rewards
from authenticators.simple_authenticators import RandomAuthenticator, \
    HeuristicAuthenticator, OracleAuthenticator, NeverSecondAuthenticator, \
    AlwaysSecondAuthenticator


auths = [
         (Environment(SafeAgent('random')), 'Safe Agent'),
         (RandomAuthenticator(), 'Random'),
         (OracleAuthenticator(), 'Oracle'),
         (HeuristicAuthenticator(50), 'Heuristic'),
         (NeverSecondAuthenticator(), 'NeverSecond'),
         (AlwaysSecondAuthenticator(), 'AlwaysSecond'),
         (Environment(QLearnAgent('random')), 'Q-Learning (from scratch)'),
]

authenticator = None
auth_name = ''

for k in range(len(auths)+1):

    if auth_name != 'Q-Learning (from scratch)':
        authenticator, auth_name = auths[k]
    else:  # if we just did Q-Learning, run it again with the pre-trained one
        auth_name = 'Q-Learning (pre-trained)'

    print("-----")
    print(auth_name)
    print("-----")

    # the parameters for the simulation
    params = parameters.get_default_parameters()
    params['init_satisfaction'] = 0.9
    params['stay_prob'] = [0.9, 0.6]
    params['num_customers'] = 100
    params['num_fraudsters'] = 10
    params['end_date'] = datetime(2016, 12, 31).replace(tzinfo=timezone('US/Pacific'))

    # get the model for transactions
    model = TransactionModel(params, authenticator=authenticator)

    # run
    while not model.terminated:
        model.step()

    agent_vars = model.log_collector.get_agent_vars_dataframe()
    agent_vars.index = agent_vars.index.droplevel(1)

    monetary_rewards = rewards.monetary_reward_per_timestep(agent_vars)

    plt.plot(range(len(monetary_rewards)), np.cumsum(monetary_rewards), label=auth_name)

plt.xlabel('time step')
plt.ylabel('monetary reward (cumulative)')
plt.legend()
plt.tight_layout()
plt.show()
