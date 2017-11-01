import numpy as np
import os
from datetime import datetime
from pytz import timezone
import matplotlib.pyplot as plt
from agent_qlean import QLearnAgent
from agent_bandit import BanditAgent
from environment import Environment
from simulator import parameters
from simulator.transaction_model import TransactionModel
from experiments import rewards
from authenticators.simple_authenticators import RandomAuthenticator, \
    HeuristicAuthenticator, OracleAuthenticator, NeverSecondAuthenticator, \
    AlwaysSecondAuthenticator


auths = [
         # (Environment(BanditAgent(do_reward_shaping=True)), 'Bandit (reward shaping)'),
         # (RandomAuthenticator(), 'Random'),
         # (OracleAuthenticator(), 'Oracle'),
         # (HeuristicAuthenticator(50), 'Heuristic'),
         # (NeverSecondAuthenticator(), 'NeverSecond'),
         # (AlwaysSecondAuthenticator(), 'AlwaysSecond'),

         (Environment(QLearnAgent('zero', 0.01, 0.1, 0.1, False)), 'Q-Learn'),
         (Environment(QLearnAgent('zero', 0.01, 0.1, 0.1, True)), 'Q-Learn with reward shaping'),

         (Environment(BanditAgent()), 'Bandit'),
         (Environment(BanditAgent(do_reward_shaping=True)), 'Bandit with reward shaping'),
]

authenticator = None
auth_name = ''

for k in range(len(auths)):

    if auth_name != 'Q-Learning (from scratch)':
        authenticator, auth_name = auths[k]
    else:  # if we just did Q-Learning, run it again with the pre-trained one
        auth_name = 'Q-Learning (pre-trained)'

    seed = 666

    print("-----")
    print(auth_name)
    print("-----")

    sum_monetary_rewards = None

    for i in range(1):

        # the parameters for the simulation
        params = parameters.get_default_parameters()
        params['seed'] = seed
        params['init_satisfaction'] = 0.9
        params['stay_prob'] = [0.9, 0.6]
        params['num_customers'] = 100
        params['num_fraudsters'] = 10
        params['end_date'] = datetime(2016, 12, 31).replace(tzinfo=timezone('US/Pacific'))

        path = 'results/{}_{}_{}_{}_{}_{}'.format(auth_name,
                                         seed,
                                         int(params['init_satisfaction']*10),
                                         params['num_customers'],
                                         params['num_fraudsters'],
                                         params['end_date'].year)

        if os.path.exists(path+'.npy'):
            monetary_rewards = np.load(path+'.npy')
        else:

            # get the model for transactions
            model = TransactionModel(params, authenticator=authenticator)

            # run
            while not model.terminated:
                model.step()

            agent_vars = model.log_collector.get_agent_vars_dataframe()
            agent_vars.index = agent_vars.index.droplevel(1)
            monetary_rewards = rewards.monetary_reward_per_timestep(agent_vars)

            np.save(path, monetary_rewards)

        if sum_monetary_rewards is None:
            sum_monetary_rewards = monetary_rewards
        else:
            sum_monetary_rewards += monetary_rewards

        seed += 1

    sum_monetary_rewards /= (i+1)
    if k == 0:
        color = 'r'
    elif k == 1:
        color = 'r--'
    elif k == 2:
        color = 'b'
    elif k == 3:
        color = 'b--'
    plt.plot(range(len(monetary_rewards)), np.cumsum(monetary_rewards), color, label=auth_name)

plt.xlabel('time step')
plt.ylabel('monetary reward (cumulative)')
plt.legend()
plt.tight_layout()
plt.show()
