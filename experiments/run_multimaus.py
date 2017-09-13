from authenticators.simple_authenticators import RandomAuthenticator, \
    HeuristicAuthenticator, OracleAuthenticator, NeverSecondAuthenticator, \
    AlwaysSecondAuthenticator
from simulator import parameters
from simulator.transaction_model import TransactionModel
from experiments import rewards
import numpy as np
import matplotlib.pyplot as plt
from experiments import result_handling


def run_single():

    # get the parameters for the simulation
    params = parameters.get_default_parameters()
    params['init_satisfaction'] = 0.9

    # increase the probability of making another transaction
    new_stay_prob = [0.8, 0.5]
    print('changing stay prob from', params['stay_prob'], 'to', new_stay_prob)
    params['stay_prob'] = new_stay_prob

    plt.figure(figsize=(10, 5))
    for a in ['random', 'oracle', 'never_second', 'heuristic', 'always_second']:

        # the authenticator
        authenticator = get_authenticator(a)

        # initialise transaction model
        model = TransactionModel(params, authenticator)

        # run the simulation until termination
        while not model.terminated:
            model.step()

        # get the collected data
        agent_vars = model.log_collector.get_agent_vars_dataframe()
        agent_vars.index = agent_vars.index.droplevel(1)

        model_vars = model.log_collector.get_model_vars_dataframe()

        # save the results
        result_handling.save_results(model)

        reward_fraud = rewards.money_lost_per_timestep(agent_vars)
        reward_genuine = rewards.money_made_per_timestep(agent_vars)
        monetary_rewards = rewards.monetary_reward_per_timestep(agent_vars)
        true_satisfactions = rewards.satisfaction_per_timestep(model_vars)

        plt.subplot(1, 4, 1)
        plt.ylabel('revenue (total)')
        plt.plot(range(len(monetary_rewards)), np.cumsum(monetary_rewards), label=a)
        plt.legend()

        plt.subplot(1, 4, 2)
        plt.ylabel('cumulative satisfaction')
        plt.plot(range(len(true_satisfactions)), np.cumsum(true_satisfactions), label=a)

        plt.subplot(1, 4, 3)
        plt.ylabel('revenue (money lost by fraud)')
        plt.plot(range(len(true_satisfactions)), np.cumsum(true_satisfactions), label=a)

        plt.subplot(1, 4, 4)
        plt.ylabel('revenue (money gained by genuine transactions)')
        plt.plot(range(len(true_satisfactions)), np.cumsum(true_satisfactions), label=a)

    plt.tight_layout()
    plt.show()


def get_authenticator(auth_type):
    if auth_type == 'random':
        return RandomAuthenticator()
    elif auth_type == 'heuristic':
        return HeuristicAuthenticator(50)
    elif auth_type == 'oracle':
        return OracleAuthenticator()
    elif auth_type == 'never_second':
        return NeverSecondAuthenticator()
    elif auth_type == 'always_second':
        return AlwaysSecondAuthenticator()


if __name__ == '__main__':

    run_single()
