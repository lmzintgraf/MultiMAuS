"""
Wrapper for an environment
"""
import state_space


class Environment:
    def __init__(self, agent):

        # the RL agent we use to ask for authentications
        self.agent = agent

        self.prev_state = None

    def authorise_transaction(self, customer):

        # get the current state we will show to the agent
        new_state = state_space.get_state(customer)

        # call the step function of the model
        action = self.agent.take_action(new_state)

        # ask the user for authentication
        auth_result = 1
        if action:
            auth_result = customer.give_authentication()

        # calculate the reward
        reward = 0
        if auth_result is not None:
            reward += customer.fraudster * (-customer.curr_amount)
            reward += (1-customer.fraudster) * (0.003 * customer.curr_amount + 0.001)

        # update agent
        if self.prev_state is not None:
            self.agent.update(self.prev_state, action, reward, new_state)

        self.prev_state = new_state
