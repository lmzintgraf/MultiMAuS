from mesa import Agent
import random


class Customer(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, customer_id, transaction_model):
        super().__init__(customer_id, transaction_model)
        self.awaiting_transaction_feedback = False
        self.wealth = 1

    def step(self):

        if not self.awaiting_transaction_feedback:
            self.give_money()
            self.awaiting_transaction_feedback = True
        else:
            self.move()
            self.awaiting_transaction_feedback = False

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

