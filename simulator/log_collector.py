from mesa.mesa.datacollection import DataCollector
from collections import defaultdict
import pandas as pd


class LogCollector(DataCollector):
    """ 
    Inherits from the DataCollector from the mesa framework,
    and overwrites some functions for our simulator
    """

    def collect(self, model):
        """ collect only logs from agents that make a transation"""
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(model))

        if self.agent_reporters:
            for var, reporter in self.agent_reporters.items():
                agent_records = []
                for agent in model.schedule.agents:
                    if agent.active:  # this is the line we changed
                        agent_records.append((agent.unique_id, reporter(agent)))
                self.agent_vars[var].append(agent_records)

    def get_agent_vars_dataframe(self):
        """ Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        This function was modified from the original implementation in mesa
        to return None if there are no entries at all

        (the df.index.names = ["Step", "AgentID"] line crashes with "ValueError:
        Length of new names must be 1, got 2" if there are no entries in original
        mesa implementation)

        """
        data = defaultdict(dict)
        found_entries = False

        for var, records in self.agent_vars.items():
            for step, entries in enumerate(records):
                for entry in entries:
                    agent_id = entry[0]
                    val = entry[1]
                    data[(step, agent_id)][var] = val
                    found_entries = True

        if not found_entries:
            return None

        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.names = ["Step", "AgentID"]
        return df
