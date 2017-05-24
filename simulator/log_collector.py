from mesa.datacollection import DataCollector


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
