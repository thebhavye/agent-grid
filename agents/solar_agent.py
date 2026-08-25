from base_agent import Agent

class SolarAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.solar_gen_kw = None