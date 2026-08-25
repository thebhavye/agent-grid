from base_agent import Agent

class ApplianceAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.appliance_id = None
        self.appliance_priority = None
        self.appliance_deferable = None
        self.appliance_power_kw = None