from agents.base_agent import Agent
class BatteryAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.battery_soc_pct = None
        self.battery_health_pct = None