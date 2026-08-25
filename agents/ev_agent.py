from base_agent import Agent

class EVAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.ev_soc_pct = None
        self.ev_departure_min = None
        self.ev_plugged_in = None