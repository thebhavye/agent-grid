class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def update_state(self, row):
        raise NotImplementedError("Subclass must implement this")

    def compute_urgency(self):
        raise NotImplementedError("Subclass must implement this")

    def get_bid(self):
        # Phase 2 prep: one generic packaging of the agent's current
        # demand + urgency for the future bidding/arbiter loop. No
        # timestamp/context here -- timestep ownership stays with the
        # caller, which already holds the row.
        return {
            "agent_id": self.agent_id,
            "demand_kw": self.get_demand_kw(),
            "urgency": self.compute_urgency(),
        }