class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def update_state(self, row):
        raise NotImplementedError("Subclass must implement this")

    def compute_urgency(self):
        raise NotImplementedError("Subclass must implement this")