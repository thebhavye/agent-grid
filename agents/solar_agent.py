from agents.base_agent import Agent


class SolarAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.solar_gen_kw = None

    def update_state(self, row):
        # row is a timeseries dict; default 0.0 so legacy fake rows
        # without full columns still work.
        self.solar_gen_kw = row.get("solar_gen_kw", 0.0)

    def get_demand_kw(self):
        # Solar supplies power (run_simulation uses solar_gen_kw as
        # available_power); it never consumes, so demand is always 0.
        return 0.0
    def compute_urgency(self):
        return 0.0
    