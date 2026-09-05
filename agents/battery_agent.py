from agents.base_agent import Agent

BATTERY_MAX_CHARGE_KW = 5.0


class BatteryAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.battery_soc_pct = None
        self.battery_health_pct = None

    def update_state(self, row):
        # Defaults (50%/100%) keep legacy fake rows with missing keys
        # producing sane non-zero demand so old tests still exercise
        # the proportional allocator.
        self.battery_soc_pct = row.get("battery_soc_pct", 50.0)
        self.battery_health_pct = row.get("battery_health_pct", 100.0)

    def get_demand_kw(self):
        # Linear top-up demand: full 5 kW at 0% SOC, 0 kW at 100%.
        # Independent assumption (option (a)): generate_dataset.py has
        # NO max charge/discharge rate concept -- battery SOC there is
        # driven purely by surplus_kw = solar_kw - load_kw against
        # capacity_kwh (rng.uniform(10, 14), internal only, not exposed
        # in the returned tables), with no inverter/rate cap. So this
        # 5 kW is our own inverter-limit choice from a different
        # physical model, not derived from the generator. Health is
        # tracked but does not scale demand yet (out of scope).
        if self.battery_soc_pct is None:
            return 0.0
        gap = max(0.0, 100.0 - float(self.battery_soc_pct))
        return BATTERY_MAX_CHARGE_KW * (gap / 100.0)