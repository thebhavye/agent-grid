from agents.base_agent import Agent

EV_TARGET_SOC_PCT = 100.0


class EVAgent(Agent):
    def __init__(self, agent_id, max_charge_kw):
        super().__init__(agent_id)
        # max_charge_kw is required (no default): it must come from the
        # appliances roster's ev_charger appliance_power_kw via
        # _build_agents(). A missing argument raises TypeError loudly
        # instead of silently falling back to a guessed constant.
        self.max_charge_kw = float(max_charge_kw)
        self.ev_soc_pct = None
        self.ev_departure_min = None
        self.ev_plugged_in = None

    def update_state(self, row):
        # Defaults mirror BatteryAgent: legacy fake rows without EV
        # columns behave as a plugged-in EV at 50% SOC.
        self.ev_soc_pct = row.get("ev_soc_pct", 50.0)
        self.ev_departure_min = row.get("ev_departure_min", 0)
        self.ev_plugged_in = row.get("ev_plugged_in", True)
        # Normalise numpy bools to plain Python bool.
        if self.ev_plugged_in is not None:
            self.ev_plugged_in = bool(self.ev_plugged_in)

    def get_demand_kw(self):
        # Unplugged -> 0 regardless of SOC. Plugged in -> linear
        # charge demand toward 100% (dataset has no per-row target
        # column, so 100% is the assumed target), scaled by the
        # roster-sourced ev_charger rating (self.max_charge_kw).
        if not self.ev_plugged_in:
            return 0.0
        if self.ev_soc_pct is None:
            return 0.0
        gap = max(0.0, EV_TARGET_SOC_PCT - float(self.ev_soc_pct))
        return self.max_charge_kw * (gap / 100.0)