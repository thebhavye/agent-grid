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
        self._current_min = None  # minutes from midnight parsed from timestamp

    def update_state(self, row):
        # Defaults: legacy fake rows without EV columns behave as a plugged-in EV at 50% SOC.
        # Note: urgency additionally requires ev_departure_min and timestamp; missing values yield 0.0 urgency.
        self.ev_soc_pct = row.get("ev_soc_pct", 50.0)
        self.ev_departure_min = row.get("ev_departure_min")
        self.ev_plugged_in = row.get("ev_plugged_in", True)
        
        if self.ev_plugged_in is not None:
            self.ev_plugged_in = bool(self.ev_plugged_in)
            
        ts = row.get("timestamp")
        self._current_min = None
        if isinstance(ts, str):
            try:
                hh, mm = ts.split(":", 1)
                self._current_min = int(hh) * 60 + int(mm)
            except (ValueError, TypeError):
                self._current_min = None

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

    def compute_urgency(self):
        if self.ev_soc_pct is None or not getattr(self, 'ev_plugged_in', False):
            return 0.0
            
        if self.ev_departure_min is None or self._current_min is None:
            return 0.0
            
        soc_gap = (100.0 - float(self.ev_soc_pct)) / 100.0
        dep_min = float(self.ev_departure_min)
        curr_min = float(self._current_min)
        
        # Wrapped overnight logic pulled from ApplianceAgent
        minutes_left = dep_min - curr_min if dep_min >= curr_min else (dep_min + 1440.0) - curr_min
        
        if minutes_left <= 0:
            return 1.0
            
        time_pressure = max(0.0, 1.0 - (minutes_left / 720.0))
        return max(0.0, min(1.0, 0.5 * soc_gap + 0.5 * time_pressure))
    