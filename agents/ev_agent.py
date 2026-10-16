from agents.base_agent import Agent

import math

EV_TARGET_SOC_PCT = 100.0


def _require_finite(value, name):
    """Coerce to float, rejecting NaN/Inf loudly (issue #26)."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_supplied_finite(value, name):
    """Validate a supplied (non-None) value is finite.

    None/missing values are allowed and handled elsewhere; this only
    rejects explicitly provided NaN/Inf.
    """
    if value is not None:
        _require_finite(value, name)


def _validate_preference(value, name):
    if name == "ev_ready_by_min" and value is None:
        return None
    try:
        result = _require_finite(value, name)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    valid = (0 < result <= 100 if name == "ev_target_soc_pct"
             else 0 <= result < 1440)
    if not valid:
        bounds = "(0, 100]" if name == "ev_target_soc_pct" else "[0, 1440)"
        raise ValueError(f"{name} must be in {bounds}")
    return result


class EVAgent(Agent):
    """Bid toward a target SOC by an optional minute-of-day deadline.

    Row preferences override constructor defaults for one update only.
    An explicit ev_ready_by_min=None selects the reported departure.
    """

    def __init__(self, agent_id, max_charge_kw,
                 target_soc_pct=EV_TARGET_SOC_PCT, ready_by_min=None):
        super().__init__(agent_id)
        self.max_charge_kw = float(max_charge_kw)
        self.target_soc_pct = _validate_preference(target_soc_pct, "ev_target_soc_pct")
        self.ready_by_min = _validate_preference(ready_by_min, "ev_ready_by_min")
        self.ev_target_soc_pct = self.target_soc_pct
        self.ev_ready_by_min = self.ready_by_min
        self.ev_soc_pct = None
        self.ev_departure_min = None
        self.ev_plugged_in = None
        self._current_min = None  # minutes from midnight parsed from timestamp

    def update_state(self, row):
        # Resolve each row against constructor defaults, never the previous row.
        target = _validate_preference(
            row.get("ev_target_soc_pct", self.target_soc_pct), "ev_target_soc_pct")
        ready_by = _validate_preference(
            row.get("ev_ready_by_min", self.ready_by_min), "ev_ready_by_min")
        self.ev_target_soc_pct = target
        self.ev_ready_by_min = ready_by
        # Defaults: legacy fake rows without EV columns behave as a plugged-in EV at 50% SOC.
        # Note: urgency additionally requires ev_departure_min and timestamp; missing values yield 0.0 urgency.
        self.ev_soc_pct = row.get("ev_soc_pct", 50.0)
        self.ev_departure_min = row.get("ev_departure_min")
        self.ev_plugged_in = row.get("ev_plugged_in", True)
        
        if self.ev_plugged_in is not None:
            self.ev_plugged_in = bool(self.ev_plugged_in)
            
        ts = row.get("timestamp")
        self._current_min = None

        # Validate numeric timestamps before discarding (issue #26)
        if isinstance(ts, (int, float)):
            _validate_supplied_finite(ts, "timestamp")
        elif isinstance(ts, str):
            try:
                hh, mm = ts.split(":", 1)
                self._current_min = int(hh) * 60 + int(mm)
            except (ValueError, TypeError):
                self._current_min = None
        # Other types (None, etc.) leave _current_min as None

    def get_demand_kw(self):
        # Validate supplied SOC before early returns (issue #26)
        _validate_supplied_finite(self.ev_soc_pct, "ev_soc_pct")

        if not self.ev_plugged_in:
            return 0.0
        if self.ev_soc_pct is None:
            return 0.0
        gap = max(0.0, self.ev_target_soc_pct - _require_finite(self.ev_soc_pct, "ev_soc_pct"))
        return self.max_charge_kw * (gap / 100.0)

    def compute_urgency(self):
        # Validate supplied non-None values BEFORE early returns (issue #26)
        _validate_supplied_finite(self.ev_soc_pct, "ev_soc_pct")
        _validate_supplied_finite(self.ev_departure_min, "ev_departure_min")
        _validate_supplied_finite(self._current_min, "timestamp")

        if self.ev_soc_pct is None or not getattr(self, 'ev_plugged_in', False):
            return 0.0
            
        deadline = (self.ev_ready_by_min if self.ev_ready_by_min is not None
                    else self.ev_departure_min)
        if deadline is None or self._current_min is None:
            return 0.0
            
        soc_gap = max(
            0.0, self.ev_target_soc_pct - _require_finite(self.ev_soc_pct, "ev_soc_pct")
        ) / 100.0
        if soc_gap <= 0.0:
            return 0.0
            
        dep_min = _require_finite(deadline, "EV deadline")
        curr_min = _require_finite(self._current_min, "timestamp")
        
        # Wrapped overnight logic pulled from ApplianceAgent
        minutes_left = dep_min - curr_min if dep_min >= curr_min else (dep_min + 1440.0) - curr_min
        
        if minutes_left <= 0:
            return 1.0
            
        time_pressure = max(0.0, 1.0 - (minutes_left / 720.0))
        return max(0.0, min(1.0, 0.5 * soc_gap + 0.5 * time_pressure))
