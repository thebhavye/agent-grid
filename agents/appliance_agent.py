import math

from agents.base_agent import Agent

# Fallback draw for a generically constructed ApplianceAgent("x") with
# no roster row (e.g. legacy fake-dataset path). Matches the old
# DEFAULT_DEMANDS["appliance_1"] = 1.0 so old tests keep exercising
# the allocator.
DEFAULT_APPLIANCE_POWER_KW = 1.0


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _parse_timestamp_to_min(timestamp):
    """Parse 'HH:MM' -> minutes from midnight; None if unparseable."""
    if timestamp is None or not isinstance(timestamp, str):
        return None
    try:
        hh, mm = timestamp.split(":")
        return int(hh) * 60 + int(mm)
    except (ValueError, AttributeError):
        return None


class ApplianceAgent(Agent):
    def __init__(self, agent_id, appliance_id=None, appliance_priority=None,
                 appliance_deferable=False, appliance_power_kw=None,
                 appliance_usual_start_min=None, appliance_usual_end_min=None):
        super().__init__(agent_id)
        self.appliance_id = appliance_id if appliance_id is not None else agent_id
        self.appliance_priority = appliance_priority
        self.appliance_deferable = bool(appliance_deferable)
        self.appliance_power_kw = appliance_power_kw
        self.appliance_usual_start_min = appliance_usual_start_min
        self.appliance_usual_end_min = appliance_usual_end_min
        self._current_min = None

    def update_state(self, row):
        # Only dynamic input is the current time; static roster fields
        # were fixed at construction. Missing/unparseable timestamp ->
        # None, which get_demand_kw treats as in-window (fail-open) so
        # legacy fake rows still produce demand.
        self._current_min = _parse_timestamp_to_min(row.get("timestamp"))

    def _inside_usual_window(self):
        start = self.appliance_usual_start_min
        end = self.appliance_usual_end_min
        if _is_missing(start) or _is_missing(end):
            return True
        if self._current_min is None:
            return True
        start = float(start)
        end = float(end)
        cur = float(self._current_min)
        if start <= end:
            # Normal window, e.g. dishwasher 1200 -> 1320:
            # inside when start <= cur < end.
            return start <= cur < end
        # Overnight wrap, e.g. ev_charger 1140 -> 420:
        # inside when cur >= start OR cur < end.
        return cur >= start or cur < end

    def get_demand_kw(self):
        # Non-deferable (essentials): always full rated power.
        # Deferable: full rated power inside the usual window, else
        # strict 0.0 outside it. Threshold is binary (no partial
        # reduction) for explainability; NaN window or unknown time
        # fails open to full power.
        power = self.appliance_power_kw
        if _is_missing(power):
            power = DEFAULT_APPLIANCE_POWER_KW
        power = max(0.0, float(power))
        if not self.appliance_deferable:
            return power
        return power if self._inside_usual_window() else 0.0