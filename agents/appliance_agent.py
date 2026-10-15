import math
from agents.base_agent import Agent

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
        self.missed_windows = []

    def update_state(self, row):
        # New-timestep boundary: missed_windows is a CURRENT-TIMESTEP event
        # buffer (see Issue #10), not cumulative history. update_state()
        # clears the previous timestep's events whenever the parsed timestep
        # changes, so a repeated update_state() for the same timestep does
        # not drop an event the consumer has not read yet.
        new_min = _parse_timestamp_to_min(row.get("timestamp"))
        if new_min != self._current_min:
            self.missed_windows = []
        self._current_min = new_min

    def get_missed_windows(self):
        """Return current-timestep missed-window events (defensive copy).

        Minimal consumer accessor for the Phase 2 arbiter contract: read
        the current step's events before the next update_state() clears
        them. Returns a new list of new dicts so callers cannot mutate
        either the buffer or the buffered event dicts.
        """
        return [dict(e) for e in self.missed_windows]

    def _inside_usual_window(self):
        start = self.appliance_usual_start_min
        end = self.appliance_usual_end_min
        cur_time = self._current_min
        
        if start is None or end is None or _is_missing(start) or _is_missing(end):
            return True
        if cur_time is None:
            return True
            
        start = float(start)  # type: ignore
        end = float(end)  # type: ignore
        cur = float(cur_time)  # type: ignore
        
        if start <= end:
            return start <= cur < end
        return cur >= start or cur < end

    def get_demand_kw(self):
        power = self.appliance_power_kw
        if power is None or _is_missing(power):
            power = DEFAULT_APPLIANCE_POWER_KW

        # Reject non-finite values BEFORE clamping (issue #26)
        power_float = float(power)
        if not math.isfinite(power_float):
            raise ValueError(
                f"appliance_power_kw must be finite, got {self.appliance_power_kw!r}"
            )
        # Only clamp finite negative values
        power = max(0.0, power_float)
        
        if not getattr(self, 'appliance_deferable', False):
            return power
        return power if self._inside_usual_window() else 0.0

    def compute_urgency(self):
        if not getattr(self, 'appliance_deferable', False):
            return 0.0
            
        end_val = self.appliance_usual_end_min
        curr_val = self._current_min
        
        if end_val is None or curr_val is None or _is_missing(end_val):
            return 0.0
            
        if not self._inside_usual_window():
            # Current-step event only, deduplicated: repeated
            # compute_urgency() calls for the same timestep must not append
            # the same event twice. Buffer holds at most one event per
            # appliance per timestep and is cleared by update_state().
            if not any(
                e.get("timestamp") == self._current_min
                and e.get("appliance_id") == self.appliance_id
                for e in self.missed_windows
            ):
                self.missed_windows.append({
                    "timestamp": self._current_min,
                    "appliance_id": self.appliance_id,
                })
            return 0.0
            
        end_min = float(end_val)
        curr_min = float(curr_val)
        
        time_left = end_min - curr_min if end_min >= curr_min else (end_min + 1440.0) - curr_min
        URGENCY_WINDOW_MIN = 240.0
        urgency = 1.0 - (time_left / URGENCY_WINDOW_MIN)
        return max(0.0, min(1.0, urgency))