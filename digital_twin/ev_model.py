"""Stateful scenario-based EV model.

Reuses conceptual patterns from generate_dataset.py but implemented as a
live stateful model. Not physics-heavy — scenario-based behavior.

The EV has:
- SOC (%)
- Departure time (minute of day)
- Plugged-in state (bool)

Behavior evolves based on current simulation time.
Deterministic when initialized with a fixed seed/scenario.

Scenarios (matching generate_dataset.py):
- overnight: plugged in at midnight, departs morning, plugs in evening
- midday: plugs in morning, departs afternoon
- evening_topup: plugs in evening, departs late evening

Charging: linear ramp to target SOC over plugged-in period (80% of duration).
"""

import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EVScenario(str, Enum):
    """EV charging scenarios matching generate_dataset.py."""
    OVERNIGHT = "overnight"
    MIDDAY = "midday"
    EVENING_TOPUP = "evening_topup"


@dataclass
class EVSegment:
    """A single plugged-in charging segment."""
    start_min: float      # Minute of day when plugging in
    end_min: float        # Minute of day when unplugging (departure)
    start_soc: float      # SOC at plug-in
    target_soc: float     # Target SOC at departure


@dataclass
class EVModelConfig:
    """Configuration for the EV model.

    Attributes:
        scenario: EVScenario or None for random.
        seed: Random seed for deterministic behavior. Default 42 so bare
              DigitalTwinConfig defaults are repeatable.
        battery_capacity_kwh: EV battery capacity (kWh). Default 60 kWh.
        max_charge_kw: Maximum charge power (kW). Default 7.0 kW.
        charge_efficiency: Charging efficiency (0-1). Default 0.90.
    """
    scenario: Optional[EVScenario] = None
    seed: Optional[int] = 42
    battery_capacity_kwh: float = 60.0
    max_charge_kw: float = 7.0
    charge_efficiency: float = 0.90

    def __post_init__(self):
        if self.battery_capacity_kwh <= 0:
            raise ValueError("battery_capacity_kwh must be positive")
        if self.max_charge_kw <= 0:
            raise ValueError("max_charge_kw must be positive")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")


class EVModel:
    """Stateful EV model with scenario-based charging behavior."""

    def __init__(self, config: Optional[EVModelConfig] = None):
        """
        Args:
            config: EVModelConfig instance. If None, uses defaults
                    (deterministic default seed 42).
        """
        self.config = config or EVModelConfig()
        self._rng = random.Random(self.config.seed)

        # Initialize scenario
        if self.config.scenario is None:
            self._scenario = self._rng.choice(list(EVScenario))
        else:
            self._scenario = self.config.scenario

        # Generate segments for this scenario
        self._segments = self._generate_segments()
        self._departure_min = self._calculate_departure_min()

        # State
        self._soc_pct = self._segments[0].start_soc if self._segments else 50.0
        self._plugged_in = False
        self._current_minute = 0

        # Track which segment we're in
        self._active_segment_idx = -1

    def _generate_segments(self) -> list[EVSegment]:
        """Generate charging segments based on scenario (matching generate_dataset.py logic)."""
        def snap(m):
            return float(round(m / 5) * 5)  # Snap to 5-min grid

        if self._scenario == EVScenario.OVERNIGHT:
            departure_min = snap(self._rng.uniform(6 * 60 + 30, 8 * 60 + 30))
            evening_plug_in_min = snap(self._rng.uniform(19 * 60, 21 * 60 + 30))
            return [
                # Tail of previous night's charge (00:00 -> morning departure)
                EVSegment(0.0, departure_min,
                          self._rng.uniform(70, 95), self._rng.uniform(90, 100)),
                # Evening plug-in after day's driving
                EVSegment(evening_plug_in_min, 1440.0,
                          self._rng.uniform(20, 45), self._rng.uniform(85, 100)),
            ]
        elif self._scenario == EVScenario.MIDDAY:
            plug_in_min = snap(self._rng.uniform(9 * 60, 10 * 60 + 30))
            departure_min = snap(self._rng.uniform(16 * 60, 17 * 60 + 30))
            return [
                EVSegment(plug_in_min, departure_min,
                          self._rng.uniform(35, 60), self._rng.uniform(80, 95)),
            ]
        else:  # EVENING_TOPUP
            plug_in_min = snap(self._rng.uniform(17 * 60 + 30, 19 * 60))
            departure_min = snap(self._rng.uniform(21 * 60, 22 * 60 + 30))
            return [
                EVSegment(plug_in_min, departure_min,
                          self._rng.uniform(50, 70), self._rng.uniform(75, 90)),
            ]

    def _calculate_departure_min(self) -> int:
        """Calculate the reported departure minute (first departure of the day)."""
        # For overnight, it's the morning departure
        # For others, it's the single departure
        if self._segments:
            return int(round(self._segments[0].end_min)) % 1440
        return 480  # Default 08:00

    def _find_active_segment(self, minute: float) -> Optional[int]:
        """Find the index of the segment active at the given minute."""
        for i, seg in enumerate(self._segments):
            if seg.start_min <= minute < seg.end_min:
                return i
        return None

    def _update_plugged_state(self, minute: float) -> None:
        """Update plugged-in state based on current minute."""
        active_idx = self._find_active_segment(minute)
        self._plugged_in = active_idx is not None
        self._active_segment_idx = active_idx if active_idx is not None else -1

    def _calculate_soc_at_minute(self, minute: float) -> float:
        """Calculate SOC at a specific minute (for initialization or jumps)."""
        # Find the segment that contains this minute, or the most recent one
        active_idx = self._find_active_segment(minute)

        if active_idx is not None:
            seg = self._segments[active_idx]
            # Linear charging over 80% of segment duration (matching generator)
            seg_duration = seg.end_min - seg.start_min
            charge_duration = max(seg_duration * 0.8, 1.0)
            elapsed = minute - seg.start_min
            progress = min(elapsed / charge_duration, 1.0)
            soc = seg.start_soc + (seg.target_soc - seg.start_soc) * progress
            return max(0.0, min(100.0, soc))

        # Not in any segment - find nearest known SOC
        # Forward fill from last segment, backfill from first
        last_known_soc = None
        for seg in self._segments:
            if seg.end_min <= minute:
                last_known_soc = seg.target_soc
            elif seg.start_min > minute and last_known_soc is None:
                last_known_soc = seg.start_soc
                break

        if last_known_soc is not None:
            return max(0.0, min(100.0, last_known_soc))

        return 50.0  # Fallback

    def initialize_at(self, start_datetime: datetime) -> None:
        """Initialize EV state at a specific simulation start time.

        Args:
            start_datetime: Simulation start datetime.
        """
        self._current_minute = start_datetime.hour * 60 + start_datetime.minute
        self._update_plugged_state(self._current_minute)
        self._soc_pct = self._calculate_soc_at_minute(self._current_minute)

    def _advance_schedule(self, time_delta_min: float) -> None:
        """Advance the exogenous clock and resolve plug-in transitions."""
        self._current_minute = (self._current_minute + time_delta_min) % 1440
        prev_plugged = self._plugged_in
        self._update_plugged_state(self._current_minute)
        if self._plugged_in and not prev_plugged:
            # The EV was away between segments.  Its driving is represented
            # by the next segment's start SOC, not by autonomous charging.
            self._soc_pct = self._segments[self._active_segment_idx].start_soc

    def _apply_autonomous_charge(self, time_delta_min: float) -> dict:
        """Apply the legacy ramp/noise behavior after the schedule advances."""
        charging_kw = 0.0

        if self._plugged_in and self._active_segment_idx >= 0:
            seg = self._segments[self._active_segment_idx]
            seg_duration = seg.end_min - seg.start_min
            charge_duration = max(seg_duration * 0.8, 1.0)
            elapsed = self._current_minute - seg.start_min

            if elapsed < charge_duration:
                progress = elapsed / charge_duration
                target_soc = seg.start_soc + (seg.target_soc - seg.start_soc) * progress
                soc_delta = target_soc - self._soc_pct
                energy_needed_kwh = (soc_delta / 100.0) * self.config.battery_capacity_kwh
                time_delta_h = time_delta_min / 60.0
                required_power_kw = energy_needed_kwh / time_delta_h / self.config.charge_efficiency
                charging_kw = max(0.0, min(self.config.max_charge_kw, required_power_kw))
                energy_stored_kwh = charging_kw * time_delta_h * self.config.charge_efficiency
                self._soc_pct += (energy_stored_kwh / self.config.battery_capacity_kwh) * 100.0
            else:
                self._soc_pct = seg.target_soc

        self._soc_pct = max(0.0, min(100.0, self._soc_pct))
        if self._plugged_in and charging_kw > 0:
            noise = self._rng.normalvariate(0, 0.3)
            self._soc_pct = max(0.0, min(100.0, self._soc_pct + noise))

        return {
            "soc_pct": self._soc_pct,
            "departure_min": self._departure_min,
            "plugged_in": self._plugged_in,
            "charging_kw": charging_kw,
        }

    def observe(self, time_delta_min: float = 5.0) -> dict:
        """Advance the EV schedule without applying charging power."""
        self._advance_schedule(time_delta_min)
        return {
            "soc_pct": self._soc_pct,
            "departure_min": self._departure_min,
            "plugged_in": self._plugged_in,
            "charging_kw": 0.0,
        }

    def commit_delivered(self, time_delta_min: float, delivered_kw: float) -> dict:
        """Apply caller-delivered EV charging power for the observed interval."""
        if delivered_kw < 0:
            raise ValueError("EV delivered power must be non-negative")
        if delivered_kw > 0 and not self._plugged_in:
            raise ValueError("cannot deliver positive EV power while unplugged")

        requested_kw = min(self.config.max_charge_kw, delivered_kw)
        time_delta_h = time_delta_min / 60.0
        if not self._plugged_in:
            requested_kw = 0.0

        headroom_kwh = (100.0 - self._soc_pct) / 100.0 * self.config.battery_capacity_kwh
        stored_energy_kwh = min(
            headroom_kwh,
            requested_kw * time_delta_h * self.config.charge_efficiency,
        )
        self._soc_pct += (stored_energy_kwh / self.config.battery_capacity_kwh) * 100.0
        actual_kw = (
            stored_energy_kwh / time_delta_h / self.config.charge_efficiency
            if time_delta_h > 0 and self.config.charge_efficiency > 0
            else 0.0
        )
        return {
            "soc_pct": max(0.0, min(100.0, self._soc_pct)),
            "departure_min": self._departure_min,
            "plugged_in": self._plugged_in,
            "charging_kw": actual_kw,
        }

    def commit_autonomous(self, time_delta_min: float) -> dict:
        """Apply the legacy autonomous charging ramp after ``observe()``."""
        return self._apply_autonomous_charge(time_delta_min)

    def tick(self, time_delta_min: float = 5.0) -> dict:
        """Advance simulation by one timestep.

        Args:
            time_delta_min: Minutes to advance (default 5).

        Returns:
            Dict with current state:
                - "soc_pct": Current SOC (%)
                - "departure_min": Departure minute of day (int)
                - "plugged_in": Whether EV is plugged in (bool)
                - "charging_kw": Current charging power (kW), 0 if not charging
        """
        self._advance_schedule(time_delta_min)
        return self._apply_autonomous_charge(time_delta_min)

    @property
    def soc_pct(self) -> float:
        return self._soc_pct

    @property
    def departure_min(self) -> int:
        return self._departure_min

    @property
    def plugged_in(self) -> bool:
        return self._plugged_in

    @property
    def scenario(self) -> EVScenario:
        return self._scenario

    @property
    def current_minute(self) -> float:
        return self._current_minute

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset EV to its initial state.

        With no argument the original config seed is reused, so the reset
        model regenerates identical segments and replays identically to a
        freshly constructed one (issue #23). Pass a seed to re-seed with
        a different stream instead. (A model originally built with
        seed=None cannot replay deterministically.)
        """
        # Use provided seed, or fall back to original config seed for exact replay
        effective_seed = seed if seed is not None else self.config.seed
        self._rng = random.Random(effective_seed)
        self.__init__(EVModelConfig(
            scenario=self.config.scenario,
            seed=effective_seed,
            battery_capacity_kwh=self.config.battery_capacity_kwh,
            max_charge_kw=self.config.max_charge_kw,
            charge_efficiency=self.config.charge_efficiency,
        ))
