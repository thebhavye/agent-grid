"""Lightweight live household-load/appliance model.

Scenario/state-based behavior rather than detailed electrical physics.
Produces reasonable household load for the current timestep.

Reuses conceptual appliance roster/patterns from generate_dataset.py but
keeps implementation independent. Stateful enough that consecutive ticks
produce reasonable time-dependent behavior.

Appliance categories (matching generator):
- Essential (always on): fridge, lighting, security_system
- Flexible (deferable, with usual windows): ev_charger, dishwasher,
  washing_machine, water_heater, pool_pump

The model simulates each appliance's on/off state based on time-of-day
and usual windows, with some stochastic variation.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# Appliance IDs excluded from household_load_kw totals. The EV charger's
# power is tracked separately by EVModel, and generate_dataset.py's
# household_load_kw has no EV term — counting it here double-counts
# (PR #4 double-counting fix). The charger state is still simulated and
# reported in appliance_loads/active_appliances for visibility.
EXCLUDED_FROM_HOUSEHOLD_TOTAL = frozenset({"ev_charger"})


@dataclass
class ApplianceSpec:
    """Specification for a single appliance."""
    appliance_id: str
    priority: int           # 1 = critical, 5 = deferable
    deferable: bool
    power_kw: float         # Power draw when active
    usual_start_min: Optional[int] = None  # Usual window start (min of day)
    usual_end_min: Optional[int] = None    # Usual window end (min of day)
    # Runtime behavior
    min_runtime_min: float = 30.0     # Minimum run duration once started
    max_runtime_min: float = 120.0    # Maximum run duration
    cycle_prob_per_hour: float = 0.5  # Probability of starting per hour in window


@dataclass
class ApplianceModelConfig:
    """Configuration for the appliance model.

    Attributes:
        seed: Random seed for deterministic behavior. Default 42 so bare
              DigitalTwinConfig defaults are repeatable.
        appliances: List of ApplianceSpec. If None, uses default roster
                    matching generate_dataset.py.
        base_load_kw: Base always-on load (kW). Default 0.6 kW (matches generator).
    """
    seed: Optional[int] = 42
    appliances: Optional[list[ApplianceSpec]] = None
    base_load_kw: float = 0.6

    def __post_init__(self):
        if self.appliances is None:
            self.appliances = self._default_appliances()

    @staticmethod
    def _default_appliances() -> list[ApplianceSpec]:
        """Default appliance roster matching generate_dataset.py ranges."""
        return [
            # Essential - always on, no usual window
            ApplianceSpec("fridge", 1, False, 0.15),
            ApplianceSpec("lighting", 2, False, 0.25),
            ApplianceSpec("security_system", 1, False, 0.05),
            # Flexible - with usual windows
            ApplianceSpec("ev_charger", 3, True, 5.0, 19 * 60, 7 * 60),  # Overnight
            ApplianceSpec("dishwasher", 4, True, 1.2, 20 * 60, 22 * 60),
            ApplianceSpec("washing_machine", 4, True, 1.0, 9 * 60, 17 * 60),
            ApplianceSpec("water_heater", 3, True, 2.0, 6 * 60, 8 * 60),
            ApplianceSpec("pool_pump", 5, True, 1.0, 11 * 60, 15 * 60),
        ]


class ApplianceModel:
    """Stateful household appliance/load model."""

    def __init__(self, config: Optional[ApplianceModelConfig] = None):
        """
        Args:
            config: ApplianceModelConfig instance. If None, uses defaults.
        """
        self.config = config or ApplianceModelConfig()
        self._rng = random.Random(self.config.seed)

        # Per-appliance state
        self._appliance_states: dict[str, dict] = {}
        for spec in self.config.appliances:
            self._appliance_states[spec.appliance_id] = {
                "is_on": not spec.deferable,  # Essentials start on
                "time_remaining_min": 0.0,
                "next_cycle_start": None,
            }

        self._current_minute = 0

    def _in_usual_window(self, spec: ApplianceSpec, minute: float) -> bool:
        """Check if current minute is in appliance's usual window."""
        if not spec.deferable or spec.usual_start_min is None or spec.usual_end_min is None:
            return True  # Essentials always "in window"

        start = spec.usual_start_min
        end = spec.usual_end_min
        cur = minute

        if start <= end:
            return start <= cur < end
        else:  # Overnight window (e.g., 19:00 to 07:00)
            return cur >= start or cur < end

    def _maybe_start_cycle(self, spec: ApplianceSpec, state: dict, minute: float,
                             time_delta_min: float = 5.0) -> None:
        """Randomly decide whether to start a new cycle."""
        if state["is_on"] or state["time_remaining_min"] > 0:
            return

        if not self._in_usual_window(spec, minute):
            return

        # Probability per step scales with step size (base rate is per hour).
        prob_per_step = spec.cycle_prob_per_hour * (time_delta_min / 60.0)
        if self._rng.random() < prob_per_step:
            # Start a cycle
            runtime = self._rng.uniform(spec.min_runtime_min, spec.max_runtime_min)
            state["is_on"] = True
            state["time_remaining_min"] = runtime

    def _maybe_end_cycle(self, spec: ApplianceSpec, state: dict, minute: float,
                         time_delta_min: float = 5.0) -> None:
        """Decrement runtime and possibly end cycle."""
        if not state["is_on"]:
            return

        state["time_remaining_min"] -= time_delta_min
        if state["time_remaining_min"] <= 0:
            state["is_on"] = False
            state["time_remaining_min"] = 0.0

    def initialize_at(self, start_datetime: datetime) -> None:
        """Initialize appliance states at a specific simulation start time.

        Args:
            start_datetime: Simulation start datetime.
        """
        self._current_minute = start_datetime.hour * 60 + start_datetime.minute

        # Initialize each appliance to a reasonable state for the start time
        for spec in self.config.appliances:
            state = self._appliance_states[spec.appliance_id]

            if not spec.deferable:
                # Essentials always on
                state["is_on"] = True
                state["time_remaining_min"] = 0.0
            else:
                # For deferable, randomly decide if running at start time
                in_window = self._in_usual_window(spec, self._current_minute)
                if in_window and self._rng.random() < 0.3:
                    state["is_on"] = True
                    state["time_remaining_min"] = self._rng.uniform(
                        spec.min_runtime_min, spec.max_runtime_min
                    )
                else:
                    state["is_on"] = False
                    state["time_remaining_min"] = 0.0

    def tick(self, time_delta_min: float = 5.0) -> dict:
        """Advance simulation by one timestep.

        Args:
            time_delta_min: Minutes to advance (default 5).

        Returns:
            Dict with:
                - "total_load_kw": Total household load (kW, excluding
                  EXCLUDED_FROM_HOUSEHOLD_TOTAL — the EV charger is
                  modeled separately by EVModel)
                - "appliance_loads": Dict of appliance_id -> load_kw
                  (includes ev_charger state for visibility)
                - "active_appliances": List of currently active appliance_ids
        """
        self._current_minute = (self._current_minute + time_delta_min) % 1440

        total_load = self.config.base_load_kw
        appliance_loads = {}
        active = []

        for spec in self.config.appliances:
            state = self._appliance_states[spec.appliance_id]

            # Update cycle state
            self._maybe_end_cycle(spec, state, self._current_minute, time_delta_min)
            self._maybe_start_cycle(spec, state, self._current_minute, time_delta_min)

            # Calculate load (ev_charger excluded from the household total:
            # the EV is modeled separately by EVModel)
            load = spec.power_kw if state["is_on"] else 0.0
            appliance_loads[spec.appliance_id] = load
            if spec.appliance_id not in EXCLUDED_FROM_HOUSEHOLD_TOTAL:
                total_load += load

            if state["is_on"]:
                active.append(spec.appliance_id)

        return {
            "total_load_kw": total_load,
            "appliance_loads": appliance_loads,
            "active_appliances": active,
        }

    @property
    def current_minute(self) -> float:
        return self._current_minute

    @property
    def total_load_kw(self) -> float:
        """Current total load without advancing time (excludes ev_charger)."""
        total = self.config.base_load_kw
        for spec in self.config.appliances:
            if spec.appliance_id in EXCLUDED_FROM_HOUSEHOLD_TOTAL:
                continue
            state = self._appliance_states[spec.appliance_id]
            if state["is_on"]:
                total += spec.power_kw
        return total

    def get_appliance_state(self, appliance_id: str) -> Optional[dict]:
        """Get current state of a specific appliance."""
        return self._appliance_states.get(appliance_id)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset to initial state."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.__init__(ApplianceModelConfig(
            seed=seed,
            appliances=self.config.appliances,
            base_load_kw=self.config.base_load_kw,
        ))