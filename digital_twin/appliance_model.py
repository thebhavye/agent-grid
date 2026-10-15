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

import hashlib
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

    def __init__(
        self,
        config: Optional[ApplianceModelConfig] = None,
        closed_loop: bool = False,
    ):
        """
        Args:
            config: ApplianceModelConfig instance. If None, uses defaults.
        """
        self.config = config or ApplianceModelConfig()
        self._closed_loop = bool(closed_loop)
        self._rng = random.Random(self.config.seed)

        # Closed-loop runs use one stable stream per appliance.  The legacy
        # mode intentionally keeps the original shared RNG untouched so the
        # pre-Phase-2 tick output remains byte-for-byte compatible.
        self._appliance_rngs: dict[str, random.Random] = {}
        self._rng_draw_count = 0
        self._cycle_start_times: dict[str, list[float]] = {}
        self._drawn_cycle_start_times: dict[str, list[float]] = {}
        self._blocked_starts: dict[str, int] = {}
        if self._closed_loop:
            for spec in self.config.appliances:
                self._cycle_start_times[spec.appliance_id] = []
                self._drawn_cycle_start_times[spec.appliance_id] = []
                self._blocked_starts[spec.appliance_id] = 0
                seed_material = f"{self.config.seed!r}:{spec.appliance_id}".encode()
                stable_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:8], "big"
                )
                self._appliance_rngs[spec.appliance_id] = random.Random(stable_seed)

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
        if not spec.deferable:
            return  # Essentials stay on; their zero runtime is not an expiring cycle.
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
                # For deferable, randomly decide if running at start time.
                # Closed-loop mode consumes a fixed pair of draws per
                # appliance, regardless of whether the start is usable.
                in_window = self._in_usual_window(spec, self._current_minute)
                if self._closed_loop:
                    start_roll = self._draw_random(spec.appliance_id)
                    runtime = self._draw_uniform(
                        spec.appliance_id, spec.min_runtime_min, spec.max_runtime_min
                    )
                else:
                    runtime = None
                    start_roll = self._rng.random() if in_window else 1.0

                if in_window and start_roll < 0.3:
                    if runtime is None:
                        runtime = self._rng.uniform(
                            spec.min_runtime_min, spec.max_runtime_min
                        )
                    state["is_on"] = True
                    state["time_remaining_min"] = runtime
                    if self._closed_loop:
                        self._cycle_start_times[spec.appliance_id].append(
                            self._current_minute
                        )
                        self._drawn_cycle_start_times[spec.appliance_id].append(
                            self._current_minute
                        )
                else:
                    state["is_on"] = False
                    state["time_remaining_min"] = 0.0

    def _draw_random(self, appliance_id: str) -> float:
        """Draw from an appliance-local RNG and count the draw."""
        self._rng_draw_count += 1
        return self._appliance_rngs[appliance_id].random()

    def _draw_uniform(self, appliance_id: str, low: float, high: float) -> float:
        """Draw a runtime from an appliance-local RNG and count the draw."""
        self._rng_draw_count += 1
        return self._appliance_rngs[appliance_id].uniform(low, high)

    def _observe_closed_loop(self, time_delta_min: float) -> dict:
        """Advance schedule decisions without consuming cycle runtime."""
        self._current_minute = (self._current_minute + time_delta_min) % 1440

        for spec in self.config.appliances:
            if not spec.deferable:
                continue

            state = self._appliance_states[spec.appliance_id]
            start_roll = self._draw_random(spec.appliance_id)
            runtime = self._draw_uniform(
                spec.appliance_id, spec.min_runtime_min, spec.max_runtime_min
            )
            if not self._in_usual_window(spec, self._current_minute):
                continue

            prob_per_step = spec.cycle_prob_per_hour * (time_delta_min / 60.0)
            if start_roll >= prob_per_step:
                continue

            self._drawn_cycle_start_times[spec.appliance_id].append(
                self._current_minute
            )
            if state["is_on"] or state["time_remaining_min"] > 0:
                self._blocked_starts[spec.appliance_id] += 1
                continue

            state["is_on"] = True
            state["time_remaining_min"] = runtime
            self._cycle_start_times[spec.appliance_id].append(self._current_minute)

        return self._current_result()

    def _current_result(self) -> dict:
        """Return desired loads for the current schedule state."""
        total_load = self.config.base_load_kw
        appliance_loads = {}
        desired = {}
        active = []
        for spec in self.config.appliances:
            state = self._appliance_states[spec.appliance_id]
            load = spec.power_kw if state["is_on"] else 0.0
            appliance_loads[spec.appliance_id] = load
            desired[spec.appliance_id] = load
            if spec.appliance_id not in EXCLUDED_FROM_HOUSEHOLD_TOTAL:
                total_load += load
            if state["is_on"]:
                active.append(spec.appliance_id)
        return {
            "total_load_kw": total_load,
            "appliance_loads": appliance_loads,
            "active_appliances": active,
            "appliance_desired_kw": desired,
        }

    def observe(self, time_delta_min: float = 5.0) -> dict:
        """Advance schedule state and return desired loads without progress."""
        if not self._closed_loop:
            raise RuntimeError("observe() requires closed-loop appliance mode")
        return self._observe_closed_loop(time_delta_min)

    def commit(self, time_delta_min: float, delivered: Optional[dict] = None) -> dict:
        """Apply delivered appliance power and proportionally complete cycles."""
        if not self._closed_loop:
            raise RuntimeError("commit() requires closed-loop appliance mode")

        autonomous = delivered is None
        delivered = delivered or {}
        current = self._current_result()
        served = {}
        for spec in self.config.appliances:
            desired = current["appliance_loads"][spec.appliance_id]
            if spec.appliance_id == "ev_charger":
                requested = (
                    desired
                    if autonomous
                    else delivered.get(
                        "appliance_ev_charger", delivered.get("ev_1", 0.0)
                    )
                )
            else:
                requested = (
                    desired
                    if autonomous
                    else delivered.get(f"appliance_{spec.appliance_id}", 0.0)
                )
            served_kw = min(max(0.0, float(requested)), desired)
            served[spec.appliance_id] = served_kw

            state = self._appliance_states[spec.appliance_id]
            if spec.deferable and state["is_on"] and desired > 0.0:
                progress = time_delta_min * served_kw / desired
                state["time_remaining_min"] -= progress
                if state["time_remaining_min"] <= 0.0:
                    state["is_on"] = False
                    state["time_remaining_min"] = 0.0

        result = self._current_result()
        result["appliance_desired_kw"] = dict(current["appliance_desired_kw"])
        result["appliance_served_kw"] = served
        return result

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
    def rng_draw_count(self) -> int:
        """Number of closed-loop schedule RNG draws made so far."""
        return self._rng_draw_count

    @property
    def cycle_start_times(self) -> dict[str, list[float]]:
        """Actual cycle starts, copied so callers cannot mutate state."""
        return {key: list(value) for key, value in self._cycle_start_times.items()}

    @property
    def drawn_cycle_start_times(self) -> dict[str, list[float]]:
        """Successful start draws, including starts blocked by a running cycle."""
        return {
            key: list(value) for key, value in self._drawn_cycle_start_times.items()
        }

    @property
    def blocked_starts(self) -> dict[str, int]:
        """Count of successful start draws skipped while an appliance was busy."""
        return dict(self._blocked_starts)

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
        """Reset to initial state.

        With no argument the original config seed is reused, so the reset
        model replays identically to a freshly constructed one (issue #23).
        Pass a seed to re-seed with a different stream instead. (A model
        originally built with seed=None cannot replay deterministically.)
        """
        # Use provided seed, or fall back to original config seed for exact replay
        effective_seed = seed if seed is not None else self.config.seed
        self._rng = random.Random(effective_seed)
        self.__init__(ApplianceModelConfig(
            seed=effective_seed,
            appliances=self.config.appliances,
            base_load_kw=self.config.base_load_kw,
        ), closed_loop=self._closed_loop)
