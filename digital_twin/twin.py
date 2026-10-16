"""Digital Twin orchestrator.

Coordinates weather, solar, battery, EV, and appliance models to produce
a single timeseries row per tick matching the existing dataset schema.

The DigitalTwin.tick() advances simulation by one timestep and returns
a dict with the same keys as data/generate_dataset.py timeseries rows.

Usage:
    twin = DigitalTwin(DigitalTwinConfig(start_datetime=datetime(2024, 1, 15, 0, 0)))
    for _ in range(288):  # 24 hours at 5-min resolution
        row = twin.tick(5)
        # row has: solar_gen_kw, battery_soc_pct, battery_health_pct,
        #          household_load_kw, ev_soc_pct, ev_departure_min, ev_plugged_in
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from digital_twin.weather_client import WeatherClient
from digital_twin.solar_model import SolarModel, SolarModelConfig
from digital_twin.battery_model import BatteryModel, BatteryModelConfig
from digital_twin.ev_model import EVModel, EVModelConfig
from digital_twin.appliance_model import ApplianceModel, ApplianceModelConfig


@dataclass
class DigitalTwinConfig:
    """Configuration for the DigitalTwin.

    All sub-configs are optional; defaults will be used if not provided.
    """
    start_datetime: datetime
    weather_client: Optional[WeatherClient] = None
    solar_config: Optional[SolarModelConfig] = None
    battery_config: Optional[BatteryModelConfig] = None
    ev_config: Optional[EVModelConfig] = None
    appliance_config: Optional[ApplianceModelConfig] = None
    time_delta_min: float = 5.0

    # Weather date override (for historical/repeatable tests)
    # If None, uses start_datetime's date. If set, uses that date for weather.
    weather_date_override: Optional[datetime] = None
    closed_loop: bool = False


class DigitalTwin:
    """Main digital twin orchestrator.

    Each tick() returns the row for the current timestep (interval start)
    and then advances simulated time by time_delta_min, so a twin started
    at 00:00 emits 00:00, 00:05, ..., 23:55 for a full day. Returned rows
    match the generate_dataset.py timeseries schema.
    """

    def __init__(self, config: DigitalTwinConfig):
        """
        Args:
            config: DigitalTwinConfig with start_datetime (required) and
                    optional sub-model configurations.
        """
        if config.start_datetime is None:
            raise ValueError("start_datetime is required")

        self._start_datetime = config.start_datetime
        self._current_time = config.start_datetime
        self._run_start_time = self._current_time
        self._time_delta_min = config.time_delta_min
        self._step_count = 0
        self._closed_loop = bool(config.closed_loop)
        self._pending_observation = None

        # Initialize weather client
        self._weather_client = config.weather_client or WeatherClient()
        # Store weather date override for reset()
        self._weather_date_override = config.weather_date_override
        # Anchor date for the weather profile. Multi-day runs reuse this
        # single day's profile (time-of-day lookup) instead of issuing a
        # new network request per simulated date, consistent with EV and
        # appliances which already repeat their day-1 pattern.
        self._weather_date = (
            config.weather_date_override.date()
            if config.weather_date_override
            else config.start_datetime.date()
        )

        # Initialize sub-models
        self._solar = SolarModel(config.solar_config)
        self._battery = BatteryModel(config.battery_config)
        self._ev = EVModel(config.ev_config)
        self._appliances = ApplianceModel(
            config.appliance_config,
            closed_loop=self._closed_loop,
        )

        # Initialize sub-models at start time
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize all sub-models at the start time.

        Sub-model clocks are rewound one step behind the twin clock: their
        tick() advances-then-reports, so the rewind makes the first tick
        land exactly on the labeled start time. Without this, rows carry
        interval-end EV/appliance state under interval-start labels and
        every transition edge (e.g. EV departure) fires one tick early
        (issue #22). Rewound values may go slightly negative; tick()
        wraps with % 1440.
        """
        self._ev.initialize_at(self._current_time)
        self._appliances.initialize_at(self._current_time)
        self._ev._current_minute -= self._time_delta_min
        self._appliances._current_minute -= self._time_delta_min
        # Battery and solar don't need explicit initialization beyond config

    def _get_weather_for_current_time(self) -> tuple[float, float, float]:
        """Get weather data for current simulation time.

        The anchor date (start date, or weather_date_override when set) is
        combined with the current time-of-day, so the override actually
        selects which date's data is used and multi-day runs reuse the
        same daily profile instead of fetching new dates mid-run.

        Returns:
            Tuple of (irradiance_w_m2, temperature_c, cloudcover_pct)
        """
        query_time = datetime.combine(self._weather_date, self._current_time.time())
        irradiance = self._weather_client.get_irradiance_at_minute(query_time)
        temperature = self._weather_client.get_temperature_at_minute(query_time)
        cloudcover = self._weather_client.get_cloudcover_at_minute(query_time)
        return irradiance, temperature, cloudcover

    def observe(self, time_delta_min: Optional[float] = None) -> dict:
        """Observe one interval without applying battery or delivered power.

        The exogenous EV and appliance schedules advance exactly once. In
        closed-loop mode their SOC/runtime progress is held until commit().
        Calling observe() again before commit() is an ordering error.
        """
        if self._pending_observation is not None:
            raise RuntimeError("observe() requires commit() before the next observation")
        if time_delta_min is None:
            time_delta_min = self._time_delta_min
        if not math.isfinite(time_delta_min) or time_delta_min <= 0:
            raise ValueError("time_delta_min must be finite and positive")
        next_time = self._current_time + timedelta(minutes=time_delta_min)

        timestamp = self._current_time.strftime("%H:%M")
        irradiance, ambient_temp, _cloudcover = self._get_weather_for_current_time()
        solar_gen_kw = self._solar.calculate_power_kw_from_ambient(
            irradiance, ambient_temp
        )

        if self._closed_loop:
            # Submodels advance before sampling; anchor that sample to this
            # interval's start even when its duration differs from the last.
            minute = (
                self._current_time.hour * 60 + self._current_time.minute
                + self._current_time.second / 60
                + self._current_time.microsecond / 60_000_000
            )
            self._appliances._current_minute = minute - time_delta_min
            self._ev._current_minute = minute - time_delta_min
            appliance_result = self._appliances.observe(time_delta_min)
            ev_result = self._ev.observe(time_delta_min)
            desired = dict(appliance_result["appliance_desired_kw"])
            desired.pop("ev_charger", None)
        else:
            # Keep the legacy schedule path unchanged. Only battery physics
            # is deferred to commit(), so the default tick output is stable.
            appliance_result = self._appliances.tick(time_delta_min)
            ev_result = self._ev.tick(time_delta_min)
            desired = {
                appliance_id: load
                for appliance_id, load in appliance_result["appliance_loads"].items()
            }

        household_load_kw = appliance_result["total_load_kw"]
        self._pending_observation = {
            "next_time": next_time,
            "time_delta_min": time_delta_min,
            "timestamp": timestamp,
            "ambient_temp": ambient_temp,
            "solar_gen_kw": solar_gen_kw,
            "household_load_kw": household_load_kw,
            "appliance_result": appliance_result,
            "appliance_desired_kw": desired,
            "ev_result": ev_result,
        }
        return {
            "timestamp": timestamp,
            "solar_gen_kw": round(solar_gen_kw, 3),
            "battery_soc_pct": round(self._battery.soc_pct, 2),
            "battery_health_pct": round(self._battery.health_pct, 2),
            "ev_soc_pct": round(self._ev.soc_pct, 2),
            "ev_departure_min": ev_result["departure_min"],
            "ev_plugged_in": ev_result["plugged_in"],
            "appliance_desired_kw": desired,
        }

    def _validate_delivered(self, delivered: dict) -> dict[str, float]:
        """Validate and normalize a closed-loop delivery mapping."""
        allowed = {"battery_1", "ev_1"}
        allowed.update(
            f"appliance_{spec.appliance_id}"
            for spec in self._appliances.config.appliances
            if spec.appliance_id != "ev_charger"
        )
        unknown = set(delivered) - allowed
        if unknown:
            raise ValueError(
                f"unknown delivered power keys: {sorted(unknown, key=repr)!r}"
            )

        normalized = {}
        for key, value in delivered.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"delivered power for {key!r} must be numeric") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"delivered power for {key!r} must be finite")
            if key != "battery_1" and numeric < 0:
                raise ValueError(f"delivered power for {key!r} must be non-negative")
            normalized[key] = numeric
        return normalized

    def commit(self, delivered: Optional[dict] = None) -> dict:
        """Apply the observed interval and advance the twin clock.

        ``delivered=None`` preserves autonomous behavior. A delivery mapping
        is accepted only for a twin configured with ``closed_loop``; missing
        keys mean zero power for that device.
        """
        if self._pending_observation is None:
            raise RuntimeError("commit() requires observe() first")
        if delivered is not None and not self._closed_loop:
            raise ValueError("delivered power requires closed-loop mode")
        if delivered is not None and not isinstance(delivered, dict):
            raise ValueError("delivered power must be a dict or None")

        pending = self._pending_observation
        time_delta_min = pending["time_delta_min"]
        if delivered is not None:
            delivered = self._validate_delivered(delivered)
            if (
                delivered.get("ev_1", 0.0) > 0.0
                and not pending["ev_result"]["plugged_in"]
            ):
                raise ValueError("cannot deliver positive EV power while unplugged")

        if self._closed_loop:
            if delivered is None:
                battery_power_kw = pending["solar_gen_kw"] - pending["household_load_kw"]
                appliance_result = self._appliances.commit(time_delta_min, None)
                ev_result = self._ev.commit_autonomous(time_delta_min)
            else:
                battery_power_kw = delivered.get("battery_1", 0.0)
                appliance_result = self._appliances.commit(time_delta_min, delivered)
                ev_result = self._ev.commit_delivered(
                    time_delta_min, delivered.get("ev_1", 0.0)
                )
            battery_result = self._battery.update(
                battery_power_kw, time_delta_min, pending["ambient_temp"]
            )
        else:
            appliance_result = pending["appliance_result"]
            ev_result = pending["ev_result"]
            battery_power_kw = pending["solar_gen_kw"] - pending["household_load_kw"]
            battery_result = self._battery.update(
                battery_power_kw, time_delta_min, pending["ambient_temp"]
            )

        row = {
            "timestamp": pending["timestamp"],
            "solar_gen_kw": round(pending["solar_gen_kw"], 3),
            "battery_soc_pct": round(battery_result["soc_pct"], 2),
            "battery_health_pct": round(battery_result["health_pct"], 2),
            "household_load_kw": round(pending["household_load_kw"], 3),
            "ev_soc_pct": round(ev_result["soc_pct"], 2),
            "ev_departure_min": ev_result["departure_min"],
            "ev_plugged_in": ev_result["plugged_in"],
        }
        if self._closed_loop:
            row["appliance_desired_kw"] = dict(pending["appliance_desired_kw"])
            row["appliance_served_kw"] = dict(
                appliance_result.get("appliance_served_kw", {})
            )
            dt_hours = time_delta_min / 60.0
            row["battery_applied_kw"] = float(
                battery_result["energy_charged_kwh"]
                / self._battery.config.charge_efficiency / dt_hours
                - battery_result["energy_discharged_kwh"]
                * self._battery.config.discharge_efficiency / dt_hours
            )
            row["ev_applied_kw"] = float(ev_result["charging_kw"])

        self._current_time = pending["next_time"]
        self._step_count += 1
        self._pending_observation = None
        return row

    def tick(self, time_delta_min: Optional[float] = None) -> dict:
        """Observe and autonomously commit one interval."""
        self.observe(time_delta_min)
        return self.commit(None)

    @property
    def current_time(self) -> datetime:
        """Current simulation time."""
        return self._current_time

    @property
    def step_count(self) -> int:
        """Number of ticks executed."""
        return self._step_count

    @property
    def simulated_minutes(self) -> float:
        """Minutes advanced since construction or the most recent reset."""
        return (self._current_time - self._run_start_time).total_seconds() / 60.0

    def run_full_day(self, steps: int = 288, time_delta_min: Optional[float] = None) -> list[dict]:
        """Run a full 24-hour simulation and return all rows.

        Args:
            steps: Number of timesteps (default 288 for 24h at 5-min).
            time_delta_min: Minutes per step (default from config).

        Returns:
            List of row dicts (length = steps).
        """
        rows = []
        for _ in range(steps):
            rows.append(self.tick(time_delta_min))
        return rows

    def reset(self, start_datetime: Optional[datetime] = None) -> None:
        """Reset twin to its initial state.

        Restores the start time (or jumps to start_datetime when given),
        rewinds EV/appliance RNG and segments via their reset() methods,
        rebuilds the battery from config, and re-initializes sub-models —
        so a reset twin replays exactly like a freshly constructed one
        (issue #23).
        """
        if self._pending_observation is not None:
            raise RuntimeError("cannot reset with an uncommitted observation")

        # Reset to original start datetime unless a new one is provided
        self._current_time = (
            start_datetime if start_datetime is not None else self._start_datetime
        )
        self._run_start_time = self._current_time
        self._step_count = 0
        # Recompute weather anchor: use override if configured, else the (new) start date
        self._weather_date = (
            self._weather_date_override.date()
            if self._weather_date_override
            else self._current_time.date()
        )
        self._ev.reset()
        self._appliances.reset()
        # Re-initialize battery to config defaults
        self._battery = BatteryModel(self._battery.config)
        self._initialize_models()
