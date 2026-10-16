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
        self._appliances = ApplianceModel(config.appliance_config)

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

    def tick(self, time_delta_min: Optional[float] = None) -> dict:
        """Return the row for the current timestep, then advance simulation.

        The row is both sampled AND labeled at the current time (interval
        start), so a twin started at 00:00 emits 00:00 first and 23:55 as
        its 288th row.

        Args:
            time_delta_min: Minutes to advance. If None, uses config default (5 min).

        Returns:
            Dict with keys matching generate_dataset.py timeseries schema:
                - "timestamp": str "HH:MM"
                - "solar_gen_kw": float
                - "battery_soc_pct": float
                - "battery_health_pct": float
                - "household_load_kw": float
                - "ev_soc_pct": float
                - "ev_departure_min": int
                - "ev_plugged_in": bool
        """
        if time_delta_min is None:
            time_delta_min = self._time_delta_min

        # Sample AND label at the current time (interval start).
        timestamp = self._current_time.strftime("%H:%M")

        # Get weather for current time
        irradiance, ambient_temp, cloudcover = self._get_weather_for_current_time()

        # Solar generation
        solar_gen_kw = self._solar.calculate_power_kw_from_ambient(irradiance, ambient_temp)

        # Household load from appliances
        appliance_result = self._appliances.tick(time_delta_min)
        household_load_kw = appliance_result["total_load_kw"]

        # Battery update: net power = solar - load
        net_power_kw = solar_gen_kw - household_load_kw
        battery_result = self._battery.update(net_power_kw, time_delta_min, ambient_temp)

        # EV update
        ev_result = self._ev.tick(time_delta_min)

        # Build row matching generate_dataset.py schema exactly
        row = {
            "timestamp": timestamp,
            "solar_gen_kw": round(solar_gen_kw, 3),
            "battery_soc_pct": round(battery_result["soc_pct"], 2),
            "battery_health_pct": round(battery_result["health_pct"], 2),
            "household_load_kw": round(household_load_kw, 3),
            "ev_soc_pct": round(ev_result["soc_pct"], 2),
            "ev_departure_min": ev_result["departure_min"],
            "ev_plugged_in": ev_result["plugged_in"],
        }

        # Advance time after sampling/labeling
        self._current_time += timedelta(minutes=time_delta_min)
        self._step_count += 1

        return row

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
