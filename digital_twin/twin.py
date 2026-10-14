"""Digital Twin orchestrator.

Coordinates weather, solar, battery, EV, and appliance models to produce
a single timeseries row per tick matching the existing dataset schema.

The DigitalTwin.tick() advances simulation by one timestep and returns
a dict with the same keys as data/generate_dataset.py timeseries rows.

Usage:
    twin = DigitalTwin(start_datetime=datetime(2024, 1, 15, 0, 0))
    for _ in range(288):  # 24 hours at 5-min resolution
        row = twin.tick(5)
        # row has: solar_gen_kw, battery_soc_pct, battery_health_pct,
        #          household_load_kw, ev_soc_pct, ev_departure_min, ev_plugged_in
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from digital_twin.weather_client import WeatherClient
from digital_twin.solar_model import SolarModel, SolarModelConfig
from digital_twin.battery_model import BatteryModel, BatteryModelConfig
from digital_twin.ev_model import EVModel, EVModelConfig, EVScenario
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

    Each tick() advances simulated time by time_delta_min and returns
    a row dict matching the generate_dataset.py timeseries schema.
    """

    def __init__(self, config: DigitalTwinConfig):
        """
        Args:
            config: DigitalTwinConfig with start_datetime (required) and
                    optional sub-model configurations.
        """
        if config.start_datetime is None:
            raise ValueError("start_datetime is required")

        self._current_time = config.start_datetime
        self._time_delta_min = config.time_delta_min
        self._step_count = 0

        # Initialize weather client
        self._weather_client = config.weather_client or WeatherClient()
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
        self._solar_initialized = False
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize all sub-models at the start time."""
        self._ev.initialize_at(self._current_time)
        self._appliances.initialize_at(self._current_time)
        # Battery and solar don't need explicit initialization beyond config

    def _get_weather_for_current_time(self) -> tuple[float, float, float]:
        """Get weather data for current simulation time.

        Returns:
            Tuple of (irradiance_w_m2, temperature_c, cloudcover_pct)
        """
        irradiance = self._weather_client.get_irradiance_at_minute(self._current_time)
        temperature = self._weather_client.get_temperature_at_minute(self._current_time)
        cloudcover = self._weather_client.get_cloudcover_at_minute(self._current_time)
        return irradiance, temperature, cloudcover

    def tick(self, time_delta_min: Optional[float] = None) -> dict:
        """Advance simulation by one timestep and return current row.

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

        # Get weather for current time (before advancing)
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

        # Advance time
        self._current_time += timedelta(minutes=time_delta_min)
        self._step_count += 1

        # Format timestamp as HH:MM
        timestamp = self._current_time.strftime("%H:%M")

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
        """Total simulated minutes elapsed."""
        return self._step_count * self._time_delta_min

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
        """Reset twin to initial state."""
        if start_datetime is not None:
            self._current_time = start_datetime
        self._step_count = 0
        self._initialize_models()
        # Re-initialize battery to config defaults
        self._battery = BatteryModel(self._battery.config)