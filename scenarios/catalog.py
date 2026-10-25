"""Offline scenario catalog using the existing twin configuration APIs."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from digital_twin.appliance_model import ApplianceModelConfig
from digital_twin.battery_model import BatteryModelConfig
from digital_twin.ev_model import EVModelConfig, EVScenario
from digital_twin.solar_model import SolarModelConfig
from digital_twin.twin import DigitalTwin, DigitalTwinConfig
from digital_twin.weather_client import WeatherClient


_WEATHER_DIR = Path(__file__).parent / "weather"
_BASE = {
    "weather_file": "vellore_2024-01-25.json",
    "start_date": "2024-01-25",
    "ev_scenario": EVScenario.MIDDAY,
    "ev_seed": 42,
    "appliance_seed": 42,
    "initial_soc_pct": 50.0,
    "capacity_kwh": 12.0,
    "steps": 288,
    "time_delta_min": 5.0,
}
CATALOG = {
    "clear_day": dict(_BASE),
    "demo_sunny_home": {**_BASE, "panel_area_m2": 60.0},
    "monsoon_cloudy": {
        **_BASE, "weather_file": "vellore_2023-12-04.json", "start_date": "2023-12-04",
    },
    "heatwave": {
        **_BASE, "weather_file": "vellore_2024-05-01.json", "start_date": "2024-05-01",
    },
    "low_battery_start": {**_BASE, "initial_soc_pct": 20.0},
    "degraded_battery": {**_BASE, "capacity_kwh": 6.0},
    "overnight_ev_cloudy": {
        **_BASE, "weather_file": "vellore_2023-12-04.json", "start_date": "2023-12-04",
        "ev_scenario": EVScenario.OVERNIGHT,
    },
    "three_days": {**_BASE, "steps": 864},
}


class _SavedResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return deepcopy(self._payload)


class _SavedSession:
    """Requests-compatible transport that only reads a bundled JSON file."""

    def __init__(self, path):
        self._payload = json.loads(path.read_text(encoding="utf-8"))

    def get(self, url, timeout=30):
        return _SavedResponse(self._payload)


class _OfflineWeatherClient(WeatherClient):
    """Keep scenario profiles independent of the live client's global cache."""

    def __init__(self, path):
        self._profiles = {}
        super().__init__(session=_SavedSession(path))

    def _get_cached(self, target_date):
        return self._profiles.get(target_date)

    def _set_cached(self, target_date, data):
        self._profiles[target_date] = data

    def clear_cache(self):
        self._profiles.clear()


def list_scenarios() -> list[str]:
    """Return the available names in deterministic alphabetical order."""
    return sorted(CATALOG)


def build_twin(name, closed_loop=False, *, seed=None, battery_config=None) -> DigitalTwin:
    """Build a fresh twin using saved weather, with no network or disk cache.

    The weather anchor remains the scenario's source day across multi-day
    runs and resets. Step count is available as CATALOG[name]["steps"].
    """
    try:
        scenario = CATALOG[name]
    except KeyError as exc:
        raise ValueError(f"unknown scenario {name!r}; choose from {list_scenarios()}") from exc
    start = datetime.fromisoformat(scenario["start_date"])
    return DigitalTwin(DigitalTwinConfig(
        start_datetime=start,
        weather_date_override=start,
        weather_client=_OfflineWeatherClient(_WEATHER_DIR / scenario["weather_file"]),
        solar_config=SolarModelConfig(panel_area_m2=scenario.get("panel_area_m2", 20.0)),
        ev_config=EVModelConfig(scenario=scenario["ev_scenario"], seed=scenario["ev_seed"] if seed is None else seed),
        appliance_config=ApplianceModelConfig(seed=scenario["appliance_seed"] if seed is None else seed),
        battery_config=battery_config or BatteryModelConfig(
            initial_soc_pct=scenario["initial_soc_pct"], capacity_kwh=scenario["capacity_kwh"],
        ),
        time_delta_min=scenario["time_delta_min"],
        closed_loop=closed_loop,
    ))
