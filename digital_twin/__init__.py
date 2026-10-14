"""Digital Twin module for AgentGrid.

A tick-by-tick live simulator that models the same system as the static
dataset generator (solar, battery, EV, appliances) using real physics
equations and real-world weather data, instead of pre-generated random curves.

Output matches the existing dataset schema exactly, so existing agents work
with zero changes.
"""

from digital_twin.twin import DigitalTwin, DigitalTwinConfig
from digital_twin.weather_client import WeatherClient
from digital_twin.solar_model import SolarModel, SolarModelConfig
from digital_twin.battery_model import BatteryModel, BatteryModelConfig
from digital_twin.ev_model import EVModel, EVModelConfig, EVScenario
from digital_twin.appliance_model import ApplianceModel, ApplianceModelConfig

__all__ = [
    "DigitalTwin",
    "DigitalTwinConfig",
    "WeatherClient",
    "SolarModel",
    "SolarModelConfig",
    "BatteryModel",
    "BatteryModelConfig",
    "EVModel",
    "EVModelConfig",
    "EVScenario",
    "ApplianceModel",
    "ApplianceModelConfig",
]