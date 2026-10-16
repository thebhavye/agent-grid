"""Saved scenarios work deterministically with every network path blocked."""

from copy import deepcopy
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import socket

import pytest
import requests

from digital_twin import weather_client
from digital_twin.solar_model import SolarModel
from scenarios import CATALOG, build_twin, list_scenarios
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("scenario attempted network access")

    monkeypatch.setattr(requests, "get", forbid_network)
    monkeypatch.setattr(requests.sessions.Session, "request", forbid_network)
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(weather_client, "_cache", {})


def test_catalog_contains_all_eight_scenarios():
    expected = {"clear_day", "monsoon_cloudy", "heatwave", "low_battery_start",
                "degraded_battery", "overnight_ev_cloudy", "three_days", "demo_sunny_home"}
    assert set(list_scenarios()) == expected
    assert list_scenarios() == sorted(expected)
    names = list_scenarios()
    names.clear()
    assert set(list_scenarios()) == expected
    assert CATALOG["three_days"]["steps"] == 864
    assert CATALOG["low_battery_start"]["initial_soc_pct"] == 20
    assert CATALOG["degraded_battery"]["capacity_kwh"] < CATALOG["clear_day"]["capacity_kwh"]


@pytest.mark.parametrize("name", ["unknown", ""])
def test_unknown_scenario_is_clear(name):
    with pytest.raises(ValueError, match="unknown scenario"):
        build_twin(name)


@pytest.mark.parametrize("closed_loop", [False, True])
@pytest.mark.parametrize("name", list_scenarios())
def test_complete_scenario_runs_are_valid_and_deterministic(name, closed_loop):
    spec = CATALOG[name]
    before = deepcopy(CATALOG)
    twin = build_twin(name, closed_loop=closed_loop)
    assert twin._ev.scenario == spec["ev_scenario"]
    assert twin._ev.config.seed == spec["ev_seed"]
    assert twin._appliances.config.seed == spec["appliance_seed"]
    assert twin._battery.soc_pct == spec["initial_soc_pct"]
    assert twin._battery.capacity_kwh == spec["capacity_kwh"]
    rows = twin.run_full_day(steps=spec["steps"])
    other = build_twin(name, closed_loop=closed_loop)
    assert rows == other.run_full_day(steps=spec["steps"])
    assert CATALOG == before
    assert len(rows) == twin.step_count == spec["steps"]
    assert twin.current_time == datetime.fromisoformat(spec["start_date"]) + timedelta(
        minutes=spec["steps"] * spec["time_delta_min"])
    for index, row in enumerate(rows):
        assert row["timestamp"] == f"{(index * 5 // 60) % 24:02d}:{index * 5 % 60:02d}"
        assert math.isfinite(row["solar_gen_kw"]) and row["solar_gen_kw"] >= 0
        assert math.isfinite(row["household_load_kw"]) and row["household_load_kw"] >= 0
        for key in ("battery_soc_pct", "battery_health_pct", "ev_soc_pct"):
            assert math.isfinite(row[key]) and 0 <= row[key] <= 100
        assert isinstance(row["ev_plugged_in"], bool)
    if name == "three_days":
        solar = [row["solar_gen_kw"] for row in rows]
        assert solar[:288] == solar[288:576] == solar[576:864]


@pytest.mark.parametrize("name", list_scenarios())
def test_scenarios_also_run_with_actual_allocated_delivery(name):
    twin = build_twin(name, closed_loop=True)
    records = run_closed_loop(twin, hybrid_allocate, steps=CATALOG[name]["steps"])
    assert len(records) == CATALOG[name]["steps"]
    for record in records:
        loads = sum(value for key, value in record["allocated_kw"].items()
                    if key not in {"solar_1", "battery_1"})
        power = record["battery_kw"]
        assert record["solar_kw"] + max(0, -power) == pytest.approx(
            loads + max(0, power) + record["curtailed_kw"], abs=1e-9)
        assert record["battery_soc_pct"] >= 20 - 1e-9


@pytest.mark.parametrize("name", ["clear_day", "monsoon_cloudy", "heatwave"])
def test_saved_weather_passes_existing_validation_and_sanitization(name):
    twin = build_twin(name)
    day = twin.current_time.date()
    path = Path(__file__).parent / "weather" / CATALOG[name]["weather_file"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["_provenance"]["source_date"] == day.isoformat()
    assert raw["_provenance"]["modified_hourly_values"] is False
    parsed = twin._weather_client.get_weather(day)
    assert twin._weather_client._is_valid_cached_profile(parsed)
    hourly = deepcopy(raw["hourly"])
    twin._weather_client._validate_data(hourly)
    assert hourly == raw["hourly"]
    assert parsed["date"] == day
    assert parsed["time"] == [f"{day}T{hour:02d}:00" for hour in range(24)]
    for key in ("shortwave_radiation", "cloudcover", "temperature_2m"):
        assert len(parsed[key]) == 24 and parsed[key] == hourly[key]
        assert all(math.isfinite(value) for value in parsed[key])
    assert all(value >= 0 for value in parsed["shortwave_radiation"])
    assert all(0 <= value <= 100 for value in parsed["cloudcover"])
    assert all(-20 <= value <= 60 for value in parsed["temperature_2m"])


def test_cloudy_peak_is_clearly_lower_than_clear_day():
    peaks = [max(row["solar_gen_kw"] for row in build_twin(name).run_full_day())
             for name in ("monsoon_cloudy", "clear_day")]
    assert 0 < peaks[0] < peaks[1] * 0.5


def test_hot_day_derates_relative_to_same_irradiance_at_25_c_ambient():
    twin = build_twin("heatwave")
    weather = twin._weather_client.get_weather(twin.current_time.date())
    index = max(range(24), key=lambda i: weather["shortwave_radiation"][i])
    irradiance = weather["shortwave_radiation"][index]
    ambient = weather["temperature_2m"][index]
    assert ambient > 35
    model = SolarModel()
    hot = model.calculate_power_kw_from_ambient(irradiance, ambient)
    control = model.calculate_power_kw_from_ambient(irradiance, 25)
    assert 0 < hot < control * 0.95
    rows = twin.run_full_day()
    assert rows[index * 12]["solar_gen_kw"] == pytest.approx(hot, abs=0.0005)


def test_scenario_cache_is_independent_of_live_cache_and_other_twins():
    first = build_twin("clear_day")
    day = first.current_time.date()
    profile = first._weather_client.get_weather(day)
    expected = deepcopy(profile)
    key = (12.97, 79.16, day.isoformat())
    poisoned = {**deepcopy(profile), "shortwave_radiation": [0.0] * 24}
    weather_client._cache[key] = poisoned
    profile["shortwave_radiation"][:] = [0.0] * 24
    second = build_twin("clear_day")
    assert second._weather_client.get_weather(day) == expected
    first._weather_client.clear_cache()
    assert first._weather_client.get_weather(day) == expected
    assert weather_client._cache[key] == poisoned


def test_saved_weather_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    twin = build_twin("clear_day")
    assert len(twin.run_full_day(steps=2)) == 2


@pytest.mark.parametrize("name", ["clear_day", "monsoon_cloudy", "heatwave",
                                  "low_battery_start", "degraded_battery",
                                  "overnight_ev_cloudy", "three_days"])
@pytest.mark.parametrize("closed_loop", [False, True])
def test_original_scenarios_exactly_match_legacy_solar_defaults(name, closed_loop):
    current = build_twin(name, closed_loop=closed_loop)
    legacy = build_twin(name, closed_loop=closed_loop)
    # Previously no solar_config was supplied: SolarModel() is the reference.
    legacy._solar = SolarModel()
    assert current._solar.config == legacy._solar.config
    steps = CATALOG[name]["steps"]
    assert current.run_full_day(steps=steps) == legacy.run_full_day(steps=steps)


@pytest.mark.parametrize("allocator", [hybrid_allocate, proportional_adapter])
def test_demo_sunny_home_has_visible_battery_cycling_offline(allocator):
    twin = build_twin("demo_sunny_home", closed_loop=True)
    assert twin._solar.config.panel_area_m2 == 60.0
    assert twin._solar.config.panel_efficiency == 0.20
    assert twin._battery.soc_pct == 50.0
    assert twin._battery.capacity_kwh == 12.0
    records = run_closed_loop(twin, allocator, steps=CATALOG["demo_sunny_home"]["steps"])
    assert len(records) == 288
    assert sum(max(0, row["battery_kw"]) * row["dt_min"] / 60 for row in records) > 5
    assert max(row["battery_soc_pct"] for row in records
               if "12:00" <= row["timestamp"] < "18:00") > 80
    assert any(row["battery_kw"] < 0 for row in records)
