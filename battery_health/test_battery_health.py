"""Synthetic degradation checks, independent held-out data and legacy parity."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import requests

from battery_health.baseline import DEFAULT_SCENARIOS, fit_baseline
from battery_health.dataset import generate_battery_dataset, summarize_days
from digital_twin.battery_model import BatteryModel, BatteryModelConfig
from scenarios import build_twin
from simulation.closed_loop import run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("battery dataset attempted network")
    monkeypatch.setattr(requests.sessions.Session, "request", fail)


def test_degradation_off_preserves_bytes_and_constant_health():
    left = build_twin("demo_sunny_home", closed_loop=True)
    right = build_twin("demo_sunny_home", closed_loop=True)
    right._battery = BatteryModel(replace(right._battery.config, degradation=False,
                                         cycle_fade_pct_per_kwh=10,
                                         calendar_fade_pct_per_day=10))
    first = run_closed_loop(left, hybrid_allocate)
    second = run_closed_loop(right, hybrid_allocate)
    assert json.dumps(first) == json.dumps(second)
    assert left._battery.health_pct == right._battery.health_pct == 95.0


def test_monotonic_health_and_reset():
    battery = BatteryModel(BatteryModelConfig(degradation=True))
    health = [battery.health_pct]
    for i in range(100):
        battery.update(3 if i % 2 else -3, 5, ambient_temp_c=35)
        health.append(battery.health_pct)
    assert np.all(np.diff(health) <= 0)
    assert health[-1] < health[0]
    assert battery.throughput_kwh > 0
    battery.reset()
    assert battery.health_pct == 95
    assert battery.throughput_kwh == battery.cycle_exposure == battery.calendar_exposure == 0


def test_faster_c_rate_at_equal_terminal_energy():
    cfg = BatteryModelConfig(degradation=True, calendar_fade_pct_per_day=0,
                             initial_soc_pct=20, thermal_resistance=0)
    low, high = BatteryModel(cfg), BatteryModel(cfg)
    low.update(2, 60)
    high.update(4, 30)
    assert low.throughput_kwh == pytest.approx(high.throughput_kwh)
    assert low.soc_pct == high.soc_pct
    assert high.health_pct < low.health_pct


@pytest.mark.parametrize("parameter,low,high", [("ambient_temp_c", 20, 40),
                                                ("initial_soc_pct", 20, 80)])
def test_temperature_and_soc_sensitivity(parameter, low, high):
    cfg = BatteryModelConfig(degradation=True)
    left = BatteryModel(replace(cfg, **{parameter: low}))
    right = BatteryModel(replace(cfg, **{parameter: high}))
    left.update(0, 60)
    right.update(0, 60)
    assert right.health_pct < left.health_pct


def test_soc_clamp_does_not_age_untransferred_cycle_energy():
    battery = BatteryModel(BatteryModelConfig(degradation=True, initial_soc_pct=100,
                                              calendar_fade_pct_per_day=0))
    battery.update(6, 5)
    assert battery.throughput_kwh == battery.cycle_exposure == 0
    assert battery.health_pct == 95


@pytest.mark.parametrize("name", ["cycle_fade_pct_per_kwh", "calendar_fade_pct_per_day",
                                 "c_rate_stress", "aging_temperature_scale_c"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_aging_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        BatteryModelConfig(**{name: value})


def test_invalid_enabled_timestep_does_not_mutate():
    battery = BatteryModel(BatteryModelConfig(degradation=True))
    before = vars(battery).copy()
    with pytest.raises(ValueError):
        battery.update(1, 0, 40)
    assert vars(battery) == before


def test_three_scenarios_multi_day_dataset_determinism_and_baseline():
    first = generate_battery_dataset(DEFAULT_SCENARIOS, 3, 42)
    second = generate_battery_dataset(DEFAULT_SCENARIOS, 3, 42)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 3*3*288
    assert set(first.scenario) == set(DEFAULT_SCENARIOS)
    assert set(first.day) == {0, 1, 2}
    assert np.isfinite(first.select_dtypes(include="number")).all().all()
    for _, rows in first.groupby("scenario"):
        assert np.all(np.diff(rows.health_pct) <= 0)
        assert (rows.soc_pct >= 20-1e-9).all()
        expected = np.cumsum(abs(rows.power_kw)*rows.dt_min/60)
        np.testing.assert_allclose(rows.throughput_kwh, expected, atol=1e-10)
    daily = summarize_days(first)
    assert len(daily) == 9
    assert (daily.end_health_pct <= daily.start_health_pct).all()
    assert daily.throughput_kwh.sum() == pytest.approx(
        first.groupby("scenario").throughput_kwh.last().sum())
    result = fit_baseline(first)
    assert result["train_days"] == 6
    assert result["test_days"] == 3
    assert result["held_out_scenario"] == "heatwave"
    assert result["health_mae_pct"] < 1e-8
    assert 0 <= result["temperature_mae_c"] < 2
    with pytest.raises(ValueError):
        fit_baseline(first, held_out="absent")


def test_seed_changes_actual_operation_and_reset_replays():
    first = generate_battery_dataset(["demo_sunny_home"], 1, 1)
    second = generate_battery_dataset(["demo_sunny_home"], 1, 7)
    assert not first.power_kw.equals(second.power_kw)
    twin = build_twin("clear_day", closed_loop=True)
    twin._battery = BatteryModel(replace(twin._battery.config, degradation=True))
    original = run_closed_loop(twin, hybrid_allocate)
    twin.reset()
    assert run_closed_loop(twin, hybrid_allocate) == original


@pytest.mark.parametrize("days", [0, -1, 1.5, True])
def test_invalid_days(days):
    with pytest.raises(ValueError):
        generate_battery_dataset(DEFAULT_SCENARIOS, days, 42)
