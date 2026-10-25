"""Empirical degradation, closed-loop data and optional tiny PINN tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from battery_health.dataset import EmpiricalBaseline, daily_summary, generate_battery_dataset
from digital_twin.battery_model import BatteryModel, BatteryModelConfig
from scenarios import build_twin
from simulation.closed_loop import run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate


def test_degradation_off_preserves_records_and_metrics():
    from simulation.metrics import score_run
    old = run_closed_loop(build_twin("clear_day", closed_loop=True), hybrid_allocate)
    explicit = run_closed_loop(build_twin("clear_day", closed_loop=True,
                                          battery_config=BatteryModelConfig(degradation=False)), hybrid_allocate)
    assert old == explicit
    assert score_run(old) == score_run(explicit)
    assert "battery_telemetry" not in old[0]


def _loss(config, power=0):
    model = BatteryModel(config)
    start = model.health_pct
    health = []
    for _ in range(12):
        # Alternating terminal power keeps SOC away from the clamps.
        model.update(power if len(health) % 2 else -power, 5)
        health.append(model.health_pct)
    assert np.all(np.diff(health) <= 0)
    return start - model.health_pct


def test_aging_faster_at_higher_c_rate_temperature_and_soc():
    base = BatteryModelConfig(degradation=True)
    assert _loss(base, 6) > _loss(base, 1)
    assert _loss(replace(base, ambient_temp_c=40), 1) > _loss(base, 1)
    assert _loss(replace(base, initial_soc_pct=80)) > _loss(replace(base, initial_soc_pct=20))
    assert _loss(replace(base, degradation=False), 6) == 0


@pytest.fixture(scope="module")
def dataset():
    return generate_battery_dataset(days=2, seed=7)


def test_closed_loop_data_deterministic_multiday_and_monotonic(dataset):
    pd.testing.assert_frame_equal(dataset, generate_battery_dataset(days=2, seed=7), check_exact=True)
    assert len(dataset) == 3 * 2 * 288
    assert len(daily_summary(dataset)) == 6
    for _, group in dataset.groupby("scenario"):
        assert np.all(np.diff(group.next_health_pct) <= 0)
        assert np.all(np.diff(group.throughput_kwh) >= 0)
        assert group.power_kw.abs().sum() > 0
    assert (dataset.next_health_pct < dataset.health_pct).all()


def test_empirical_baseline_heldout_error_is_reported(dataset):
    training = dataset[dataset.scenario != "heatwave"]
    heldout = dataset[dataset.scenario == "heatwave"]
    baseline = EmpiricalBaseline().fit(training)
    error = np.abs(baseline.predict(heldout) - heldout[["next_health_pct", "next_temperature_c"]].to_numpy()).mean()
    assert error < 1e-10


def test_tiny_pinn_reproducible_and_physics_ablation(dataset):
    pytest.importorskip("torch")
    from battery_health.pinn import train
    tiny = dataset.iloc[::15]
    first = train(tiny, epochs=15)
    second = train(tiny, epochs=15)
    np.testing.assert_array_equal(first.predict(tiny), second.predict(tiny))
    plain = train(tiny, epochs=15, physics_weight=0)
    assert np.isfinite(first.predict(tiny)).all()
    assert not np.array_equal(first.predict(tiny), plain.predict(tiny))


def test_hot_high_c_rate_case_raises_predicted_risk(dataset):
    pytest.importorskip("torch")
    from battery_health.pinn import thermal_risk, train
    from battery_health.dataset import operation_row
    config = BatteryModelConfig(degradation=True, ambient_temp_c=50, max_charge_rate_c=2)
    model = BatteryModel(config)
    before = {"soc_pct": model.soc_pct, "health_pct": model.health_pct,
              "temperature_c": model.temperature_c}
    state = model.update(24, 5, 50)
    record = {"dt_min": 5, "battery_kw": 24, "battery_telemetry": {
        "before": before, "soc_pct": state["soc_pct"], "health_pct": state["health_pct"],
        "temperature_c": state["temperature_c"], "ambient_temp_c": 50}}
    frame = pd.DataFrame([operation_row(record, config)])
    predicted = train(dataset.iloc[::15], epochs=15).predict(frame)
    assert "high_temperature" in thermal_risk(predicted[0, 1], 50, 5)
    assert "rapid_temperature_rise" in thermal_risk(40, 30, 5)
    assert thermal_risk(30, 30, 5) == []


def test_health_dashboard_records_are_for_selected_run():
    from dashboard.data import RunKey, run
    records = run(RunKey("heatwave", "baseline", battery_health=True))
    assert len(records) == 288
    assert records[0]["battery_telemetry"]["before"]["health_pct"] == 95
    assert records[-1]["battery_telemetry"]["health_pct"] < 95



def test_physics_basis_does_not_read_next_state_labels():
    from copy import deepcopy
    from battery_health.dataset import operation_row
    config = BatteryModelConfig(degradation=True)
    record = run_closed_loop(build_twin("clear_day", closed_loop=True, battery_config=config),
                              hybrid_allocate, steps=1)[0]
    original = operation_row(record, config)
    changed = deepcopy(record)
    changed["battery_telemetry"].update(soc_pct=99, health_pct=5, temperature_c=-15)
    after = operation_row(changed, config)
    for key in ("physics_loss_pct", "cycle_loss_pct", "calendar_loss_pct", "physics_temp_delta_c"):
        assert after[key] == original[key]
