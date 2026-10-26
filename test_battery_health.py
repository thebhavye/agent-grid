"""PINN uses #69's unchanged schema, split and daily targets."""

import numpy as np
import pytest

from battery_health.baseline import DEFAULT_SCENARIOS, fit_baseline
from battery_health.dataset import generate_battery_dataset


@pytest.fixture(scope="module")
def steps():
    return generate_battery_dataset(DEFAULT_SCENARIOS, 3, 42)


def test_same_daily_split_targets_and_baseline(steps):
    pytest.importorskip("torch")
    from battery_health.pinn import daily_frame, train
    from battery_health.train import evaluate
    daily = daily_frame(steps)
    left, right = daily[daily.scenario != "heatwave"], daily[daily.scenario == "heatwave"]
    baseline = fit_baseline(steps)
    assert len(left) == baseline["train_days"] == 6
    assert len(right) == baseline["test_days"] == 3
    first, second = train(left, epochs=20), train(left, epochs=20)
    np.testing.assert_array_equal(first.predict(right), second.predict(right))
    result = evaluate(right, first.predict(right))
    assert set(result) == {"health_mae_pct", "temperature_mae_c"}
    assert np.isfinite(list(result.values())).all()
    assert not np.array_equal(first.predict(right), train(left, epochs=20, physics_weight=0).predict(right))


def test_physics_targets_do_not_read_measured_health_or_temperature(steps):
    pytest.importorskip("torch")
    from battery_health.pinn import daily_frame
    original = daily_frame(steps)
    changed = steps.copy()
    changed["health_pct"], changed["temperature_c"] = 1, -15
    after = daily_frame(changed)
    np.testing.assert_array_equal(original[["physics_loss_pct", "physics_peak_c"]],
                                  after[["physics_loss_pct", "physics_peak_c"]])


def test_selected_dashboard_run_uses_existing_telemetry():
    from dashboard.data import RunKey, battery_steps, run
    records = run(RunKey("heatwave", "baseline", battery_health=True))
    assert "battery_telemetry" not in records[0]
    assert records[-1]["battery_health_pct"] < 95
    assert len(battery_steps(records, "heatwave", seed=7)) == 288
    assert set(battery_steps(records, "heatwave", seed=7).seed) == {7}


def test_hot_high_c_rate_prediction_flags_risk(steps):
    pytest.importorskip("torch")
    from battery_health.pinn import daily_frame, thermal_risk, train
    hot = steps.copy()
    hot["initial_temperature_c"] = 50
    hot["ambient_temperature_c"] = 50
    hot["temperature_c"] = 50.5
    hot["power_kw"] = 12
    hot["c_rate"] = 1
    daily = daily_frame(hot)
    predicted = train(daily, epochs=200).predict(daily)
    assert "high_temperature" in thermal_risk(predicted[0, 1], predicted[0, 1], 1440)
    assert "rapid_temperature_rise" in thermal_risk(40, 30, 5)
