import math

import pandas as pd
import pytest

from agents.battery_agent import BatteryAgent
from digital_twin.faults import FaultConfig, FaultInjector, generate_fault_dataset


def test_fixed_faults_labels_and_input_are_clean():
    row = {"battery_soc_pct": 50.0, "solar_gen_kw": 2.0}
    original = dict(row)
    config = FaultConfig(enabled=True, faults=[
        {"type": "dropout", "key": "solar_gen_kw", "start_step": 1, "end_step": 3},
        {"type": "nan", "key": "battery_soc_pct", "start_step": 2, "end_step": 3},
    ])
    injector = FaultInjector(config)
    observed, labels = injector.apply(row, 2)
    assert row == original
    assert "solar_gen_kw" not in observed
    assert math.isnan(observed["battery_soc_pct"])
    assert {(item["key"], item["fault_type"]) for item in labels} == {
        ("solar_gen_kw", "dropout"), ("battery_soc_pct", "nan")}


@pytest.mark.parametrize("kind", ["stuck", "spike", "drift"])
def test_other_faults_are_deterministic_and_finite(kind):
    spec = {"type": kind, "key": "x", "start_step": 1, "end_step": 4}
    if kind == "spike":
        spec["offset"] = 10
    if kind == "drift":
        spec["rate"] = 0.5
    rows = [{"x": float(step)} for step in range(5)]
    a = FaultInjector(FaultConfig(enabled=True, seed=9, faults=[spec]))
    b = FaultInjector(FaultConfig(enabled=True, seed=9, faults=[spec]))
    got_a = [a.apply(row, step)[0] for step, row in enumerate(rows)]
    got_b = [b.apply(row, step)[0] for step, row in enumerate(rows)]
    assert got_a == got_b
    assert all(math.isfinite(item["x"]) for item in got_a)
    if kind == "stuck":
        assert [item["x"] for item in got_a] == [0.0, 0.0, 0.0, 0.0, 4.0]


def test_disabled_is_pass_through_and_nan_reaches_agent():
    row = {"battery_soc_pct": 42.0}
    observed, labels = FaultInjector(FaultConfig()).apply(row, 0)
    assert observed == row and labels == []
    nan_row, _ = FaultInjector(FaultConfig(enabled=True, faults=[
        {"type": "nan", "key": "battery_soc_pct", "start_step": 0, "end_step": 1}
    ])).apply(row, 0)
    agent = BatteryAgent("battery")
    agent.update_state(nan_row)
    with pytest.raises(ValueError):
        agent.get_bid()


def test_dataset_columns_probability_and_csv(tmp_path):
    rows = pd.DataFrame({"battery_soc_pct": [50.0] * 8})
    config = FaultConfig(enabled=True, seed=2, faults=[
        {"type": "spike", "key": "battery_soc_pct", "probability": 1.0,
         "duration_range": (1, 1), "offset": 1}
    ])
    path = tmp_path / "faults.csv"
    result = generate_fault_dataset(rows, config, path)
    assert len(result) == 8
    assert list(result.columns) == ["step", "battery_soc_pct_clean",
                                    "battery_soc_pct_observed", "battery_soc_pct_fault"]
    assert set(result["battery_soc_pct_fault"]) == {"spike"}
    assert pd.read_csv(path).equals(result)
