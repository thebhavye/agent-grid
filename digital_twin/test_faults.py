import math

import pandas as pd
import pytest

from agents.battery_agent import BatteryAgent
from digital_twin.faults import (
    DeviceFaultConfig,
    DeviceFaultInjector,
    FaultConfig,
    FaultInjector,
    generate_fault_dataset,
)


def test_fault_type_alias_is_canonicalized():
    injector = FaultInjector(FaultConfig(enabled=True, faults=[
        {"fault_type": "spike", "key": "x", "start_step": 0, "end_step": 1,
         "value": 150},
    ]))
    observed, labels = injector.apply({"x": 10.0}, 0)
    assert observed["x"] == 150
    assert labels[0]["fault_type"] == "spike"


@pytest.mark.parametrize("duration_key", ["duration_range", "duration_steps"])
def test_scalar_probability_duration_is_accepted(duration_key):
    injector = FaultInjector(FaultConfig(enabled=True, faults=[
        {"type": "spike", "key": "x", "probability": 1,
         duration_key: 2, "offset": 10},
    ]))
    assert injector.apply({"x": 1.0}, 0)[0]["x"] == 11.0


def test_stuck_from_first_step_freezes_initial_reading():
    result = generate_fault_dataset(
        [{"x": float(step)} for step in range(4)],
        FaultConfig(enabled=True, faults=[
            {"type": "stuck", "key": "x", "start_step": 0, "end_step": 3},
        ]),
    )
    assert result["x_observed"].tolist() == [0.0, 0.0, 0.0, 3.0]
    assert result["x_fault"].tolist() == ["stuck", "stuck", "stuck", "none"]


def test_probability_faults_continue_beyond_old_horizon():
    result = generate_fault_dataset(
        ({"x": 1.0} for _ in range(10002)),
        FaultConfig(enabled=True, faults=[
            {"type": "spike", "key": "x", "probability": 1,
             "duration_range": (1, 1), "offset": 10},
        ]),
    )
    assert (result["x_observed"] == 11.0).all()
    assert (result["x_fault"] == "spike").all()


@pytest.mark.parametrize("extra", [
    {"start_step": 0.5, "end_step": 2},
    {"start_step": 0, "end_step": float("inf")},
    {"probability": 1, "duration_range": (1.5, 2)},
    {"probability": 1, "duration_range": None},
    {"start_step": 0, "end_step": 2, "offset": float("nan")},
    {"start_step": 0, "end_step": 2, "offset": float("inf")},
])
def test_invalid_fault_parameters_fail_at_configuration(extra):
    with pytest.raises(ValueError):
        FaultConfig(enabled=True, faults=[{"type": "spike", "key": "x", **extra}])


@pytest.mark.parametrize("kind", ["dropout", "nan", "stuck", "spike", "drift"])
def test_half_open_boundaries_and_label_values(kind):
    rows = [{"x": float(step)} for step in range(4)]
    injector = FaultInjector(FaultConfig(enabled=True, faults=[
        {"type": kind, "key": "x", "start_step": 1, "end_step": 3},
    ]))
    for step, row in enumerate(rows):
        observed, labels = injector.apply(row, step)
        assert row == {"x": float(step)}
        if step in (0, 3):
            assert observed == row and labels == []
        else:
            assert labels[0]["true_value"] == float(step)
            assert labels[0]["fault_type"] == kind
            if kind == "dropout":
                assert "x" not in observed and labels[0]["observed_value"] is None
            elif kind == "nan":
                assert math.isnan(observed["x"]) and math.isnan(labels[0]["observed_value"])
            else:
                assert labels[0]["observed_value"] == observed["x"]
                assert math.isfinite(observed["x"])
                if kind == "stuck":
                    assert observed["x"] == 0.0
                elif kind == "drift":
                    assert observed["x"] == float(step + step)
                else:
                    assert observed["x"] == float(step + 1)


def test_probabilistic_day_is_deterministic_without_changing_source():
    rows = [{"x": float(step), "y": 50.0} for step in range(288)]
    original = [dict(row) for row in rows]
    specs = [
        {"type": "drift", "key": "x", "probability": 0.4,
         "duration_range": (2, 4), "rate": 0.5},
        {"type": "nan", "key": "y", "probability": 0.2,
         "duration_range": (1, 3)},
    ]
    config = FaultConfig(enabled=True, seed=42, faults=specs)
    first = generate_fault_dataset(rows, config)
    second = generate_fault_dataset(rows, config)
    pd.testing.assert_frame_equal(first, second)
    assert rows == original
    assert set(first["x_fault"]) == {"none", "drift"}
    assert set(first["y_fault"]) == {"none", "nan"}
    assert not first.loc[first["y_fault"] == "none", "y_observed"].isna().any()
    assert first.loc[first["y_fault"] == "nan", "y_observed"].isna().all()
    assert all("start_step" not in spec for spec in specs)


def test_disabled_pass_through_for_a_complete_twin_day():
    from digital_twin.test_digital_twin import make_test_twin

    rows = make_test_twin(seed=42).run_full_day()
    injector = FaultInjector(FaultConfig())
    for step, row in enumerate(rows):
        observed, labels = injector.apply(row, step)
        assert observed == row and observed is not row
        assert labels == []


def test_backward_steps_are_rejected_without_restarting_fault_stream():
    injector = FaultInjector(FaultConfig(enabled=True, faults=[
        {"type": "spike", "key": "x", "probability": 1,
         "duration_range": (1, 1)},
    ]))
    injector.apply({"x": 1.0}, 1)
    with pytest.raises(ValueError, match="chronological"):
        injector.apply({"x": 1.0}, 0)
    assert injector.apply({"x": 1.0}, 2)[0] == {"x": 2.0}


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


def test_device_faults_disabled_are_neutral_and_pass_through():
    injector = DeviceFaultInjector(DeviceFaultConfig())
    effects, labels = injector.apply(0)
    assert effects == {"solar_scale": 1.0, "ev_limit_kw": None, "battery_factor": 1.0}
    assert labels == []


def test_device_fault_effects_are_windowed_labelled_and_deterministic():
    faults = [
        {"type": "inverter_trip", "start_step": 0, "end_step": 2},
        {"type": "battery_derate", "start_step": 1, "end_step": 3, "factor": 0.25},
        {"type": "charger_fault", "start_step": 2, "end_step": 4},
    ]

    def run():
        injector = DeviceFaultInjector(DeviceFaultConfig(enabled=True, faults=faults))
        return [injector.apply(step) for step in range(5)]

    first, second = run(), run()
    assert first == second

    assert first[0][0] == {"solar_scale": 0.0, "ev_limit_kw": None,
                           "battery_factor": 1.0}
    assert first[1][0] == {"solar_scale": 0.0, "ev_limit_kw": None,
                           "battery_factor": 0.25}
    assert first[2][0] == {"solar_scale": 1.0, "ev_limit_kw": 0.0,
                           "battery_factor": 0.25}
    assert first[3][0] == {"solar_scale": 1.0, "ev_limit_kw": 0.0,
                           "battery_factor": 1.0}
    assert first[4][0] == {"solar_scale": 1.0, "ev_limit_kw": None,
                           "battery_factor": 1.0}
    assert [label["fault_type"] for label in first[2][1]] == ["battery_derate",
                                                              "charger_fault"]
    assert first[2][1][0]["factor"] == 0.25


def test_device_fault_overlapping_derate_uses_the_strongest_factor():
    injector = DeviceFaultInjector(DeviceFaultConfig(enabled=True, faults=[
        {"type": "battery_derate", "start_step": 0, "end_step": 1, "factor": 0.8},
        {"type": "battery_derate", "start_step": 0, "end_step": 1, "factor": 0.3},
    ]))
    effects, labels = injector.apply(0)
    assert effects["battery_factor"] == 0.3
    assert len(labels) == 2


@pytest.mark.parametrize("spec", [
    {"type": "not_a_device_fault", "start_step": 0, "end_step": 1},
    {"type": "battery_derate", "start_step": 0, "end_step": 1, "factor": 0},
    {"type": "battery_derate", "start_step": 0, "end_step": 1, "factor": 1.5},
    {"type": "inverter_trip"},
    {"type": "inverter_trip", "start_step": 0, "end_step": 1, "probability": 0.5},
])
def test_invalid_device_fault_parameters_fail_at_configuration(spec):
    with pytest.raises(ValueError):
        DeviceFaultConfig(enabled=True, faults=[spec])


def test_device_fault_probability_is_seeded_and_chronological():
    config = DeviceFaultConfig(enabled=True, seed=7, faults=[
        {"type": "charger_fault", "probability": 0.5, "duration_range": (1, 2)},
    ])
    a = DeviceFaultInjector(config)
    b = DeviceFaultInjector(config)
    assert [a.apply(step) for step in range(10)] == [b.apply(step) for step in range(10)]

    injector = DeviceFaultInjector(config)
    injector.apply(1)
    with pytest.raises(ValueError, match="chronological"):
        injector.apply(0)
