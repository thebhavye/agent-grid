"""Known-answer outcome metrics and observational runner event tests."""

from copy import deepcopy
from datetime import datetime

import pytest

from digital_twin.appliance_model import ApplianceSpec
from simulation import closed_loop
from simulation.compare import _make_twin
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.metrics import METRICS, score_run


def record(**overrides):
    value = {
        "dt_min": 15.0, "solar_kw": 4.0, "battery_kw": 1.0,
        "curtailed_kw": 0.5, "battery_soc_pct": 50.0, "battery_reserve_pct": 20.0,
        "agents": {
            "household_base_load": {"demand_kw": 2.0, "allocated_kw": 1.0},
            "appliance_light": {"demand_kw": 2.0, "allocated_kw": 1.0},
            "ev_1": {"demand_kw": 2.0, "allocated_kw": 0.5},
            "battery_1": {"demand_kw": 6.0, "allocated_kw": 1.0},
            "solar_1": {"demand_kw": 0.0, "allocated_kw": 0.0},
        },
        "priorities": {"household_base_load": 1, "appliance_light": 2, "ev_1": 3},
        "ev_applied_kw": 0.5, "appliance_served_kw": {"light": 1.0},
        "ev_departures": [{"target_soc_pct": 80.0, "soc_pct": 75.0}],
        "appliance_cycles_completed": ["dishwasher"],
        "appliance_cycles_incomplete_at_window_end": ["washer"],
        "blocked_starts": {"washer": 2},
    }
    value.update(overrides)
    return value


def test_known_answer_non_five_minute_and_actual_power_balance():
    records = [record(), record(dt_min=30, battery_kw=-1.0, solar_kw=2.0,
                               battery_soc_pct=19.0,
                               ev_departures=[{"target_soc_pct": 80, "soc_pct": 90}])]
    before = deepcopy(records)
    assert score_run(records) == pytest.approx({
        "critical_unmet_kwh": 1.5, "priority_1_unmet_kwh": 0.75,
        "flexible_unmet_kwh": 1.125, "ev_shortfall_pct_at_departure": 5.0,
        "appliance_cycles_completed": 2, "appliance_cycles_incomplete_at_window_end": 2,
        "blocked_starts": 4, "curtailed_solar_kwh": 0.375,
        "battery_reserve_violation_steps": 1, "battery_charge_kwh": 0.25,
        "battery_discharge_kwh": 0.5, "energy_balance_max_error_kw": 0.0,
    })
    assert records == before


def test_balance_uses_actual_ev_and_appliance_service_without_double_counting():
    assert score_run([record(solar_kw=4.125)])["energy_balance_max_error_kw"] == 0.125


def test_empty_run_and_no_departure():
    assert score_run([]) == dict.fromkeys(METRICS, 0)
    assert score_run([record(ev_departures=[])])["ev_shortfall_pct_at_departure"] == 0


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_timestep(dt):
    with pytest.raises(ValueError, match="dt_min"):
        score_run([record(dt_min=dt)])


def test_nonfinite_balance_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        score_run([record(solar_kw=float("nan"))])


def test_overnight_departure_uses_segment_target_and_last_committed_soc():
    twin = _make_twin(42, "overnight")
    records = closed_loop.run_closed_loop(twin, hybrid_allocate)
    events = [(i, event) for i, row in enumerate(records) for event in row["ev_departures"]]
    assert len(events) == 1
    index, event = events[0]
    assert records[index]["ev_plugged_in"]
    assert not records[index + 1]["ev_plugged_in"]
    assert event["target_soc_pct"] == twin._ev._segments[0].target_soc
    assert event["soc_pct"] == records[index]["ev_soc_pct"]
    assert records[-1]["ev_plugged_in"]
    assert records[-1]["ev_departures"] == []  # Midnight is continuously plugged.
    assert score_run(records)["ev_shortfall_pct_at_departure"] == pytest.approx(
        max(0, event["target_soc_pct"] - event["soc_pct"])
    )


@pytest.mark.parametrize("runtime,expected_missed", [(8, 0), (12, 1), (20, 1)])
def test_deadline_inside_interval_and_completion_are_separate(runtime, expected_missed):
    twin = _make_twin(42, "midday")
    twin._appliances.config.appliances = [
        ApplianceSpec("washer", 4, True, 1, 0, 10, cycle_prob_per_hour=0)
    ]
    twin._appliances._appliance_states["washer"] = {
        "is_on": True, "time_remaining_min": runtime,
    }
    twin._appliances._blocked_starts["washer"] = 0
    states = {"washer": dict(twin._appliances.get_appliance_state("washer"))}
    twin._appliances.get_appliance_state("washer").update(
        is_on=runtime > 15, time_remaining_min=max(0, runtime - 15)
    )
    result = closed_loop._outcome_metadata(
        twin, datetime(2024, 1, 15), 15, states, {"washer": 0}, {}, {}, 20,
        {"appliance_desired_kw": {"washer": 1}},
        {"appliance_served_kw": {"washer": 1}},
    )
    assert len(result["appliance_cycles_incomplete_at_window_end"]) == expected_missed
    assert len(result["appliance_cycles_completed"]) == int(runtime <= 15)


def test_outcome_metadata_does_not_change_existing_records_or_rng(monkeypatch):
    twin = _make_twin(7, "overnight")
    records = closed_loop.run_closed_loop(twin, hybrid_allocate)
    monkeypatch.setattr(closed_loop, "_outcome_metadata", lambda *args: {})
    other = _make_twin(7, "overnight")
    without = closed_loop.run_closed_loop(other, hybrid_allocate)
    assert [{key: row[key] for key in plain} for row, plain in zip(records, without)] == without
    assert twin._appliances.rng_draw_count == other._appliances.rng_draw_count
    assert sum(sum(row["blocked_starts"].values()) for row in records) == sum(
        value for key, value in twin._appliances.blocked_starts.items() if key != "ev_charger"
    )
    for key, drawn in twin._appliances.drawn_cycle_start_times.items():
        assert [time for row in records for time in row["drawn_cycle_starts"][key]] == drawn


def test_non_five_minute_runner_records_are_scorable():
    twin = _make_twin(42, "midday")
    twin._time_delta_min = 15
    records = closed_loop.run_closed_loop(twin, hybrid_allocate, steps=96)
    assert all(row["dt_min"] == 15 for row in records)
    assert score_run(records)["energy_balance_max_error_kw"] < 1e-6
