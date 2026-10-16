"""Regression and invariant tests for the closed-loop runner."""

import json
import math

import pytest

from digital_twin.appliance_model import ApplianceModelConfig
from digital_twin.battery_model import BatteryModelConfig
from digital_twin.ev_model import EVModelConfig, EVScenario
from digital_twin.faults import (
    DeviceFaultConfig,
    DeviceFaultInjector,
    FaultConfig,
    FaultInjector,
)
from digital_twin.test_digital_twin import make_test_twin
from simulation.baseline_allocator import allocate
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate


ALLOCATORS = [hybrid_allocate, proportional_adapter]


def _assert_power_balance_and_exclusivity(records):
    """Shared fault-run invariant check: balance + charge/discharge exclusivity."""
    for record in records:
        loads = sum(
            value
            for agent_id, value in record["allocated_kw"].items()
            if agent_id not in {"solar_1", "battery_1"}
        )
        battery_kw = record["battery_kw"]
        assert not (battery_kw > 1e-9 and battery_kw < -1e-9)
        for values in record["agents"].values():
            assert 0 <= values["allocated_kw"] <= values["demand_kw"] + 1e-12
        assert record["solar_kw"] + max(0.0, -battery_kw) == pytest.approx(
            loads + max(0.0, battery_kw) + record["curtailed_kw"],
            abs=1e-9,
        )
        if record["mode"] == "discharge":
            assert record["curtailed_kw"] == pytest.approx(0.0, abs=1e-9)



@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize("scenario", list(EVScenario))
@pytest.mark.parametrize("seed", [1, 7, 42])
def test_closed_loop_runs_all_scenarios_and_preserves_invariants(
    allocator, scenario, seed
):
    twin = make_test_twin(
        seed=seed,
        ev_config=EVModelConfig(scenario=scenario, seed=seed),
        appliance_config=ApplianceModelConfig(seed=seed),
        closed_loop=True,
    )
    records = run_closed_loop(twin, allocator)

    assert len(records) == 288
    assert twin.step_count == 288
    previous_battery_soc = twin._battery.config.initial_soc_pct
    for record in records:
        agents = record["agents"]
        assert set(record["demand_kw"]) == set(agents)
        assert set(record["allocated_kw"]) == set(agents)
        for agent_id, values in agents.items():
            assert math.isfinite(values["demand_kw"])
            assert 0 <= values["allocated_kw"] <= values["demand_kw"] + 1e-12

        loads = sum(
            value
            for agent_id, value in record["allocated_kw"].items()
            if agent_id not in {"solar_1", "battery_1"}
        )
        battery_kw = record["battery_kw"]
        assert record["ev_applied_kw"] == pytest.approx(record["allocated_kw"]["ev_1"])
        assert loads == pytest.approx(
            sum(record["appliance_served_kw"].values()) + record["ev_applied_kw"]
            + record["allocated_kw"]["household_base_load"]
        )
        cfg = twin._battery.config
        delta_kwh = battery_kw * record["dt_min"] / 60
        delta_kwh *= cfg.charge_efficiency if battery_kw >= 0 else 1 / cfg.discharge_efficiency
        assert record["battery_soc_pct"] == pytest.approx(
            previous_battery_soc + delta_kwh / cfg.capacity_kwh * 100, abs=1e-9
        )
        assert record["battery_soc_pct"] >= 20 - 1e-9
        previous_battery_soc = record["battery_soc_pct"]
        assert record["solar_kw"] + max(0.0, -battery_kw) == pytest.approx(
            loads + max(0.0, battery_kw) + record["curtailed_kw"],
            abs=1e-9,
        )
        if record["mode"] == "discharge":
            assert record["curtailed_kw"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_closed_loop_is_deterministic(allocator):
    def run():
        twin = make_test_twin(seed=42, closed_loop=True)
        return run_closed_loop(twin, allocator)

    assert run() == run()


def test_proportional_adapter_matches_baseline_allocator():
    bids = [
        {"agent_id": "a", "demand_kw": 2.0, "urgency": 0.1},
        {"agent_id": "b", "demand_kw": 3.0, "urgency": 0.9},
    ]
    priorities = {"a": 1, "b": 5}
    assert proportional_adapter(2.0, bids, priorities) == allocate(
        2.0, {"a": 2.0, "b": 3.0}
    )


def test_closed_loop_discharges_at_night_and_reserve_stops_supply():
    active = make_test_twin(closed_loop=True)
    active_record = run_closed_loop(active, hybrid_allocate, steps=1)[0]
    assert active_record["mode"] == "discharge"
    assert active_record["battery_kw"] < 0
    assert active_record["allocated_kw"]["appliance_fridge"] > 0

    reserve = make_test_twin(
        closed_loop=True,
        battery_config=BatteryModelConfig(initial_soc_pct=20.0),
    )
    reserve_record = run_closed_loop(reserve, hybrid_allocate, steps=1)[0]
    assert reserve_record["mode"] == "idle"
    assert reserve_record["battery_kw"] == 0.0
    assert reserve_record["allocated_kw"]["appliance_fridge"] == 0.0


def test_closed_loop_overnight_ev_receives_evening_power():
    twin = make_test_twin(
        seed=1,
        ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=1),
        closed_loop=True,
    )
    records = run_closed_loop(twin, hybrid_allocate)
    evening = [
        record for record in records
        if record["timestamp"] >= "19:00"
        and record["ev_plugged_in"]
    ]
    assert evening
    assert any(record["ev_applied_kw"] > 0 for record in evening)
    assert all("ev_charger" not in record["appliances"] for record in records)


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize("base_kw", [0.0, 0.6, 1.2])
def test_base_load_is_bid_and_included_in_actual_power_balance(allocator, base_kw):
    twin = make_test_twin(
        closed_loop=True,
        ev_config=EVModelConfig(scenario=EVScenario.MIDDAY),
        appliance_config=ApplianceModelConfig(appliances=[], base_load_kw=base_kw),
    )
    before = twin._battery.available_energy_kwh
    record = run_closed_loop(twin, allocator, steps=1)[0]
    assert record["demand_kw"]["household_base_load"] == base_kw
    assert record["allocated_kw"]["household_base_load"] == pytest.approx(base_kw)
    assert record["battery_kw"] == pytest.approx(-base_kw)
    assert before - twin._battery.available_energy_kwh == pytest.approx(
        base_kw * record["dt_min"] / 60 / twin._battery.config.discharge_efficiency
    )


@pytest.mark.parametrize("dt", [0.25, 5.0, 15.0, 60.0])
def test_supply_does_not_round_soc_above_reserve(dt):
    twin = make_test_twin(
        closed_loop=True, time_delta_min=dt,
        battery_config=BatteryModelConfig(initial_soc_pct=20.006),
    )
    run_closed_loop(twin, hybrid_allocate, steps=1)
    assert twin._battery.soc_pct >= 20.0 - 1e-10


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_ev_delivery_is_capped_by_physical_headroom(allocator):
    twin = make_test_twin(
        closed_loop=True,
        ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=1,
                               battery_capacity_kwh=0.01),
    )
    before = twin._ev.soc_pct
    record = run_closed_loop(twin, allocator, steps=1)[0]
    actual_kw = ((twin._ev.soc_pct - before) / 100
                 * twin._ev.config.battery_capacity_kwh
                 / twin._ev.config.charge_efficiency / (record["dt_min"] / 60))
    assert record["allocated_kw"]["ev_1"] == pytest.approx(actual_kw)
    assert record["ev_applied_kw"] == pytest.approx(actual_kw)


def test_closed_loop_requires_fresh_closed_loop_twin():
    with pytest.raises(ValueError, match="closed_loop=True"):
        run_closed_loop(make_test_twin(), hybrid_allocate, steps=0)

    twin = make_test_twin(closed_loop=True)
    twin.tick()
    with pytest.raises(ValueError, match="fresh"):
        run_closed_loop(twin, hybrid_allocate, steps=1)


SENSOR_FAULTS = FaultConfig(enabled=True, seed=3, faults=[
    {"type": "nan", "key": "battery_soc_pct", "start_step": 100, "end_step": 104},
    {"type": "dropout", "key": "solar_gen_kw", "start_step": 200, "end_step": 202},
    {"type": "spike", "key": "battery_soc_pct", "start_step": 300, "end_step": 302,
     "offset": 30},
])


def test_raise_policy_keeps_agents_fail_fast_on_nan_soc():
    config = FaultConfig(enabled=True, faults=[
        {"type": "nan", "key": "battery_soc_pct", "start_step": 0, "end_step": 1},
    ])
    with pytest.raises(ValueError, match="battery_soc_pct"):
        run_closed_loop(
            make_test_twin(closed_loop=True), hybrid_allocate, steps=1,
            sensor_faults=config,
        )


def test_invalid_fault_policy_and_hold_bound_are_rejected():
    twin = make_test_twin(closed_loop=True)
    with pytest.raises(ValueError, match="on_bad_reading"):
        run_closed_loop(twin, hybrid_allocate, steps=0, on_bad_reading="ignore")
    with pytest.raises(ValueError, match="max_hold_steps"):
        run_closed_loop(
            make_test_twin(closed_loop=True), hybrid_allocate, steps=0,
            on_bad_reading="hold_last", max_hold_steps=-1,
        )


def test_hold_last_completes_full_run_and_logs_every_substitution():
    twin = make_test_twin(closed_loop=True)
    records = run_closed_loop(
        twin, hybrid_allocate, on_bad_reading="hold_last", sensor_faults=SENSOR_FAULTS
    )

    assert len(records) == 288
    assert twin.step_count == 288
    _assert_power_balance_and_exclusivity(records)

    substitutions = [entry for record in records for entry in record["substitutions"]]
    assert substitutions
    for entry in substitutions:
        assert set(entry) == {"agent_id", "key", "policy", "observed", "used"}
        assert entry["policy"] in {"hold_last", "exclude"}
    logged = {(entry["agent_id"], entry["key"], entry["policy"]) for entry in substitutions}
    assert ("battery_1", "battery_soc_pct", "hold_last") in logged
    assert ("solar_1", "solar_gen_kw", "hold_last") in logged
    # The finite spike is a labelled sensor fault but not a runner substitution.
    assert any(record["sensor_faults"] for record in records)


def test_exclude_policy_records_true_demand_while_bidding_zero():
    config = FaultConfig(enabled=True, faults=[
        {"type": "nan", "key": "battery_soc_pct", "start_step": 10, "end_step": 14},
    ])
    clean = run_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate)
    twin = make_test_twin(closed_loop=True)
    records = run_closed_loop(
        twin, hybrid_allocate, on_bad_reading="exclude", sensor_faults=config
    )
    _assert_power_balance_and_exclusivity(records)

    faulted = records[10]
    assert faulted["allocated_kw"]["battery_1"] == 0.0
    assert faulted["excluded_agents"] == ["battery_1"]
    # The clean demand is still recorded (unserved), not erased by the
    # zero allocator bid.
    assert faulted["demand_kw"]["battery_1"] == pytest.approx(
        clean[10]["demand_kw"]["battery_1"]
    )
    assert faulted["demand_kw"]["battery_1"] > 0.0
    assert faulted["unserved_kw"]["battery_1"] == pytest.approx(
        faulted["demand_kw"]["battery_1"]
    )
    assert any(
        entry["policy"] == "exclude" and entry["agent_id"] == "battery_1"
        for entry in faulted["substitutions"]
    )
    assert records[9]["excluded_agents"] == []


def test_exclude_records_unserved_ev_demand_from_clean_value():
    config = FaultConfig(enabled=True, faults=[
        {"type": "nan", "key": "ev_soc_pct", "start_step": 0, "end_step": 1},
    ])
    clean = run_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate, steps=1)
    twin = make_test_twin(closed_loop=True)
    records = run_closed_loop(
        twin, hybrid_allocate, steps=1, on_bad_reading="exclude", sensor_faults=config
    )
    record = records[0]
    assert record["excluded_agents"] == ["ev_1"]
    assert record["allocated_kw"]["ev_1"] == 0.0
    assert record["demand_kw"]["ev_1"] == pytest.approx(clean[0]["demand_kw"]["ev_1"])
    assert record["demand_kw"]["ev_1"] > 0.0
    assert record["unserved_kw"]["ev_1"] == pytest.approx(record["demand_kw"]["ev_1"])


def test_raise_policy_rejects_missing_and_malformed_watched_readings():
    dropout = FaultConfig(enabled=True, faults=[
        {"type": "dropout", "key": "battery_soc_pct", "start_step": 0, "end_step": 1},
    ])
    with pytest.raises(ValueError, match="battery_soc_pct"):
        run_closed_loop(
            make_test_twin(closed_loop=True), hybrid_allocate, steps=1,
            sensor_faults=dropout,
        )

    malformed = FaultConfig(enabled=True, faults=[
        {"type": "spike", "key": "timestamp", "start_step": 0, "end_step": 1,
         "value": 123},
    ])
    with pytest.raises(ValueError, match="timestamp"):
        run_closed_loop(
            make_test_twin(closed_loop=True), hybrid_allocate, steps=1,
            sensor_faults=malformed,
        )


def test_hold_last_near_reserve_never_crosses_reserve():
    faults = [
        {"type": "nan", "key": "battery_soc_pct", "start_step": 13, "end_step": 16},
    ]
    for allocator in ALLOCATORS:
        config = FaultConfig(enabled=True, faults=faults)
        records = run_closed_loop(
            make_test_twin(closed_loop=True), allocator, steps=20,
            on_bad_reading="hold_last", sensor_faults=config,
        )
        _assert_power_balance_and_exclusivity(records)
        assert min(record["battery_soc_pct"] for record in records) >= 20.0 - 1e-9


def test_hold_last_ev_departure_transition_cannot_charge_unplugged():
    config = FaultConfig(enabled=True, faults=[
        {"type": "nan", "key": "ev_plugged_in", "start_step": 93, "end_step": 96},
    ])
    records = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate, steps=100,
        on_bad_reading="hold_last", sensor_faults=config,
    )
    _assert_power_balance_and_exclusivity(records)
    for record in records[93:96]:
        assert record["ev_plugged_in"] is False
        assert record["ev_applied_kw"] == 0.0


def test_hold_last_appliance_demand_drop_transition_completes():
    config = FaultConfig(enabled=True, faults=[
        {"type": "nan", "key": "appliance_desired_kw", "start_step": 109,
         "end_step": 112},
    ])
    records = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate, steps=120,
        on_bad_reading="hold_last", sensor_faults=config,
    )
    _assert_power_balance_and_exclusivity(records)
    assert len(records) == 120


def test_hold_last_malformed_timestamp_uses_previous_valid_reading():
    config = FaultConfig(enabled=True, faults=[
        {"type": "spike", "key": "timestamp", "start_step": 5, "end_step": 7,
         "value": 123},
    ])
    records = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate, steps=10,
        on_bad_reading="hold_last", sensor_faults=config,
    )
    _assert_power_balance_and_exclusivity(records)
    assert any(
        entry["key"] == "timestamp" and entry["policy"] == "hold_last"
        for record in records[5:7] for entry in record["substitutions"]
    )


def test_hold_last_falls_back_to_exclude_after_max_steps():
    config = FaultConfig(enabled=True, faults=[
        {"type": "nan", "key": "battery_soc_pct", "start_step": 10, "end_step": 20},
    ])
    twin = make_test_twin(closed_loop=True)
    records = run_closed_loop(
        twin, hybrid_allocate, on_bad_reading="hold_last", max_hold_steps=2,
        sensor_faults=config,
    )

    def battery_policies(index):
        return [
            entry["policy"] for entry in records[index]["substitutions"]
            if entry["agent_id"] == "battery_1"
        ]

    assert battery_policies(10) == ["hold_last"]
    assert battery_policies(11) == ["hold_last"]
    assert battery_policies(12) == ["exclude"]
    assert records[12]["excluded_agents"] == ["battery_1"]


def test_fault_features_off_leave_records_identical():
    plain = run_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate)
    disabled = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate,
        sensor_faults=FaultConfig(), device_faults=DeviceFaultConfig(),
    )
    assert plain == disabled


def test_inverter_trip_zeroes_delivered_solar_and_battery_covers_load():
    config = DeviceFaultConfig(enabled=True, faults=[
        {"type": "inverter_trip", "start_step": 144, "end_step": 150},
    ])
    baseline = run_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate)
    faulted = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate, device_faults=config
    )
    _assert_power_balance_and_exclusivity(faulted)

    assert baseline[144]["solar_kw"] > 0
    trip = faulted[144]
    assert trip["solar_kw"] == 0.0
    assert trip["mode"] == "discharge"
    assert trip["battery_kw"] < 0
    assert {"step": 144, "device": "solar", "fault_type": "inverter_trip"} in (
        trip["device_faults"]
    )
    assert faulted[150]["device_faults"] == []


def test_charger_fault_zeroes_ev_delivery_while_plugged():
    baseline = run_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate)
    target = next(
        index for index, record in enumerate(baseline)
        if record["ev_plugged_in"] and record["ev_applied_kw"] > 0
    )
    config = DeviceFaultConfig(enabled=True, faults=[
        {"type": "charger_fault", "start_step": target, "end_step": target + 1},
    ])
    faulted = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate, device_faults=config
    )

    assert faulted[target]["ev_plugged_in"] is True
    assert faulted[target]["ev_applied_kw"] == 0.0
    assert {"step": target, "device": "ev", "fault_type": "charger_fault"} in (
        faulted[target]["device_faults"]
    )


def test_battery_derate_caps_charge_and_discharge_power():
    baseline = run_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate)
    target = next(
        index for index, record in enumerate(baseline)
        if record["mode"] == "discharge" and abs(record["battery_kw"]) > 1.5
    )
    config = DeviceFaultConfig(enabled=True, faults=[
        {"type": "battery_derate", "start_step": target, "end_step": target + 1,
         "factor": 0.1},
    ])
    faulted = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate, device_faults=config
    )
    _assert_power_balance_and_exclusivity(faulted)

    derated = faulted[target]
    assert derated["mode"] == "discharge"
    assert derated["battery_kw"] == pytest.approx(-1.2, abs=1e-9)  # 0.1 * 12 kW
    assert abs(derated["battery_kw"]) < abs(baseline[target]["battery_kw"])
    assert derated["device_faults"][0]["fault_type"] == "battery_derate"


def test_device_faults_are_deterministic():
    faults = [
        {"type": "inverter_trip", "start_step": 144, "end_step": 150},
        {"type": "battery_derate", "start_step": 10, "end_step": 15, "factor": 0.1},
    ]

    def run():
        return run_closed_loop(
            make_test_twin(closed_loop=True), hybrid_allocate,
            device_faults=DeviceFaultConfig(enabled=True, faults=faults),
        )

    assert run() == run()


def test_held_dropout_substitution_logs_are_serializable():
    config = FaultConfig(enabled=True, faults=[
        {"type": "dropout", "key": "battery_soc_pct", "start_step": 10, "end_step": 14},
    ])
    records = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate,
        on_bad_reading="hold_last", sensor_faults=config,
    )
    entries = [
        entry for record in records for entry in record["substitutions"]
        if entry["key"] == "battery_soc_pct"
    ]
    assert any(entry["policy"] == "hold_last" for entry in entries)
    # A held dropout has no observed value; the private sentinel must not
    # reach the exported record. json.dumps fails on the raw object.
    assert all(entry["observed"] is None for entry in entries)
    json.dumps(records)


def _stuck_one_step_run(start_soc, sensor_faults):
    twin = make_test_twin(
        closed_loop=True,
        battery_config=BatteryModelConfig(initial_soc_pct=start_soc),
    )
    return run_closed_loop(
        twin, hybrid_allocate, steps=1,
        on_bad_reading="hold_last", sensor_faults=sensor_faults,
    )


def test_reused_sensor_injector_starts_each_run_from_its_config():
    spec = {"type": "stuck", "key": "battery_soc_pct", "start_step": 0, "end_step": 1}
    reused = FaultInjector(FaultConfig(enabled=True, faults=[spec]))
    fresh_first = _stuck_one_step_run(50.0, FaultConfig(enabled=True, faults=[spec]))
    fresh_second = _stuck_one_step_run(80.0, FaultConfig(enabled=True, faults=[spec]))
    # The two scenarios must differ, or the equality checks are vacuous.
    assert (
        fresh_first[0]["demand_kw"]["battery_1"]
        != fresh_second[0]["demand_kw"]["battery_1"]
    )

    assert _stuck_one_step_run(50.0, reused) == fresh_first
    # A leaked last-good SOC would make this run bid like the first (50%).
    assert _stuck_one_step_run(80.0, reused) == fresh_second


def test_reused_sensor_injector_after_full_run_restarts_cleanly():
    # A dropout label carries None/finite values, so full-record equality is
    # meaningful (a nan fault would embed NaN and never compare equal).
    spec = {"type": "dropout", "key": "battery_soc_pct", "start_step": 100, "end_step": 104}
    reused = FaultInjector(FaultConfig(enabled=True, seed=3, faults=[spec]))
    first = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate,
        on_bad_reading="hold_last", sensor_faults=reused,
    )
    fresh = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate,
        on_bad_reading="hold_last",
        sensor_faults=FaultConfig(enabled=True, seed=3, faults=[spec]),
    )
    assert first == fresh
    # No chronological-order error or stale last-good from the first run.
    again = run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate,
        on_bad_reading="hold_last", sensor_faults=reused,
    )
    assert again == fresh


def test_reused_device_injector_starts_each_run_from_its_config():
    spec = {"type": "inverter_trip", "start_step": 5, "end_step": 6}
    reused = DeviceFaultInjector(DeviceFaultConfig(enabled=True, faults=[spec]))

    def run(device_faults):
        return run_closed_loop(
            make_test_twin(closed_loop=True), hybrid_allocate, steps=10,
            device_faults=device_faults,
        )

    fresh = run(DeviceFaultConfig(enabled=True, faults=[spec]))
    assert run(reused) == fresh
    # The second reuse must not hit leftover chronology state.
    assert run(reused) == fresh
