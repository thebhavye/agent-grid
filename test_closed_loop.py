"""Regression and invariant tests for the closed-loop runner."""

import math

import pytest

from digital_twin.appliance_model import ApplianceModelConfig
from digital_twin.battery_model import BatteryModelConfig
from digital_twin.ev_model import EVModelConfig, EVScenario
from digital_twin.test_digital_twin import make_test_twin
from simulation.baseline_allocator import allocate
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate


ALLOCATORS = [hybrid_allocate, proportional_adapter]


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
