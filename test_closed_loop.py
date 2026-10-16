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
        assert not (battery_kw > 1e-9 and battery_kw < -1e-9)
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
        and record["appliances"]["ev_charger"]["desired_kw"] > 0
    ]
    assert evening
    assert any(record["appliances"]["ev_charger"]["served_kw"] > 0 for record in evening)


def test_closed_loop_requires_fresh_closed_loop_twin():
    with pytest.raises(ValueError, match="closed_loop=True"):
        run_closed_loop(make_test_twin(), hybrid_allocate, steps=0)

    twin = make_test_twin(closed_loop=True)
    twin.tick()
    with pytest.raises(ValueError, match="fresh"):
        run_closed_loop(twin, hybrid_allocate, steps=1)
