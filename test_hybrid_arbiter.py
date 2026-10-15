"""Behavioral checks for the standalone hybrid solar arbiter."""

from copy import deepcopy

import pytest

from data.generate_dataset import generate_dataset
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.run_simulation import _build_agents


def bid(agent_id, demand, urgency=0.5):
    return {"agent_id": agent_id, "demand_kw": demand, "urgency": urgency}


def test_surplus_and_zero_demand():
    bids = [bid("fridge", 1), bid("pump", 3), bid("solar", 0)]
    assert hybrid_allocate(10, bids, {"fridge": 1}) == {
        "fridge": 1.0, "pump": 3.0, "solar": 0.0,
    }


def test_critical_first_and_proportional_shortage():
    bids = [bid("a", 1, 0), bid("b", 3, 0), bid("flex", 8, 1)]
    priorities = {"a": 1, "b": 2, "flex": 3}
    assert hybrid_allocate(6, bids, priorities) == {"a": 1, "b": 3, "flex": 2}
    assert hybrid_allocate(2, bids, priorities) == {"a": 0.5, "b": 1.5, "flex": 0}


def test_urgency_before_priority_and_missing_priority_defaults_to_five():
    bids = [bid("urgent", 2, 0.9), bid("less", 2, 0.2)]
    assert hybrid_allocate(2, bids, {"less": 3}) == {"urgent": 2, "less": 0}


def test_ties_priority_then_proportional_and_deterministic():
    bids = [bid("z", 3), bid("a", 1), bid("first", 1), bid("last", 2)]
    priorities = {"z": 4, "a": 4, "first": 3}
    original = deepcopy((bids, priorities))
    expected = {"first": 1, "a": 0.5, "z": 1.5, "last": 0}
    assert hybrid_allocate(3, bids, priorities) == expected
    assert hybrid_allocate(3, bids, priorities) == expected
    assert hybrid_allocate(3, list(reversed(bids)), priorities) == expected
    assert (bids, priorities) == original


def test_tolerance_groups_use_highest_urgency_without_chaining():
    bids = [bid("a", 1, 0.8), bid("b", 3, 0.8 - 0.75e-9),
            bid("c", 2, 0.8 - 1.5e-9)]
    assert hybrid_allocate(2, bids, {}) == {"a": 0.5, "b": 1.5, "c": 0}


def test_near_ties_use_priority_before_exact_urgency():
    bids = [bid("high", 2, 0.8), bid("critical_flex", 2, 0.8 - 0.5e-9)]
    assert hybrid_allocate(2, bids, {"critical_flex": 3}) == {
        "high": 0, "critical_flex": 2,
    }


def test_empty_zero_power_and_all_zero_demands():
    assert hybrid_allocate(3, [], {}) == {}
    assert hybrid_allocate(0, [bid("a", 2)], {}) == {"a": 0}
    assert hybrid_allocate(3, [bid("a", 0)], {}) == {"a": 0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1])
@pytest.mark.parametrize("field", ["available", "demand"])
def test_invalid_power(value, field):
    with pytest.raises(ValueError):
        hybrid_allocate(value if field == "available" else 0,
                        [bid("a", value if field == "demand" else 1)], {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_invalid_urgency_even_with_zero_power_and_demand(value):
    with pytest.raises(ValueError):
        hybrid_allocate(0, [bid("a", 0, value)], {})


def test_duplicate_ids_and_overflow():
    with pytest.raises(ValueError, match="duplicate"):
        hybrid_allocate(0, [bid("a", 0), bid("a", 0)], {})
    with pytest.raises(ValueError, match="total_demand"):
        hybrid_allocate(1, [bid("a", 1e308), bid("b", 1e308)], {})


def test_full_generated_day_power_balance_and_caps():
    tables = generate_dataset(seed=42)
    roster = tables["appliances"].to_dict(orient="records")
    priorities = {f'appliance_{row["appliance_id"]}': row["appliance_priority"]
                  for row in roster}
    priorities["ev_1"] = next(row["appliance_priority"] for row in roster
                              if row["appliance_id"] == "ev_charger")
    agents = _build_agents(tables["appliances"])
    for row in tables["timeseries"].to_dict(orient="records"):
        for agent in agents:
            agent.update_state(row)
        bids = [agent.get_bid() for agent in agents]
        allocations = hybrid_allocate(row["solar_gen_kw"], bids, priorities)
        assert set(allocations) == {item["agent_id"] for item in bids}
        for item in bids:
            assert 0 <= allocations[item["agent_id"]] <= item["demand_kw"] + 1e-12
        assert sum(allocations.values()) == pytest.approx(
            min(row["solar_gen_kw"], sum(item["demand_kw"] for item in bids)))
