"""Behavioral checks for the standalone hybrid solar arbiter."""

from copy import deepcopy
import random

import numpy as np
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
    assert hybrid_allocate(2, bids, priorities) == {"a": 1.0, "b": 1.0, "flex": 0.0}


def test_priority_one_is_fully_served_before_priority_two():
    bids = [bid("p1_a", 2), bid("p1_b", 1), bid("p2", 4), bid("flex", 5, 1)]
    priorities = {"p1_a": 1, "p1_b": 1, "p2": 2, "flex": 3}

    assert hybrid_allocate(2, bids, priorities) == {
        "flex": 0.0, "p1_a": 4 / 3, "p1_b": 2 / 3, "p2": 0.0,
    }
    assert hybrid_allocate(5, bids, priorities) == {
        "flex": 0.0, "p1_a": 2.0, "p1_b": 1.0, "p2": 2.0,
    }


def test_priority_two_never_receives_power_before_priority_one_is_full():
    bids = [bid("p1", 2), bid("p2_a", 1), bid("p2_b", 3), bid("flex", 4)]
    priorities = {"p1": 1, "p2_a": 2, "p2_b": 2, "flex": 3}
    for available in (0, 1, 1.99, 2, 2.5, 5, 10):
        result = hybrid_allocate(available, bids, priorities)
        if result["p2_a"] or result["p2_b"]:
            assert result["p1"] == pytest.approx(2.0)


def test_randomized_priority_order_invariants():
    rng = random.Random(55)
    for _ in range(200):
        bids = [
            bid(f"agent_{index}", rng.uniform(0, 5), rng.random())
            for index in range(8)
        ]
        priorities = {
            item["agent_id"]: rng.randint(1, 5)
            for item in bids
        }
        available = rng.uniform(0, sum(item["demand_kw"] for item in bids) + 2)
        allocations = hybrid_allocate(available, bids, priorities)
        reordered = hybrid_allocate(available, list(reversed(bids)), priorities)

        assert allocations == reordered
        assert sum(allocations.values()) == pytest.approx(
            min(available, sum(item["demand_kw"] for item in bids))
        )
        for item in bids:
            agent_id = item["agent_id"]
            assert 0.0 <= allocations[agent_id] <= item["demand_kw"]

        p1 = [item for item in bids if priorities[item["agent_id"]] == 1]
        p2 = [item for item in bids if priorities[item["agent_id"]] == 2]
        p1_allocated = sum(allocations[item["agent_id"]] for item in p1)
        p1_demand = sum(item["demand_kw"] for item in p1)
        if any(allocations[item["agent_id"]] > 1e-12 for item in p2):
            assert p1_allocated == pytest.approx(p1_demand)


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


@pytest.mark.parametrize("value", [None, "1", 0, 6, 2.5, float("nan"), True, False])
def test_invalid_priority_names_agent(value):
    with pytest.raises(ValueError, match="priority.*'a'"):
        hybrid_allocate(0, [bid("a", 0)], {"a": value})


@pytest.mark.parametrize("value", [1, 2, 3, 4, 5, np.int64(1), np.int32(5)])
def test_valid_integral_priorities(value):
    assert hybrid_allocate(2, [bid("a", 2)], {"a": value}) == {"a": 2.0}


def test_missing_priority_shares_with_explicit_priority_five():
    assert hybrid_allocate(2, [bid("a", 1), bid("b", 3)], {"b": 5}) == {
        "a": 0.5, "b": 1.5,
    }


@pytest.mark.parametrize("available", [0, 3])
def test_zero_demand_allocations_are_floats(available):
    result = hybrid_allocate(available, [bid("a", 0), bid("c", 0)], {"a": 1})
    assert result == {"a": 0.0, "c": 0.0}
    assert all(isinstance(value, float) for value in result.values())


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

        p1_ids = {
            item["agent_id"] for item in bids
            if priorities.get(item["agent_id"], 5) == 1
        }
        p2_ids = {
            item["agent_id"] for item in bids
            if priorities.get(item["agent_id"], 5) == 2
        }
        if any(allocations[agent_id] > 0 for agent_id in p2_ids):
            assert sum(allocations[agent_id] for agent_id in p1_ids) == pytest.approx(
                sum(item["demand_kw"] for item in bids if item["agent_id"] in p1_ids)
            )
