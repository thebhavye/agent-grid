"""Facts and templates use actual policy results and never call external APIs."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from data.generate_dataset import generate_dataset
from explain.demo import generated_day_facts
from explain.facts import build_step_facts
from explain.render import render_llm, render_template
from simulation.baseline_allocator import allocate
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.run_simulation import _build_agents


def bid(agent_id, demand=1, urgency=0):
    return {"agent_id": agent_id, "demand_kw": demand, "urgency": urgency}


def facts_for(bids, available, priorities=None, strategy="hybrid", missed=()):
    priorities = priorities or {}
    result = (hybrid_allocate(available, bids, priorities) if strategy == "hybrid"
              else allocate(available, {b["agent_id"]: b["demand_kw"] for b in bids}))
    return build_step_facts("19:30", available, bids, priorities, result,
                            missed, strategy=strategy)


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
def test_full_service_no_demand_and_surplus(strategy):
    facts = facts_for([bid("a", 2), bid("z", 0)], 5, strategy=strategy)
    assert facts["total_demand_kw"] == 2
    assert facts["curtailed_kw"] == 3
    a, z = facts["agents"]
    assert a["reason"] == "fully_served"
    assert a["allocated_kw"] == 2 and a["served_fraction"] == 1
    assert z["reason"] == "no_demand" and z["served_fraction"] == 1


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
def test_short_equal_group_is_proportional(strategy):
    facts = facts_for([bid("a", 2), bid("b", 4)], 3, strategy=strategy)
    assert [a["allocated_kw"] for a in facts["agents"]] == [1, 2]
    assert all(a["reason"] == "shared_proportionally" and a["served_fraction"] == .5
               for a in facts["agents"])


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
def test_critical_reason_matches_policy(strategy):
    facts = facts_for([bid("critical"), bid("flexible", urgency=1)], 1,
                      {"critical": 1}, strategy)
    critical, flexible = facts["agents"]
    if strategy == "hybrid":
        assert critical["allocated_kw"] == 1 and flexible["allocated_kw"] == 0
        assert critical["reason"] == flexible["reason"] == "critical_first"
    else:
        assert critical["allocated_kw"] == flexible["allocated_kw"] == .5
        assert critical["reason"] == flexible["reason"] == "shared_proportionally"


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
def test_deferred_reason_matches_urgency_policy(strategy):
    facts = facts_for([bid("high", urgency=.9), bid("low", urgency=.1)], 1,
                      strategy=strategy)
    high, low = facts["agents"]
    if strategy == "hybrid":
        assert high["reason"] == "fully_served" and high["allocated_kw"] == 1
        assert low["reason"] == "deferred_lower_urgency" and low["allocated_kw"] == 0
    else:
        assert all(a["reason"] == "shared_proportionally" for a in (high, low))


def test_equal_urgency_priority_tie_does_not_claim_lower_urgency_in_text():
    facts = facts_for([bid("a", urgency=.5), bid("b", urgency=.5)], 1,
                      {"a": 3, "b": 4})
    assert facts["agents"][1]["reason"] == "deferred_lower_urgency"
    assert "earlier urgency/priority groups" in render_template(facts)


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
@pytest.mark.parametrize("event", ["appliance_dishwasher", {"agent_id": "appliance_dishwasher"},
                                   {"appliance_id": "dishwasher", "timestamp": 1170}])
def test_missed_window_event_shapes(strategy, event):
    facts = facts_for([bid("appliance_dishwasher", 0)], 0,
                      strategy=strategy, missed=[event])
    assert facts["agents"][0]["reason"] == "missed_window"
    assert "missed window reported" in render_template(facts)


def test_missing_zero_allocation_and_default_priority():
    facts = build_step_facts("00:00", 0, [bid("a")], {}, {})
    assert facts["agents"][0] == {"agent_id": "a", "demand_kw": 1., "allocated_kw": 0.,
                                  "priority": 5, "urgency": 0., "served_fraction": 0.,
                                  "reason": "shared_proportionally"}


@pytest.mark.parametrize("allocation", [{"unknown": 0}, {"a": -1}, {"a": 2},
                                       {"a": float("nan")}, {}])
def test_bad_or_missing_positive_allocations_are_not_explained(allocation):
    with pytest.raises(ValueError):
        build_step_facts("00:00", 1, [bid("a")], {}, allocation)


def test_strategy_cannot_mislabel_baseline_as_hybrid():
    with pytest.raises(ValueError, match="policy"):
        build_step_facts("00:00", 1, [bid("a"), bid("b")], {"a": 1},
                         {"a": .5, "b": .5})


@pytest.mark.parametrize("priority", [None, "unranked", 0, 6, 2.5])
@pytest.mark.parametrize("urgency", [-1, 2])
def test_baseline_ignores_hybrid_policy_metadata(priority, urgency):
    facts = facts_for([bid("a", 2, urgency)], 1, {"a": priority}, "baseline")
    agent = facts["agents"][0]
    assert agent["allocated_kw"] == 1 and agent["served_fraction"] == .5
    assert agent["priority"] == priority and agent["urgency"] == urgency
    assert agent["reason"] == "shared_proportionally"


@pytest.mark.parametrize("urgencies,priorities", [((.5, .5), {"a": 3, "b": 4}),
                                                ((.9, .1), {"a": 3, "b": 3})])
def test_partial_singleton_after_earlier_flexible_group_is_deferred(urgencies, priorities):
    facts = facts_for([bid("a", urgency=urgencies[0]), bid("b", urgency=urgencies[1])],
                      1.5, priorities)
    assert facts["agents"][1]["allocated_kw"] == .5
    assert facts["agents"][1]["reason"] == "deferred_lower_urgency"


def test_lone_partial_group_does_not_invent_an_earlier_group():
    facts = facts_for([bid("a")], .5)
    assert facts["agents"][0]["reason"] == "shared_proportionally"


def test_flexible_group_uses_fixed_anchor_not_transitive_urgency_chain():
    facts = facts_for([bid("a", urgency=.9), bid("b", urgency=.9 - .75e-9),
                       bid("c", urgency=.9 - 1.5e-9)], 2.5)
    assert facts["agents"][2]["allocated_kw"] == .5
    assert facts["agents"][2]["reason"] == "deferred_lower_urgency"


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
def test_duplicate_bids_are_rejected_in_each_strategy(strategy):
    with pytest.raises(ValueError, match="duplicate"):
        build_step_facts("00:00", 1, [bid("a"), bid("a")], {}, {"a": 1}, strategy=strategy)


def test_partial_flexible_singleton_after_critical_group():
    facts = facts_for([bid("a"), bid("b")], 1.5, {"a": 1})
    assert facts["agents"][1]["allocated_kw"] == .5
    assert facts["agents"][1]["reason"] == "critical_first"


def test_pure_deterministic_order_and_template():
    bids = [bid("b", 2), bid("a", 4)]
    priorities, allocation = {"a": 3}, {"a": 3., "b": 0.}
    before = deepcopy((bids, priorities, allocation))
    first = build_step_facts("00:00", 3, bids, priorities, allocation)
    second = build_step_facts("00:00", 3, list(reversed(bids)), priorities, allocation)
    assert first == second and before == (bids, priorities, allocation)
    assert render_template(first) == render_template(second)
    assert all(a["agent_id"] in render_template(first) for a in first["agents"])


@pytest.mark.parametrize("response", ["a received 0.500000 kW", "a received 999 kW", RuntimeError("outage")])
def test_external_renderer_disabled_even_with_fake_client(response, caplog):
    client = Mock()
    client.return_value = response
    if isinstance(response, Exception):
        client.side_effect = response
    facts = facts_for([bid("a")], .5)
    assert render_llm(facts, client) == render_template(facts)
    client.assert_not_called()
    assert "disabled" in caplog.text


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
def test_complete_generated_day_matches_actual_allocation(strategy):
    facts_day = list(generated_day_facts(42, strategy))
    assert len(facts_day) == 288
    assert facts_day == list(generated_day_facts(42, strategy))
    data = generate_dataset(seed=42)
    agents = _build_agents(data["appliances"])
    priorities = {f"appliance_{r.appliance_id}": int(r.appliance_priority)
                  for r in data["appliances"].itertuples()}
    priorities.update(ev_1=priorities["appliance_ev_charger"], battery_1=5)
    for row, facts in zip(data["timeseries"].to_dict(orient="records"), facts_day):
        bids = []
        for agent in agents:
            agent.update_state(row)
            bids.append(agent.get_bid())
        actual = (hybrid_allocate(row["solar_gen_kw"], bids, priorities) if strategy == "hybrid"
                  else allocate(row["solar_gen_kw"], {b["agent_id"]: b["demand_kw"] for b in bids}))
        assert facts["timestamp"] == row["timestamp"]
        assert facts["curtailed_kw"] == pytest.approx(max(0, row["solar_gen_kw"] - sum(actual.values())))
        for agent in facts["agents"]:
            assert agent["allocated_kw"] == actual[agent["agent_id"]]
            assert 0 <= agent["served_fraction"] <= 1
            assert agent["allocated_kw"] <= agent["demand_kw"]
            assert agent["agent_id"] in render_template(facts)
            if strategy == "baseline" and 0 < agent["demand_kw"] and agent["served_fraction"] < 1:
                assert agent["reason"] == "shared_proportionally"
