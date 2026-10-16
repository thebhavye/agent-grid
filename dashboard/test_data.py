"""Tests for the dashboard data layer (no Streamlit needed)."""

import pytest

from dashboard import data
from scenarios import CATALOG, list_scenarios


@pytest.fixture(scope="module")
def clear_hybrid():
    return data.run(data.RunKey("clear_day", "hybrid"))


def test_run_covers_every_scenario_step_count():
    for name in list_scenarios():
        records = data.run(data.RunKey(name, "baseline"))
        assert len(records) == CATALOG[name]["steps"]


@pytest.mark.parametrize("field,value", [("scenario", "nope"), ("strategy", "nope"),
                                         ("fault", "nope")])
def test_run_rejects_unknown_inputs(field, value):
    kwargs = {"scenario": "clear_day", "strategy": "hybrid", "fault": "none"}
    kwargs[field] = value
    with pytest.raises(ValueError, match="unknown"):
        data.run(data.RunKey(**kwargs))


def test_timeseries_balances_power(clear_hybrid):
    frame = data.timeseries(clear_hybrid)
    assert len(frame) == 288
    supply = frame.solar_kw + frame.battery_discharge_kw
    use = frame.delivered_kw + frame.battery_charge_kw + frame.curtailed_kw
    assert (supply - use).abs().max() < 1e-6
    assert (frame.critical_unmet_kw >= 0).all() and (frame.flexible_unmet_kw >= 0).all()
    assert frame.clock.iloc[0] == "Day 1 00:00"


def test_allocation_groups_sum_to_delivered_plus_charging(clear_hybrid):
    groups = data.allocation_groups(clear_hybrid).groupby("step").kw.sum()
    frame = data.timeseries(clear_hybrid)
    expected = frame.delivered_kw + frame.battery_charge_kw
    assert (groups.values - expected.values).__abs__().max() < 1e-9


def test_three_day_clock_labels_days():
    frame = data.timeseries(data.run(data.RunKey("three_days", "hybrid")))
    assert frame.clock.iloc[288] == "Day 2 00:00"
    assert frame.clock.iloc[-1] == "Day 3 23:55"


@pytest.mark.parametrize("strategy", ["hybrid", "baseline"])
def test_every_step_explains_with_policy_reasons(strategy):
    records = data.run(data.RunKey("clear_day", strategy))
    for record in records:
        lines = data.explain_step(record, strategy)
        assert lines[0].startswith(record["timestamp"])
        # Fault-free steps reproduce the policy exactly, so no fallback note.
        assert not any("unavailable" in line for line in lines)


def test_explanation_wording_matches_what_was_received(clear_hybrid):
    import re
    for record in clear_hybrid:
        for line in data.explain_step(record, "hybrid")[1:]:
            match = re.match(r".+?: [\d.]+ of [\d.]+ kW( \((\d+)%\))?, (.+)\.$", line)
            if not match:
                continue
            partial, why = match[1] is not None, match[3]
            if why.startswith("served in full"):
                assert not partial, line
            if why.startswith(("held back", "shared what was left", "waited", "same urgency")):
                assert partial and int(match[2]) < 100, line


def _step(strategy, timestamp, scenario="clear_day"):
    records = data.run(data.RunKey(scenario, strategy))
    return next(r for r in records if r["timestamp"] == timestamp)


def _line_for(lines, label):
    return next(line for line in lines if line.startswith(f"{label}:"))


def test_baseline_shortage_says_proportional_not_equally_ranked():
    # clear_day baseline 03:15: priority-1 and priority-2 loads get the same
    # fraction because the baseline ignores rank (PR #80 review).
    lines = data.explain_step(_step("baseline", "03:15"), "baseline")
    for label in ("Fridge", "Lighting"):
        line = _line_for(lines, label)
        assert "proportional baseline" in line, line
        assert "ranked" not in line and "essential" not in line, line


def test_priority_tiebreak_is_not_called_more_urgent():
    # clear_day hybrid 09:35: washing machine (p4) and water heater (p3) both
    # have urgency 0.00; the water heater goes first on priority, not urgency.
    lines = data.explain_step(_step("hybrid", "09:35"), "hybrid")
    line = _line_for(lines, "Washing machine")
    assert "more urgent" not in line, line
    assert "same urgency" in line and "Water heater" in line and "3 vs 4" in line, line


@pytest.mark.parametrize("scenario", ["clear_day", "monsoon_cloudy", "overnight_ev_cloudy"])
def test_hybrid_reasons_are_true_for_every_step(scenario):
    """Each causal claim in the hybrid wording holds for that step's bids."""
    import re
    for record in data.run(data.RunKey(scenario, "hybrid")):
        agents = {data.agent_label(k): (record["priorities"].get(k, 5), v["urgency"], v["allocated_kw"])
                  for k, v in record["agents"].items()}
        for line in data.explain_step(record, "hybrid")[1:]:
            name = line.split(":", 1)[0]
            if name not in agents:
                continue
            priority, urgency, _ = agents[name]
            match = re.search(r"waited behind more urgent loads \((.+)\)", line)
            if match:
                for other in match[1].split(", "):
                    assert agents[other][1] > urgency + 1e-9 and agents[other][2] > 0, line
            match = re.search(r"same urgency \(.+\) as (.+), which went first on priority", line)
            if match:
                other = agents[match[1]]
                assert abs(other[1] - urgency) <= 1e-9 and other[0] < priority and other[2] > 0, line
            if "with other priority-" in line:
                assert priority <= 2, line


def test_baseline_wording_never_claims_priority_or_urgency():
    for record in data.run(data.RunKey("clear_day", "baseline")):
        for line in data.explain_step(record, "baseline")[1:]:
            assert "essential" not in line and "urgen" not in line and "ranked" not in line, line


def test_step_table_orders_by_priority(clear_hybrid):
    table = data.step_table(clear_hybrid[150])
    assert list(table.Priority) == sorted(table.Priority)
    assert "Battery" in set(table.Agent) and "Solar" not in set(table.Agent)


@pytest.mark.parametrize("fault", [key for key in data.FAULT_PRESETS if key != "none"])
def test_fault_presets_run_and_explain(fault):
    records = data.run(data.RunKey("clear_day", "hybrid", fault))
    assert len(records) == 288
    frame = data.timeseries(records)
    assert frame.faults.str.len().sum() > 0
    for record in records:
        data.explain_step(record, "hybrid")


def test_inverter_trip_zeroes_solar_at_noon():
    frame = data.timeseries(data.run(data.RunKey("clear_day", "hybrid", "inverter_trip")))
    assert (frame.loc[144:155, "solar_kw"] == 0).all()
    assert frame.loc[143, "solar_kw"] > 0


def test_comparison_has_every_metric_and_consistent_delta():
    table = data.comparison("clear_day")
    assert len(table) == len(data.METRIC_LABELS)
    assert ((table.Hybrid - table.Baseline) - table["Hybrid - baseline"]).abs().max() < 1e-12


def test_agent_labels():
    assert data.agent_label("appliance_pool_pump") == "Pool pump"
    assert data.agent_label("ev_1") == "EV"
    assert data.agent_label("household_base_load") == "Base load"


def test_demo_scenario_notes_describe_the_explicit_array_sizing():
    assert "demo_sunny_home" in list_scenarios()
    assert "60 m2" in data.SCENARIO_NOTES["demo_sunny_home"]
    assert "12 kWp" in data.SCENARIO_NOTES["demo_sunny_home"]
