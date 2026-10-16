"""EV preference behavior and compatibility with the original default policy."""

import pytest

from agents.ev_agent import EVAgent
from data.generate_dataset import generate_dataset
from simulation.baseline_allocator import allocate
from simulation.run_simulation import _build_agents, run_simulation


def state(**overrides):
    return dict(timestamp="06:00", ev_soc_pct=50.0, ev_departure_min=480,
                ev_plugged_in=True, **overrides)


@pytest.mark.parametrize("soc, demand", [(50, 2.1), (80, 0), (90, 0)])
def test_target_gap(soc, demand):
    ev = EVAgent("ev", 7, target_soc_pct=80)
    ev.update_state({**state(), "ev_soc_pct": soc})
    assert ev.get_demand_kw() == pytest.approx(demand)
    if soc >= 80:
        assert ev.compute_urgency() == 0


@pytest.mark.parametrize("clock, minutes_left", [("06:00", 60), ("23:00", 480),
                                                      ("07:00", 0)])
def test_ready_by_replaces_departure(clock, minutes_left):
    ev = EVAgent("ev", 7, target_soc_pct=80, ready_by_min=420)
    ev.update_state({**state(), "timestamp": clock})
    expected = 1 if minutes_left == 0 else 0.15 + 0.5 * (1 - minutes_left / 720)
    assert ev.compute_urgency() == pytest.approx(expected)


def test_row_preferences_are_step_local_and_none_restores_departure():
    ev = EVAgent("ev", 7, target_soc_pct=80, ready_by_min=420)
    ev.update_state(state(ev_target_soc_pct=90, ev_ready_by_min=390))
    assert ev.get_demand_kw() == pytest.approx(2.8)
    assert ev.compute_urgency() == pytest.approx(0.2 + 0.5 * (1 - 30 / 720))
    ev.update_state(state(ev_ready_by_min=None))
    assert ev.get_demand_kw() == pytest.approx(2.1)
    assert ev.compute_urgency() == pytest.approx(0.15 + 0.5 * (1 - 120 / 720))
    ev.update_state(state())
    assert ev.compute_urgency() == pytest.approx(0.15 + 0.5 * (1 - 60 / 720))


@pytest.mark.parametrize("field, bad", [
    ("target_soc_pct", value) for value in
    [None, 0, -1, 101, float("nan"), float("inf"), -float("inf"), "bad"]
] + [
    ("ready_by_min", value) for value in
    [-1, 1440, float("nan"), float("inf"), -float("inf"), "bad"]
])
def test_invalid_preferences_rejected_at_construction_and_update(field, bad):
    with pytest.raises(ValueError):
        EVAgent("ev", 7, **{field: bad})
    ev = EVAgent("ev", 7)
    with pytest.raises(ValueError):
        ev.update_state({**state(), "ev_plugged_in": False, "ev_" + field: bad})


@pytest.mark.parametrize("ready_by", [0, 1439.5, None])
def test_valid_ready_by_boundaries(ready_by):
    ev = EVAgent("ev", 7, target_soc_pct=0.1, ready_by_min=ready_by)
    ev.update_state(state(ev_ready_by_min=ready_by))
    assert ev.compute_urgency() == 0


def test_ready_by_without_reported_departure():
    ev = EVAgent("ev", 7, ready_by_min=420)
    ev.update_state({**state(), "ev_departure_min": None})
    assert ev.compute_urgency() == pytest.approx(0.25 + 0.5 * (1 - 60 / 720))


def test_seed_42_defaults_match_original_policy_every_step():
    dataset = generate_dataset(seed=42)
    rate = float(dataset["appliances"].set_index("appliance_id")
                 .loc["ev_charger", "appliance_power_kw"])
    ev = EVAgent("ev", rate)
    rows = dataset["timeseries"].to_dict(orient="records")
    assert len(rows) == 288
    for row in rows:
        ev.update_state(row)
        gap = max(0.0, 100.0 - row["ev_soc_pct"]) / 100.0
        demand = rate * gap if row["ev_plugged_in"] else 0.0
        hour, minute = map(int, row["timestamp"].split(":"))
        now = hour * 60 + minute
        left = row["ev_departure_min"] - now
        if left < 0:
            left += 1440
        urgency = 0.0
        if row["ev_plugged_in"] and gap > 0:
            urgency = (1.0 if left <= 0 else
                       max(0.0, min(1.0, 0.5 * gap + 0.5 * max(0.0, 1 - left / 720))))
        assert ev.get_demand_kw() == demand
        assert ev.compute_urgency() == urgency
    assert run_simulation(dataset, allocate) == run_simulation(
        dataset, allocate, ev_target_soc_pct=100, ev_ready_by_min=None)


@pytest.mark.parametrize("with_roster", [False, True])
def test_runner_and_builder_passthrough(with_roster):
    roster = generate_dataset(seed=42)["appliances"] if with_roster else None
    agents = _build_agents(roster, ev_max_charge_kw=7,
                           ev_target_soc_pct=80, ev_ready_by_min=420)
    ev = next(agent for agent in agents if isinstance(agent, EVAgent))
    ev.update_state(state())
    assert ev.get_demand_kw() == pytest.approx(2.1)
    assert ev.compute_urgency() == pytest.approx(0.15 + 0.5 * (1 - 60 / 720))
    rows = [state(solar_gen_kw=100),
            state(solar_gen_kw=100, ev_target_soc_pct=50),
            state(solar_gen_kw=100)]
    result = run_simulation(rows, allocate, appliances_df=roster,
                            ev_max_charge_kw=7, ev_target_soc_pct=80,
                            ev_ready_by_min=420)
    assert [step["ev_1"] for step in result] == pytest.approx([2.1, 0, 2.1])
