"""Regression tests for Issue #10: missed_windows current-timestep semantics.

Covers duplicate-call protection, timestep-boundary clearing, in-window /
non-deferable behavior, midnight-wrapping windows, and bounded growth.
"""

from agents.appliance_agent import ApplianceAgent


def _make_closed_window_agent():
    return ApplianceAgent(
        "app_missed",
        appliance_id="dishwasher",
        appliance_deferable=True,
        appliance_usual_start_min=1200,
        appliance_usual_end_min=1320,
    )


def test_duplicate_call_protection():
    agent = _make_closed_window_agent()
    agent.update_state({"timestamp": "12:00"})

    assert agent.compute_urgency() == 0.0
    assert agent.compute_urgency() == 0.0

    assert len(agent.missed_windows) == 1
    assert agent.missed_windows[0]["appliance_id"] == "dishwasher"
    assert agent.missed_windows[0]["timestamp"] == 12 * 60


def test_same_event_cannot_duplicate_over_many_calls():
    agent = _make_closed_window_agent()
    agent.update_state({"timestamp": "12:00"})

    for _ in range(5):
        assert agent.compute_urgency() == 0.0

    assert len(agent.missed_windows) == 1


def test_new_timestep_clears_previous_event():
    agent = _make_closed_window_agent()

    # Timestep 1: outside window -> one event.
    agent.update_state({"timestamp": "12:00"})
    assert agent.compute_urgency() == 0.0
    assert len(agent.missed_windows) == 1

    # Timestep 2 advances -> previous event cleared, before any new urgency call.
    agent.update_state({"timestamp": "12:05"})
    assert agent.missed_windows == []

    # Timestep 2 is also outside the window, so it generates its own event.
    assert agent.compute_urgency() == 0.0
    assert len(agent.missed_windows) == 1
    assert agent.missed_windows[0]["timestamp"] == 12 * 60 + 5


def test_in_window_behavior_generates_no_event():
    agent = ApplianceAgent(
        "app_1",
        appliance_id="dishwasher",
        appliance_deferable=True,
        appliance_usual_start_min=1200,
        appliance_usual_end_min=1320,
    )
    agent.update_state({"timestamp": "21:00"})  # 1260, inside 1200-1320

    urgency = agent.compute_urgency()
    # Existing formula: 1 - (60 / 240) = 0.75.
    assert urgency == 0.75
    assert agent.missed_windows == []


def test_non_deferable_behavior():
    agent = ApplianceAgent(
        "fridge",
        appliance_id="fridge",
        appliance_deferable=False,
        appliance_power_kw=0.15,
    )
    agent.update_state({"timestamp": "12:00"})

    assert agent.get_demand_kw() == 0.15
    assert agent.compute_urgency() == 0.0
    assert agent.compute_urgency() == 0.0
    assert agent.missed_windows == []


def test_midnight_wrapping_window():
    # Window 20:00 -> 00:00 (start > end): 20:10 inside, 12:00 outside.
    inside = ApplianceAgent(
        "app_in",
        appliance_id="ev_charger",
        appliance_deferable=True,
        appliance_usual_start_min=1200,
        appliance_usual_end_min=0,
    )
    inside.update_state({"timestamp": "20:10"})
    assert inside.compute_urgency() >= 0.0
    assert inside.missed_windows == []

    outside = ApplianceAgent(
        "app_out",
        appliance_id="ev_charger",
        appliance_deferable=True,
        appliance_usual_start_min=1200,
        appliance_usual_end_min=0,
    )
    outside.update_state({"timestamp": "12:00"})
    assert outside.compute_urgency() == 0.0
    assert outside.compute_urgency() == 0.0
    assert len(outside.missed_windows) == 1

    # Boundary still clears on advance.
    outside.update_state({"timestamp": "12:05"})
    assert outside.missed_windows == []


def test_no_unbounded_growth_over_288_steps():
    agent = _make_closed_window_agent()
    # Simulate a full day at 5-min resolution; most steps are outside the
    # 1200-1320 window, exercising the append+clear path every step.
    for step in range(288):
        minutes = step * 5
        agent.update_state({"timestamp": f"{minutes // 60:02d}:{minutes % 60:02d}"})
        # Phase 2 bid packaging will call compute_urgency() every step,
        # possibly more than once — exercise both.
        agent.compute_urgency()
        agent.compute_urgency()
        assert len(agent.missed_windows) <= 1

    assert len(agent.missed_windows) <= 1


def test_consumer_accessor_returns_copy():
    agent = _make_closed_window_agent()
    agent.update_state({"timestamp": "12:00"})
    agent.compute_urgency()

    snapshot = agent.get_missed_windows()
    assert snapshot == agent.missed_windows
    snapshot.clear()
    # Mutating the snapshot must not mutate the internal buffer.
    assert len(agent.missed_windows) == 1

    # Mutating a returned event dict must not mutate internal state.
    snapshot2 = agent.get_missed_windows()
    snapshot2[0]["timestamp"] = -1
    snapshot2[0]["appliance_id"] = "tampered"
    assert agent.missed_windows[0]["timestamp"] == 12 * 60
    assert agent.missed_windows[0]["appliance_id"] == "dishwasher"
