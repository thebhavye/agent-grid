from agents.solar_agent import SolarAgent
from agents.battery_agent import BatteryAgent
from agents.ev_agent import EVAgent
from agents.appliance_agent import ApplianceAgent


def _check_bid(agent):
    """Assert a bid packages the agent's live demand + urgency. Shared by
    the pytest test below and the manual __main__ smoke path."""
    bid = agent.get_bid()
    assert set(bid.keys()) == {"agent_id", "demand_kw", "urgency"}, bid
    assert bid["agent_id"] == agent.agent_id
    assert bid["demand_kw"] == agent.get_demand_kw()
    assert bid["urgency"] == agent.compute_urgency()
    return bid


def _make_bid_agents():
    """One of each real agent type in realistic state (non-zero demand
    except solar, which never consumes)."""
    solar = SolarAgent("solar_bid")
    solar.update_state({"solar_gen_kw": 5.0})

    battery = BatteryAgent("battery_bid")
    battery.update_state({"battery_soc_pct": 10.0})

    ev = EVAgent("ev_bid", max_charge_kw=6.25)
    ev.update_state({"ev_soc_pct": 20.0, "ev_departure_min": 30, "timestamp": "00:00"})

    appliance = ApplianceAgent("app_bid", appliance_id="dishwasher", appliance_deferable=True,
                               appliance_power_kw=1.2,
                               appliance_usual_start_min=1200, appliance_usual_end_min=0)
    appliance.update_state({"timestamp": "23:50"})

    return solar, battery, ev, appliance


def test_get_bid_packaging():
    solar, battery, ev, appliance = _make_bid_agents()

    _check_bid(solar)
    assert _check_bid(battery)["demand_kw"] > 0.0
    assert _check_bid(ev)["demand_kw"] > 0.0
    assert _check_bid(appliance)["demand_kw"] > 0.0


def test_nonfinite_soc_raises_loudly():
    """Issue #26: NaN/Inf state must raise, not yield contradictory pairs."""
    import math

    import pytest

    battery = BatteryAgent("battery_nan")
    battery.update_state({"battery_soc_pct": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        battery.get_demand_kw()
    with pytest.raises(ValueError, match="finite"):
        battery.compute_urgency()

    ev = EVAgent("ev_inf", max_charge_kw=7.0)
    ev.update_state({"ev_soc_pct": 20.0, "ev_departure_min": math.inf,
                     "timestamp": "00:00"})
    with pytest.raises(ValueError, match="finite"):
        ev.compute_urgency()
    ev.update_state({"ev_soc_pct": float("nan"), "ev_departure_min": 30,
                     "timestamp": "00:00"})
    with pytest.raises(ValueError, match="finite"):
        ev.get_demand_kw()


def test_unplugged_ev_invalid_soc_raises():
    """Issue #26: unplugged EV must still fail fast on NaN/Inf SOC.

    get_demand_kw()/compute_urgency() validate SOC before the
    not-plugged-in early return, so a broken sensor value cannot hide
    behind the unplugged state.
    """
    import pytest

    for bad_soc in (float("nan"), float("inf")):
        ev = EVAgent("ev_unplugged", max_charge_kw=7.0)
        ev.update_state({"ev_soc_pct": bad_soc, "ev_plugged_in": False,
                         "ev_departure_min": 30, "timestamp": "00:00"})
        with pytest.raises(ValueError, match="finite"):
            ev.get_demand_kw()
        with pytest.raises(ValueError, match="finite"):
            ev.compute_urgency()


def test_nonfinite_timestamp_raises():
    """Issue #26: non-finite numeric timestamps must raise loudly."""
    import pytest

    for bad_ts in (float("nan"), float("inf")):
        ev = EVAgent("ev_bad_ts", max_charge_kw=7.0)
        with pytest.raises(ValueError, match="timestamp"):
            ev.update_state({"ev_soc_pct": 20.0, "ev_departure_min": 30,
                             "timestamp": bad_ts})

    ev = EVAgent("ev_bad_min", max_charge_kw=7.0)
    ev.update_state({"ev_soc_pct": 20.0, "ev_departure_min": 30,
                     "timestamp": "00:00"})
    ev._current_min = float("inf")
    with pytest.raises(ValueError, match="timestamp"):
        ev.compute_urgency()


def test_infinite_appliance_power_raises():
    """Issue #26: Inf rated power must raise; None/NaN keeps the 1 kW default."""
    import pytest

    app = ApplianceAgent("app_inf", appliance_power_kw=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        app.get_demand_kw()

    assert ApplianceAgent("app_none").get_demand_kw() == 1.0
    assert ApplianceAgent("app_nan", appliance_power_kw=float("nan")).get_demand_kw() == 1.0


if __name__ == "__main__":
    print("=== AGENTGRID URGENCY TESTS ===")

    # 1. SOLAR
    s = SolarAgent("solar_1")
    s.update_state({"solar_gen_kw": 5.0})
    print(f"Solar Urgency: {s.compute_urgency():.2f}")
    print("-" * 30)

    # 2. BATTERY
    b_high = BatteryAgent("battery_high")
    b_high.update_state({"battery_soc_pct": 95.0})
    b_low = BatteryAgent("battery_low")
    b_low.update_state({"battery_soc_pct": 10.0})
    print(f"Battery (95% SoC) Urgency: {b_high.compute_urgency():.2f}")
    print(f"Battery (10% SoC) Urgency: {b_low.compute_urgency():.2f}")
    print("-" * 30)

    # 3. EV
    ev_normal = EVAgent("ev_1", max_charge_kw=6.25)
    ev_normal.update_state({"ev_soc_pct": 80.0, "ev_departure_min": 600, "timestamp": "00:00"})
    ev_urgent = EVAgent("ev_2", max_charge_kw=6.25)
    ev_urgent.update_state({"ev_soc_pct": 20.0, "ev_departure_min": 30, "timestamp": "00:00"})
    ev_overnight = EVAgent("ev_3", max_charge_kw=6.25)
    ev_overnight.update_state({"ev_soc_pct": 50.0, "ev_departure_min": 360, "timestamp": "22:00"})

    print(f"EV (80% SoC, 10hrs left) Urgency: {ev_normal.compute_urgency():.2f}")
    print(f"EV (20% SoC, 30m left) Urgency: {ev_urgent.compute_urgency():.2f}")
    print(f"EV (50% SoC, 22:00->06:00 Overnight) Urgency: {ev_overnight.compute_urgency():.2f}")
    print("-" * 30)

    # 4. APPLIANCE
    app_open = ApplianceAgent("app_1", appliance_deferable=True, appliance_usual_start_min=1200, appliance_usual_end_min=0)
    app_open.update_state({"timestamp": "20:10"})
    app_close = ApplianceAgent("app_2", appliance_deferable=True, appliance_usual_start_min=1200, appliance_usual_end_min=0)
    app_close.update_state({"timestamp": "23:50"})
    print(f"Appliance (Just started) Urgency: {app_open.compute_urgency():.2f}")
    print(f"Appliance (Closing soon) Urgency: {app_close.compute_urgency():.2f}")
    print("-" * 30)

    # 5. MISSED-WINDOW TRACKING (deferable, window closed -> logged in memory)
    app_missed = ApplianceAgent("app_missed", appliance_id="dishwasher", appliance_deferable=True,
                                appliance_usual_start_min=1200, appliance_usual_end_min=1320)
    app_missed.update_state({"timestamp": "12:00"})
    assert app_missed.compute_urgency() == 0.0
    assert len(app_missed.missed_windows) == 1
    assert app_missed.missed_windows[0]["appliance_id"] == "dishwasher"
    print(f"Missed-window tracking: {app_missed.missed_windows}")
    print("-" * 30)

    # 6. BID PACKAGING (Phase 2 prep: demand + urgency in one bid dict)
    bid_solar, bid_battery, bid_ev, bid_app = _make_bid_agents()
    print(f"Solar bid: {_check_bid(bid_solar)}")
    print(f"Battery bid: {_check_bid(bid_battery)}")
    print(f"EV bid: {_check_bid(bid_ev)}")
    print(f"Appliance bid: {_check_bid(bid_app)}")
    print("-" * 30)