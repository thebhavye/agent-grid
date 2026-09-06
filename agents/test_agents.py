from agents.solar_agent import SolarAgent
from agents.battery_agent import BatteryAgent
from agents.ev_agent import EVAgent
from agents.appliance_agent import ApplianceAgent

def test_solar_urgency_is_zero():
    agent = SolarAgent("solar_1")
    agent.update_state({"solar_gen_kw": 5.0})
    assert agent.compute_urgency() == 0.0


def test_battery_urgency_reflects_soc():
    high_soc = BatteryAgent("bat_high")
    high_soc.update_state({"battery_soc_pct": 95.0})

    low_soc = BatteryAgent("bat_low")
    low_soc.update_state({"battery_soc_pct": "10.0"})

    assert high_soc.compute_urgency() < low_soc.compute_urgency()
    assert low_soc.compute_urgency() == 0.81


def test_ev_urgency_handles_missing_departure_and_time_pressure():
    ev_missing_departure = EVAgent("ev_missing", max_charge_kw=6.25)
    ev_missing_departure.update_state(
        {"ev_soc_pct": 20.0, "ev_plugged_in": True, "timestamp": "10:00"}
    )

    ev_panic = EVAgent("ev_panic", max_charge_kw=6.25)
    ev_panic.update_state(
        {
            "ev_soc_pct": 20.0,
            "ev_plugged_in": True,
            "ev_departure_min": 630,
            "timestamp": "10:00",
        }
    )

    assert ev_missing_departure.compute_urgency() == 0.0
    assert ev_panic.compute_urgency() > 0.8


def test_appliance_urgency_rises_near_window_end():
    app_open = ApplianceAgent(
        "app_open",
        appliance_priority=4,
        appliance_deferable=True,
        appliance_power_kw=1.25,
        appliance_usual_start_min=1200,
        appliance_usual_end_min=1440,
    )
    app_open.update_state({"timestamp": "20:10"})

    app_close = ApplianceAgent(
        "app_close",
        appliance_priority=4,
        appliance_deferable=True,
        appliance_power_kw=1.25,
        appliance_usual_start_min=1200,
        appliance_usual_end_min=1440,
    )
    app_close.update_state({"timestamp": "23:50"})

    assert app_open.compute_urgency() < app_close.compute_urgency()
