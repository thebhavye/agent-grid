from agents.solar_agent import SolarAgent
from agents.battery_agent import BatteryAgent
from agents.ev_agent import EVAgent
from agents.appliance_agent import ApplianceAgent

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