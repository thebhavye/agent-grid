from agents.solar_agent import SolarAgent
from agents.battery_agent import BatteryAgent
from agents.ev_agent import EVAgent
from agents.appliance_agent import ApplianceAgent

if __name__ == "__main__":
    print("=== AGENTGRID URGENCY TESTS ===")
    
    # 1. SOLAR (Always 0.0)
    s = SolarAgent("solar_1")
    s.update_state({"solar_gen_kw": 5.0})
    print(f"Solar Urgency: {s.compute_urgency():.2f}")
    print("-" * 30)

    # 2. BATTERY
    b_high = BatteryAgent("bat_high")
    b_high.update_state({"battery_soc_pct": 95.0}) # 95% - low urgency
    print(f"Battery (95% SoC) Urgency: {b_high.compute_urgency():.2f}")
    
    b_low = BatteryAgent("bat_low")
    b_low.update_state({"battery_soc_pct": 10.0}) # 10% - high urgency
    print(f"Battery (10% SoC) Urgency: {b_low.compute_urgency():.2f}")
    print("-" * 30)

    # 3. EV
    # Assuming current time is 10:00 (600 mins)
    ev_chill = EVAgent("ev_chill", max_charge_kw=6.25)
    ev_chill.update_state({"ev_soc_pct": 80.0, "ev_plugged_in": True, "ev_departure_min": 1200, "timestamp": "10:00"}) 
    print(f"EV (80% SoC, 10hrs left) Urgency: {ev_chill.compute_urgency():.2f}")

    ev_panic = EVAgent("ev_panic", max_charge_kw=6.25)
    ev_panic.update_state({"ev_soc_pct": 20.0, "ev_plugged_in": True, "ev_departure_min": 630, "timestamp": "10:00"}) 
    print(f"EV (20% SoC, 30m left) Urgency: {ev_panic.compute_urgency():.2f}")
    print("-" * 30)

    # 4. APPLIANCE
    # Window is 1200 to 1440 (20:00 to 24:00)
    app_open = ApplianceAgent("app_open", appliance_priority=4, appliance_deferable=True, appliance_power_kw=1.25, appliance_usual_start_min=1200, appliance_usual_end_min=1440)
    app_open.update_state({"timestamp": "20:10"}) # 20:10 = 1210 mins (just started)
    print(f"Appliance (Just started) Urgency: {app_open.compute_urgency():.2f}")

    app_close = ApplianceAgent("app_close", appliance_priority=4, appliance_deferable=True, appliance_power_kw=1.25, appliance_usual_start_min=1200, appliance_usual_end_min=1440)
    app_close.update_state({"timestamp": "23:50"}) # 23:50 = 1430 mins (closing soon)
    print(f"Appliance (Closing soon) Urgency: {app_close.compute_urgency():.2f}")
    print("-" * 30)