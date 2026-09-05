from agents.solar_agent import SolarAgent
from agents.battery_agent import BatteryAgent
from agents.ev_agent import EVAgent
from agents.appliance_agent import ApplianceAgent

if __name__ == "__main__":
    print("=== AgentGrid Test Run ===")
    s = SolarAgent("solar_1")
    b = BatteryAgent("battery_1")
    e = EVAgent("ev_1", max_charge_kw=6.25)  # fixed test value (seed=42 roster), not roster-sourced
    a = ApplianceAgent("appliance_1")
    s.update_state({"solar_gen_kw": 4.2})
    b.update_state({"battery_soc_pct": 62.0, "battery_health_pct": 95.0})
    e.update_state({"ev_soc_pct": 55.0, "ev_departure_min": 480, "ev_plugged_in": True})
    a.update_state({"timestamp": "20:00"})

    print(f"Solar ID: {s.agent_id} | Gen KW: {s.solar_gen_kw}")
    print(f"Battery ID: {b.agent_id} | SOC %: {b.battery_soc_pct}")
    print(f"EV ID: {e.agent_id} | Plugged In: {e.ev_plugged_in}")
    print(f"Appliance ID: {a.agent_id} | Priority: {a.appliance_priority}")