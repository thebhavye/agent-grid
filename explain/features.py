"""Decision-time features shared by surrogate fitting and urgency experiments."""

import math

FEATURES = (
    "hour", "solar_kw", "available_kw", "battery_soc_pct", "battery_mode",
    "priority", "urgency", "requested_kw", "total_demand_kw", "demand_share",
    "ev_plugged_in", "ev_soc_pct",
)
CANDIDATES = ("soc_gap", "time_pressure", "window_width", "priority_pressure",
              "battery_health", "hour_sin", "solar_forecast_kw", "recent_unmet_kwh")


def agent_type(agent_id):
    if agent_id == "ev_1":
        return "ev"
    if agent_id == "battery_1":
        return "battery"
    return "appliance"


def decision_features(context, data, recent_unmet):
    """Build pre-commit state; forecast is persistence, not future truth.

    Time pressure is normalized to one day. No realized allocation, future
    weather or post-commit SOC is used as an input.
    """
    obs = data["observation"]
    twin = context["twin"]
    minute = data["start"].hour * 60 + data["start"].minute
    total = sum(b["demand_kw"] for b in data["allocator_bids"])
    specs = {f"appliance_{s.appliance_id}": s for s in twin._appliances.config.appliances}
    rows = {}
    for bid in data["allocator_bids"]:
        name = bid["agent_id"]
        spec = specs.get(name)
        end = obs.get("ev_departure_min") if name == "ev_1" else getattr(spec, "usual_end_min", None)
        start = getattr(spec, "usual_start_min", None)
        left = (end - minute) % 1440 if end is not None else 1440
        width = (end - start) % 1440 if end is not None and start is not None else 1440
        soc = obs["ev_soc_pct"] if name == "ev_1" else obs["battery_soc_pct"]
        target = obs.get("ev_target_soc_pct", 100) if name == "ev_1" else 100
        priority = context["priorities"].get(name, 5)
        rows[name] = {
            "hour": minute / 60, "solar_kw": obs["solar_gen_kw"],
            "available_kw": data["available_kw"], "battery_soc_pct": obs["battery_soc_pct"],
            "battery_mode": {"idle": 0, "charge": 1, "discharge": -1}[data["mode"]["mode"]],
            "priority": priority, "urgency": bid["urgency"], "requested_kw": bid["demand_kw"],
            "total_demand_kw": total, "demand_share": bid["demand_kw"] / total if total else 0,
            "ev_plugged_in": float(obs["ev_plugged_in"]), "ev_soc_pct": obs["ev_soc_pct"],
            "soc_gap": max(0, target - soc) / 100 if name in ("ev_1", "battery_1") else 0,
            "time_pressure": 1 - left / 1440 if end is not None else 0,
            "window_width": width / 1440, "priority_pressure": (6 - priority) / 5,
            "battery_health": twin._battery.health_pct / 100,
            "hour_sin": (math.sin(2 * math.pi * minute / 1440) + 1) / 2,
            "solar_forecast_kw": obs["solar_gen_kw"],
            "recent_unmet_kwh": recent_unmet.get(name, 0),
        }
    return rows


def record_features(record, agent_id):
    """Prefer captured inputs; legacy records are reconstructed and approximate."""
    if "decision_features" in record:
        return record["decision_features"][agent_id]
    agent = record["agents"][agent_id]
    total = sum(a["demand_kw"] for a in record["agents"].values())
    hh, mm = map(int, record["timestamp"].split(":"))
    return dict(zip(FEATURES, (
        hh + mm / 60, record["solar_kw"], record["available_kw"], record["battery_soc_pct"],
        {"idle": 0, "charge": 1, "discharge": -1}[record["mode"]],
        record["priorities"].get(agent_id, 5), agent["urgency"], agent["demand_kw"], total,
        agent["demand_kw"] / total if total else 0,
        float(record["ev_plugged_in"]), record["ev_soc_pct"],
    )))
