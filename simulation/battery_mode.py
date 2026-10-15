"""Shared, pure battery mode selection for baseline and hybrid callers."""

from .baseline_allocator import _require_valid_kw


def choose_battery_mode(solar_kw, other_demand_kw, battery_demand_kw,
                        supply_kw, headroom_kw) -> dict:
    """Choose charge on solar surplus, discharge on deficit, or idle.

    Charge demand is capped by headroom; supply is capped by the deficit.
    These are feasible bids, not applied power or updates to battery state.
    Every input must be finite and nonnegative, even in an unused mode.
    """
    solar_kw = _require_valid_kw(solar_kw, "solar_kw")
    other_demand_kw = _require_valid_kw(other_demand_kw, "other_demand_kw")
    battery_demand_kw = _require_valid_kw(battery_demand_kw, "battery_demand_kw")
    supply_kw = _require_valid_kw(supply_kw, "supply_kw")
    headroom_kw = _require_valid_kw(headroom_kw, "headroom_kw")
    demand = 0.0
    supply = 0.0
    mode = "idle"
    if solar_kw >= other_demand_kw:
        demand = min(battery_demand_kw, headroom_kw)
        if demand > 0:
            mode = "charge"
    else:
        supply = min(supply_kw, other_demand_kw - solar_kw)
        if supply > 0:
            mode = "discharge"
    return {"mode": mode, "battery_demand_kw": float(demand),
            "battery_supply_kw": float(supply)}
