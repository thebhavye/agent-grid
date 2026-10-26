"""Outcome totals from committed closed-loop records, not open-loop replays."""

import math


METRICS = (
    "critical_unmet_kwh", "priority_1_unmet_kwh", "flexible_unmet_kwh",
    "ev_shortfall_pct_at_departure", "appliance_cycles_completed",
    "appliance_cycles_incomplete_at_window_end", "blocked_starts",
    "curtailed_solar_kwh", "battery_reserve_violation_steps",
    "battery_charge_kwh", "battery_discharge_kwh", "energy_balance_max_error_kw",
)
GRID_METRICS = ("grid_import_kwh", "grid_export_kwh", "net_cost", "self_consumption_pct")


def score_run(records) -> dict:
    """Score records produced by ``run_closed_loop`` without mutating them.

    Unmet load excludes storage and solar; EV demand is counted once. Battery
    energies are electrical terminal energies (positive power means charging).
    EV shortfall is the sum of percentage-point deficits at real departures;
    a still-connected EV at the end of a run is not a departure. Cycle deadline
    misses and later completions are distinct events, not disjoint cycle sets.
    Reserve comparisons allow 1e-9 percentage points of floating-point noise.
    """
    result = dict.fromkeys(METRICS, 0.0)
    grid_totals = dict.fromkeys(GRID_METRICS, 0.0)
    solar_kwh = 0.0
    solar_offsite_kwh = 0.0
    grid_enabled = False
    for record in records:
        dt = float(record["dt_min"])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("record dt_min must be finite and positive")
        hours = dt / 60
        for agent_id, agent in record["agents"].items():
            if agent_id in {"battery_1", "solar_1"}:
                continue
            unmet = max(0.0, agent["demand_kw"] - agent["allocated_kw"]) * hours
            priority = record["priorities"][agent_id]
            result["critical_unmet_kwh" if priority <= 2 else "flexible_unmet_kwh"] += unmet
            if priority == 1:
                result["priority_1_unmet_kwh"] += unmet
        for departure in record["ev_departures"]:
            result["ev_shortfall_pct_at_departure"] += max(
                0.0, departure["target_soc_pct"] - departure["soc_pct"]
            )
        for key in ("appliance_cycles_completed", "appliance_cycles_incomplete_at_window_end"):
            result[key] += len(record[key])
        result["blocked_starts"] += sum(record["blocked_starts"].values())
        result["curtailed_solar_kwh"] += record["curtailed_kw"] * hours
        battery = record["battery_kw"]
        result["battery_charge_kwh"] += max(0.0, battery) * hours
        result["battery_discharge_kwh"] += max(0.0, -battery) * hours
        result["battery_reserve_violation_steps"] += int(
            record["battery_soc_pct"] < record["battery_reserve_pct"] - 1e-9
        )
        actual_load = (sum(record["appliance_served_kw"].values())
                       + record["ev_applied_kw"]
                       + record["agents"]["household_base_load"]["allocated_kw"])
        imported = record.get("grid_import_kw", 0.0)
        exported = record.get("grid_export_kw", 0.0)
        if "grid_import_kw" in record:
            grid_enabled = True
            grid_totals["grid_import_kwh"] += imported * hours
            grid_totals["grid_export_kwh"] += exported * hours
            grid_totals["net_cost"] += record["import_cost"] - record["export_credit"]
        solar_kwh += record["solar_kw"] * hours
        solar_offsite_kwh += (exported + record["curtailed_kw"]) * hours
        error = abs(record["solar_kw"] + imported - battery - actual_load
                    - exported - record["curtailed_kw"])
        if not math.isfinite(error):
            raise ValueError("non-finite power balance in record")
        result["energy_balance_max_error_kw"] = max(result["energy_balance_max_error_kw"], error)
    if grid_enabled:
        grid_totals["self_consumption_pct"] = (
            100 * max(0.0, solar_kwh - solar_offsite_kwh) / solar_kwh if solar_kwh else 0.0
        )
        result.update(grid_totals)
    return result
