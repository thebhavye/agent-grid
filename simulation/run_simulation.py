"""Timestep-based simulation runner (real agents, no hardcoded demands)."""

from agents.appliance_agent import ApplianceAgent
from agents.battery_agent import BatteryAgent
from agents.ev_agent import EVAgent
from agents.solar_agent import SolarAgent

from .baseline_allocator import allocate

# Legacy-only fallback EV rate, used SOLELY when no appliances roster is
# available (legacy list-of-dicts path, e.g. test_baseline.py fake rows).
# Midpoint of the generator's ev_charger range (3.0-7.0 kW); clearly
# separated from the real per-dataset roster value wired below. Never
# used when a roster is present.
LEGACY_EV_MAX_CHARGE_KW_FALLBACK = 5.0


def _normalize_inputs(dataset, appliances_df):
    """Accept the real two-table dataset dict, a bare timeseries
    DataFrame, or a legacy list of row dicts.

    Returns (rows, appliances_df_or_None) where rows is a list of
    plain dicts, one per timestep.
    """
    if isinstance(dataset, dict) and "timeseries" in dataset:
        ts = dataset["timeseries"]
        ap = dataset.get("appliances", appliances_df)
        rows = ts.to_dict(orient="records") if hasattr(ts, "to_dict") else list(ts)
        return rows, ap
    if hasattr(dataset, "to_dict"):
        # Bare timeseries DataFrame passed directly.
        return dataset.to_dict(orient="records"), appliances_df
    # Legacy list-of-dicts path (used by test_baseline.py).
    return list(dataset), appliances_df


def _build_agents(appliances_df):
    """One singleton each for solar/battery/EV plus one agent per
    appliance roster entry (or fallback agents when no roster is
    supplied, e.g. legacy fake rows).

    EVAgent's max_charge_kw is sourced from the roster's ev_charger
    appliance_power_kw. Missing ev_charger row with a roster present
    raises ValueError loudly; the legacy no-roster path uses
    LEGACY_EV_MAX_CHARGE_KW_FALLBACK explicitly."""
    solar = SolarAgent("solar_1")
    battery = BatteryAgent("battery_1")
    if appliances_df is not None and hasattr(appliances_df, "to_dict"):
        roster = appliances_df.to_dict(orient="records")
        ev_power = None
        for entry in roster:
            if entry.get("appliance_id") == "ev_charger":
                ev_power = entry.get("appliance_power_kw")
                break
        if ev_power is None or ev_power != ev_power:
            raise ValueError(
                "appliances roster missing ev_charger row or its "
                "appliance_power_kw; cannot construct EVAgent."
            )
        agents = [solar, battery, EVAgent("ev_1", max_charge_kw=float(ev_power))]
        for entry in roster:
            aid = entry.get("appliance_id", "unknown")
            if aid == "ev_charger":
                continue  # already handled by EVAgent above; avoid double-counting the same physical charger
            agents.append(
                ApplianceAgent(
                    agent_id=f"appliance_{aid}",
                    appliance_id=aid,
                    appliance_priority=entry.get("appliance_priority"),
                    appliance_deferable=entry.get("appliance_deferable", False),
                    appliance_power_kw=entry.get("appliance_power_kw"),
                    appliance_usual_start_min=entry.get("appliance_usual_start_min"),
                    appliance_usual_end_min=entry.get("appliance_usual_end_min"),
                )
            )
    else:
        agents = [
            solar,
            battery,
            EVAgent("ev_1", max_charge_kw=LEGACY_EV_MAX_CHARGE_KW_FALLBACK),
            ApplianceAgent("appliance_1"),
        ]
    return agents


def run_simulation(dataset, allocator, appliances_df=None):
    """Run an allocator once for every row/timestep in the dataset.

    Agents are instantiated once (not per timestep), refreshed via
    update_state(row) each step, and their get_demand_kw() values form
    the allocator's demand dict. available_power comes from the row's
    solar_gen_kw (default 5.0 for legacy rows missing that key).

    The allocator is passed as an argument so the baseline can later be
    replaced by a smarter allocation strategy without changing this loop.
    """
    rows, resolved_appliances_df = _normalize_inputs(dataset, appliances_df)
    agents = _build_agents(resolved_appliances_df)

    results = []

    for row in rows:
        for agent in agents:
            agent.update_state(row)
        agent_demands = {agent.agent_id: float(agent.get_demand_kw()) for agent in agents}
        available_power = float(row.get("solar_gen_kw", 5.0))

        allocation = allocator(available_power, agent_demands)

        results.append(allocation)

    return results


if __name__ == "__main__":
    # Small local smoke test for manual execution.
    fake_dataset = [
        {"solar_gen_kw": 5.0},
        {"solar_gen_kw": 3.0},
    ]

    results = run_simulation(fake_dataset, allocate)

    for timestep, allocation in enumerate(results, start=1):
        print(f"Timestep {timestep}: {allocation}")

    # Real-dataset demo when pandas/generator are available.
    try:
        from data.generate_dataset import generate_dataset

        tables = generate_dataset(seed=42)
        real_results = run_simulation(tables, allocate)
        print(f"Real dataset: {len(real_results)} timesteps, e.g. step 0: {real_results[0]}")
    except Exception as exc:  # pragma: no cover - manual demo only
        print(f"Real-dataset demo skipped: {exc}")
