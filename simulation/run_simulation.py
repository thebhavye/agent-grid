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

# Generator-compatible appliance table columns. Twin roster records are
# converted to exactly these columns so _build_agents() consumes twin and
# generator rosters through the same path.
TWIN_APPLIANCE_COLUMNS = [
    "appliance_id",
    "appliance_priority",
    "appliance_deferable",
    "appliance_power_kw",
    "appliance_usual_start_min",
    "appliance_usual_end_min",
]


def get_twin_appliances_df(appliance_config=None):
    """Build a generator-shaped appliances DataFrame from the twin roster.

    Source is the twin-provided ApplianceModelConfig (defaults when None),
    already carrying the same 8 appliance IDs and usual windows as
    data/generate_dataset.py. Note: the ev_charger row's
    appliance_power_kw (5.0 kW default) is only the charger's display
    rating — it is EXCLUDED from household_load_kw and does NOT clamp
    twin charging. The authoritative EV rate is EVModelConfig.max_charge_kw
    (see get_twin_ev_max_charge_kw); pass it as ev_max_charge_kw so the
    EVAgent matches twin physics.
    """
    from digital_twin.appliance_model import ApplianceModelConfig

    import pandas as pd

    if appliance_config is None:
        appliance_config = ApplianceModelConfig()
    records = [
        {
            "appliance_id": spec.appliance_id,
            "appliance_priority": spec.priority,
            "appliance_deferable": spec.deferable,
            "appliance_power_kw": spec.power_kw,
            "appliance_usual_start_min": spec.usual_start_min,
            "appliance_usual_end_min": spec.usual_end_min,
        }
        for spec in appliance_config.appliances
    ]
    return pd.DataFrame(records)[TWIN_APPLIANCE_COLUMNS]


def get_twin_ev_max_charge_kw(ev_config=None):
    """Return the authoritative twin EV charging rate in kW.

    This is EVModelConfig.max_charge_kw (7.0 kW default) — the value that
    clamps simulated charging power in EVModel.tick(). Custom EV
    configurations are preserved by passing the twin's own ev_config
    through unchanged.
    """
    from digital_twin.ev_model import EVModelConfig

    if ev_config is None:
        ev_config = EVModelConfig()
    return float(ev_config.max_charge_kw)


def _coerce_roster_records(appliances_df):
    """Return roster as a list of plain dicts, or None when no roster given.

    Accepts the generator/twin appliances DataFrame, a bare list of row
    dicts, or an ApplianceModelConfig (using its .appliances specs).
    None is reserved exclusively for "caller provided no roster" (legacy
    fallback). A non-None roster that cannot be converted raises TypeError
    instead of silently falling back.
    """
    if appliances_df is None:
        return None
    if hasattr(appliances_df, "to_dict"):
        try:
            return appliances_df.to_dict(orient="records")
        except Exception as exc:
            raise TypeError(
                "appliances roster with to_dict() could not be converted "
                "to records; expected a DataFrame."
            ) from exc
    if isinstance(appliances_df, (list, tuple)):
        if all(isinstance(entry, dict) for entry in appliances_df):
            return list(appliances_df)
        raise TypeError(
            "appliances roster list/tuple must contain only row dicts."
        )
    specs = getattr(appliances_df, "appliances", None)
    if isinstance(specs, (list, tuple)):
        records = []
        for spec in specs:
            if isinstance(spec, dict):
                records.append(dict(spec))
                continue
            records.append(
                {
                    "appliance_id": getattr(spec, "appliance_id", "unknown"),
                    "appliance_priority": getattr(spec, "priority", None),
                    "appliance_deferable": getattr(spec, "deferable", False),
                    "appliance_power_kw": getattr(spec, "power_kw", None),
                    "appliance_usual_start_min": getattr(
                        spec, "usual_start_min", None
                    ),
                    "appliance_usual_end_min": getattr(spec, "usual_end_min", None),
                }
            )
        return records
    raise TypeError(
        f"Unsupported appliances roster type: {type(appliances_df).__name__}; "
        "expected a DataFrame, a list of row dicts, or an ApplianceModelConfig."
    )


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


def _build_agents(appliances_df, ev_max_charge_kw=None):
    """One singleton each for solar/battery/EV plus one agent per
    appliance roster entry (or fallback agents when no roster is
    supplied, e.g. legacy fake rows).

    The roster may be a generator/twin appliances DataFrame, a list of
    row dicts, or an ApplianceModelConfig; all are coerced to the same
    record list. EVAgent's max_charge_kw defaults to the roster's
    ev_charger appliance_power_kw (generator path), but an explicit
    ev_max_charge_kw overrides it — the twin path must pass
    EVModelConfig.max_charge_kw (via get_twin_ev_max_charge_kw) since
    that value clamps twin charging while the roster rating does not.
    A non-None roster of an unsupported type raises TypeError; a valid
    roster missing the ev_charger row raises ValueError; only
    appliances_df=None takes the legacy no-roster path which uses
    LEGACY_EV_MAX_CHARGE_KW_FALLBACK explicitly."""
    solar = SolarAgent("solar_1")
    battery = BatteryAgent("battery_1")
    roster = _coerce_roster_records(appliances_df)
    if roster is not None:
        if ev_max_charge_kw is not None:
            ev_power = ev_max_charge_kw
        else:
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


def run_simulation(dataset, allocator, appliances_df=None, ev_max_charge_kw=None):
    """Run an allocator once for every row/timestep in the dataset.

    Agents are instantiated once (not per timestep), refreshed via
    update_state(row) each step, and their get_demand_kw() values form
    the allocator's demand dict. available_power comes from the row's
    solar_gen_kw (default 5.0 for legacy rows missing that key).

    The allocator is passed as an argument so the baseline can later be
    replaced by a smarter allocation strategy without changing this loop.
    ev_max_charge_kw optionally overrides the roster's ev_charger rating
    for the EVAgent; twin callers should pass get_twin_ev_max_charge_kw().
    """
    rows, resolved_appliances_df = _normalize_inputs(dataset, appliances_df)
    agents = _build_agents(resolved_appliances_df, ev_max_charge_kw=ev_max_charge_kw)

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
