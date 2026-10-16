"""Data layer for the AgentGrid dashboard.

Pure Python (no Streamlit imports) so it can be tested in CI. It runs the
closed-loop twin for a named offline scenario and turns the step records into
tidy tables for charts, a per-step explanation, and outcome metrics.
"""

from dataclasses import dataclass

import pandas as pd

from digital_twin.faults import DeviceFaultConfig, FaultConfig
from explain.facts import build_step_facts
from scenarios import CATALOG, build_twin, list_scenarios
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.metrics import METRICS, score_run

STRATEGIES = {
    "hybrid": ("Hybrid arbiter", hybrid_allocate),
    "baseline": ("Proportional baseline", proportional_adapter),
}

SCENARIO_NOTES = {
    "demo_sunny_home": "Sunny Vellore day, 60 m2 array (12 kWp), battery 50% / 12 kWh",
    "clear_day": "Sunny Vellore day (25 Jan 2024), battery 50%",
    "monsoon_cloudy": "Overcast monsoon day (4 Dec 2023), weak solar",
    "heatwave": "Hot pre-monsoon day (1 May 2024), panel derating",
    "low_battery_start": "Clear day, battery starts at the 20% reserve",
    "degraded_battery": "Clear day, battery capacity cut to 6 kWh",
    "overnight_ev_cloudy": "Cloudy day with an overnight-charging EV",
    "three_days": "Clear day repeated for 72 hours",
}

# Fault presets used in the demo. Each maps to (sensor FaultConfig or None,
# device DeviceFaultConfig or None); step numbers are 5-minute steps (144 = noon).
FAULT_PRESETS = {
    "none": ("No faults", None, None),
    "inverter_trip": (
        "Inverter trip, 12:00-13:00",
        None,
        DeviceFaultConfig(enabled=True, faults=[
            {"type": "inverter_trip", "start_step": 144, "end_step": 156},
        ]),
    ),
    "charger_fault": (
        "EV charger fault, 10:00-12:00",
        None,
        DeviceFaultConfig(enabled=True, faults=[
            {"type": "charger_fault", "start_step": 120, "end_step": 144},
        ]),
    ),
    "battery_derate": (
        "Battery derated to 30%, 18:00-22:00",
        None,
        DeviceFaultConfig(enabled=True, faults=[
            {"type": "battery_derate", "start_step": 216, "end_step": 264, "factor": 0.3},
        ]),
    ),
    "soc_sensor_dropout": (
        "Battery SOC sensor dropout, 08:00-08:30",
        FaultConfig(enabled=True, seed=1, faults=[
            {"type": "dropout", "key": "battery_soc_pct", "start_step": 96, "end_step": 102},
        ]),
        None,
    ),
}

METRIC_LABELS = {
    "critical_unmet_kwh": "Critical unmet (kWh)",
    "priority_1_unmet_kwh": "Priority-1 unmet (kWh)",
    "flexible_unmet_kwh": "Flexible unmet (kWh)",
    "ev_shortfall_pct_at_departure": "EV shortfall at departure (pp)",
    "appliance_cycles_completed": "Appliance cycles completed",
    "appliance_cycles_incomplete_at_window_end": "Cycles missing their window",
    "blocked_starts": "Blocked appliance starts",
    "curtailed_solar_kwh": "Curtailed solar (kWh)",
    "battery_reserve_violation_steps": "Battery reserve violations (steps)",
    "battery_charge_kwh": "Battery charged (kWh)",
    "battery_discharge_kwh": "Battery discharged (kWh)",
    "energy_balance_max_error_kw": "Max power-balance error (kW)",
}

# For these metrics a lower value is better (used for delta colouring).
LOWER_IS_BETTER = {
    "critical_unmet_kwh", "priority_1_unmet_kwh", "flexible_unmet_kwh",
    "ev_shortfall_pct_at_departure", "appliance_cycles_incomplete_at_window_end",
    "blocked_starts", "curtailed_solar_kwh", "battery_reserve_violation_steps",
    "energy_balance_max_error_kw",
}

ESSENTIAL_IDS = ("household_base_load",)  # plus every priority 1-2 appliance
GROUPS = ("Essentials", "EV", "Battery charging", "Flexible appliances")

_URGENCY_TIE = 1e-9  # same band tolerance as hybrid_allocate / explain.facts


def _deferral_text(agent: dict, agents: list[dict]) -> str:
    """Name the served flexible loads this agent actually waited behind, and why.

    Hybrid order is urgency (descending), then priority number (ascending)
    within an urgency band, so a deferral is either "more urgent" or "same
    urgency, higher priority" - never claim the first when only the second
    holds.
    """
    served = [a for a in agents
              if a is not agent and a["priority"] > 2 and a["allocated_kw"] > 1e-9]
    more_urgent = [a for a in served if a["urgency"] > agent["urgency"] + _URGENCY_TIE]
    if more_urgent:
        names = ", ".join(agent_label(a["agent_id"]) for a in more_urgent[:2])
        return f"waited behind more urgent loads ({names})"
    same_band = [a for a in served
                 if abs(a["urgency"] - agent["urgency"]) <= _URGENCY_TIE
                 and a["priority"] < agent["priority"]]
    if same_band:
        first = same_band[0]
        return (f"same urgency ({agent['urgency']:.2f}) as {agent_label(first['agent_id'])}, "
                f"which went first on priority ({first['priority']} vs {agent['priority']})")
    return "waited behind earlier urgency and priority groups"


def _reason_text(agent: dict, agents: list[dict], strategy: str) -> str:
    """Policy-faithful wording for an explain/facts reason code."""
    reason, served, demand = agent["reason"], agent["allocated_kw"], agent["demand_kw"]
    full = served >= demand - 1e-9
    if reason == "missed_window":
        return "outside its usual running window"
    if full:
        if strategy == "hybrid" and reason == "critical_first":
            return "served in full: essential loads come first when power is short"
        return "served in full"
    if strategy == "baseline":
        # The proportional baseline gives every load the same fraction of its
        # request, regardless of priority or urgency.
        return "every load received the same fraction of its request (proportional baseline)"
    if reason == "critical_first":
        return "held back because higher-priority essential loads came first"
    if reason == "deferred_lower_urgency":
        return _deferral_text(agent, agents)
    if reason == "shared_proportionally":
        if agent["priority"] <= 2:
            return f"shared what was left proportionally with other priority-{agent['priority']} loads"
        return "shared what was left proportionally with loads of the same urgency and priority"
    return "served in full"


@dataclass(frozen=True)
class RunKey:
    scenario: str
    strategy: str
    fault: str = "none"
    policy: str = "hold_last"


def run(key: RunKey) -> list[dict]:
    """Run one closed-loop simulation and return its step records."""
    if key.scenario not in CATALOG:
        raise ValueError(f"unknown scenario {key.scenario!r}; choose from {list_scenarios()}")
    if key.strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {key.strategy!r}")
    if key.fault not in FAULT_PRESETS:
        raise ValueError(f"unknown fault preset {key.fault!r}")
    _, sensor, device = FAULT_PRESETS[key.fault]
    twin = build_twin(key.scenario, closed_loop=True)
    kwargs = {}
    if sensor is not None:
        kwargs.update(sensor_faults=sensor, on_bad_reading=key.policy)
    if device is not None:
        kwargs["device_faults"] = device
    return run_closed_loop(
        twin, STRATEGIES[key.strategy][1], steps=CATALOG[key.scenario]["steps"], **kwargs
    )


def agent_label(agent_id: str) -> str:
    """Human-readable agent name, e.g. appliance_pool_pump -> Pool pump."""
    special = {"ev_1": "EV", "battery_1": "Battery", "solar_1": "Solar",
               "household_base_load": "Base load"}
    if agent_id in special:
        return special[agent_id]
    name = agent_id.removeprefix("appliance_").replace("_", " ")
    return name[:1].upper() + name[1:]


def _clock(index: int, record: dict) -> str:
    day = int(index * record["dt_min"] // 1440) + 1
    return f"Day {day} {record['timestamp']}"


def _start(start):
    return pd.Timestamp(start or "2024-01-01")


def timeseries(records: list[dict], start=None) -> pd.DataFrame:
    """One row per step: power flows, storage state and unmet demand.

    ``start`` (e.g. the scenario's start_date) anchors the ``time`` column.
    """
    origin = _start(start)
    rows = []
    for index, record in enumerate(records):
        priorities = record["priorities"]
        hours = record["dt_min"] / 60
        loads = {key: value for key, value in record["agents"].items()
                 if key not in ("solar_1", "battery_1")}
        critical_unmet = sum(max(0.0, v["demand_kw"] - v["allocated_kw"])
                             for k, v in loads.items() if priorities.get(k, 5) <= 2)
        flexible_unmet = sum(max(0.0, v["demand_kw"] - v["allocated_kw"])
                             for k, v in loads.items() if priorities.get(k, 5) > 2)
        battery_kw = record["battery_kw"]
        rows.append({
            "step": index,
            "minute": index * record["dt_min"],
            "time": origin + pd.Timedelta(minutes=index * record["dt_min"]),
            "clock": _clock(index, record),
            "timestamp": record["timestamp"],
            "solar_kw": record["solar_kw"],
            "delivered_kw": sum(v["allocated_kw"] for v in loads.values()),
            "demand_kw": sum(v["demand_kw"] for v in loads.values()),
            "battery_charge_kw": max(0.0, battery_kw),
            "battery_discharge_kw": max(0.0, -battery_kw),
            "curtailed_kw": record["curtailed_kw"],
            "critical_unmet_kw": critical_unmet,
            "flexible_unmet_kw": flexible_unmet,
            "critical_unmet_kwh": critical_unmet * hours,
            "battery_soc_pct": record["battery_soc_pct"],
            "battery_reserve_pct": record.get("battery_reserve_pct", 20.0),
            "ev_soc_pct": record["ev_soc_pct"],
            "ev_plugged_in": record["ev_plugged_in"],
            "mode": record["mode"],
            "faults": ", ".join(sorted(
                {f["fault_type"] for f in record.get("device_faults", [])}
                | {f.get("fault_type", "sensor") for f in record.get("sensor_faults", [])}
            )),
        })
    return pd.DataFrame(rows)


def allocation_groups(records: list[dict], start=None) -> pd.DataFrame:
    """Delivered kW per step in a fixed set of groups for stacked charts.

    Groups are Essentials (base load and priority 1-2 appliances), EV,
    Battery charging and Flexible appliances, so the stacked chart stays
    within four validated colour slots; per-appliance detail is in
    step_table().
    """
    origin = _start(start)
    rows = []
    for index, record in enumerate(records):
        priorities = record["priorities"]
        groups = dict.fromkeys(GROUPS, 0.0)
        for agent_id, values in record["agents"].items():
            if agent_id == "solar_1":
                continue
            if agent_id == "battery_1":
                groups["Battery charging"] += max(0.0, record["battery_kw"])
            elif agent_id == "ev_1":
                groups["EV"] += values["allocated_kw"]
            elif agent_id in ESSENTIAL_IDS or priorities.get(agent_id, 5) <= 2:
                groups["Essentials"] += values["allocated_kw"]
            else:
                groups["Flexible appliances"] += values["allocated_kw"]
        for group, kw in groups.items():
            rows.append({"step": index, "minute": index * record["dt_min"],
                         "time": origin + pd.Timedelta(minutes=index * record["dt_min"]),
                         "clock": _clock(index, record), "group": group, "kw": kw})
    return pd.DataFrame(rows)


def step_table(record: dict) -> pd.DataFrame:
    """Per-agent bids and outcome for one step, most important first."""
    priorities = record["priorities"]
    rows = []
    for agent_id, values in record["agents"].items():
        if agent_id == "solar_1":
            continue
        demand = values["demand_kw"]
        allocated = values["allocated_kw"]
        if agent_id == "battery_1":
            allocated = max(0.0, record["battery_kw"])
        rows.append({
            "Agent": agent_label(agent_id),
            "Priority": priorities.get(agent_id, 5),
            "Urgency": round(values["urgency"], 2),
            "Requested (kW)": round(demand, 3),
            "Received (kW)": round(allocated, 3),
            "Served (%)": round(100 * allocated / demand, 1) if demand > 1e-9 else None,
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["Priority", "Urgency"], ascending=[True, False]).reset_index(drop=True)


def _allocator_inputs(record: dict):
    """Rebuild the bids and allocation the arbiter actually saw at this step."""
    bids, allocation = [], {}
    for agent_id, values in record["agents"].items():
        if agent_id == "battery_1":
            if record["mode"] != "charge":
                continue  # outside charge mode the battery is a source, not a bidder
            demand = values["allocated_kw"]  # charge bid was capped to what fits
        else:
            demand = values["demand_kw"]
        bids.append({"agent_id": agent_id, "demand_kw": demand, "urgency": values["urgency"]})
        allocation[agent_id] = values["allocated_kw"]
    return bids, allocation


def explain_step(record: dict, strategy: str) -> list[str]:
    """Plain-English explanation lines for one step.

    Uses the facts-first explainer (explain/facts.py) so the wording follows
    the actual policy. If the step cannot be reproduced exactly (for example
    a sensor fault excluded an agent), falls back to describing the outcome.
    """
    mode_text = {
        "charge": f"Solar exceeded demand, so the battery charged at {record['battery_kw']:.2f} kW.",
        "discharge": f"Demand exceeded solar, so the battery supplied {-record['battery_kw']:.2f} kW.",
        "idle": "The battery stayed idle this step.",
    }[record["mode"]]
    header = (f"{record['timestamp']}: solar {record['solar_kw']:.2f} kW, "
              f"available {record['available_kw']:.2f} kW. {mode_text}")
    lines = [header]
    bids, allocation = _allocator_inputs(record)
    try:
        facts = build_step_facts(record["timestamp"], record["available_kw"], bids,
                                 record["priorities"], allocation, strategy=strategy)
        agents = sorted(facts["agents"], key=lambda a: (a["priority"], -a["urgency"]))
        for agent in agents:
            if agent["agent_id"] == "solar_1" or agent["reason"] == "no_demand":
                continue
            served, demand = agent["allocated_kw"], agent["demand_kw"]
            why = _reason_text(agent, facts["agents"], strategy)
            share = "" if served >= demand - 1e-9 else f" ({100 * served / demand:.0f}%)"
            lines.append(f"{agent_label(agent['agent_id'])}: {served:.2f} of "
                         f"{demand:.2f} kW{share}, {why}.")
    except ValueError:
        for row in step_table(record).to_dict(orient="records"):
            if row["Requested (kW)"] <= 1e-9:
                continue
            lines.append(f"{row['Agent']}: {row['Received (kW)']:.2f} of "
                         f"{row['Requested (kW)']:.2f} kW received.")
        lines.append("(Exact policy reasons unavailable for this step: a fault changed the inputs.)")
    for fault in record.get("device_faults", []):
        lines.append(f"Device fault active: {fault['fault_type'].replace('_', ' ')}.")
    for sub in record.get("substitutions", []):
        lines.append(f"Sensor reading for {sub['key']} was bad; runner used policy "
                     f"'{sub['policy']}' for {agent_label(sub['agent_id'])}.")
    return lines


def metrics(records: list[dict]) -> dict:
    """Outcome metrics for one run (see simulation/metrics.py)."""
    return score_run(records)


def comparison(scenario: str, fault: str = "none", policy: str = "hold_last") -> pd.DataFrame:
    """Hybrid vs baseline metrics for one scenario, with the difference."""
    hybrid = metrics(run(RunKey(scenario, "hybrid", fault, policy)))
    baseline = metrics(run(RunKey(scenario, "baseline", fault, policy)))
    return pd.DataFrame([
        {"Metric": METRIC_LABELS[key], "key": key, "Hybrid": hybrid[key],
         "Baseline": baseline[key],
         # Collapse floating-point noise so the table never shows "-0.000".
         "Hybrid - baseline": 0.0 if abs(hybrid[key] - baseline[key]) < 1e-9
         else hybrid[key] - baseline[key]}
        for key in METRICS
    ])
