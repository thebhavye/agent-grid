"""Deterministic multi-day data from committed, offline battery operation."""

from dataclasses import replace

import pandas as pd

from digital_twin.battery_model import BatteryModel
from scenarios import build_twin
from simulation.closed_loop import run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate


def generate_battery_dataset(scenarios, days, seed) -> pd.DataFrame:
    """Return per-step terminal power, state and cumulative aging exposures.

    Calendar days repeat the saved weather and EV schedule, while appliance
    RNG advances naturally. The seed controls EV and appliance streams. Health
    is synthetic ground truth from illustrative assumptions, not measured SoH.
    Daily rows are available separately through summarize_days(frame).
    """
    names = list(scenarios)
    if not names or len(names) != len(set(names)):
        raise ValueError("scenarios must be nonempty and unique")
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rows = []
    for name in names:
        twin = build_twin(name, closed_loop=True)
        twin._battery = BatteryModel(replace(twin._battery.config, degradation=True))
        twin._ev.reset(seed=seed)
        twin._appliances.reset(seed=seed)
        twin._initialize_models()
        dt = twin._time_delta_min
        steps = days * 1440 / dt
        if not steps.is_integer():
            raise ValueError("configured timestep must divide the requested days")
        records = run_closed_loop(twin, hybrid_allocate, steps=int(steps))
        for step, r in enumerate(records):
            rows.append({
                "scenario": name, "seed": seed, "step": step,
                "day": int(step * dt // 1440), "dt_min": r["dt_min"],
                "soc_pct": r["battery_soc_pct"], "start_soc_pct": r["battery_start_soc_pct"],
                "power_kw": r["battery_kw"], "c_rate": r["battery_c_rate"],
                "temperature_c": r["battery_temperature_c"],
                "ambient_temperature_c": r["ambient_temperature_c"],
                "throughput_kwh": r["battery_throughput_kwh"],
                "cycle_exposure": r["battery_cycle_exposure"],
                "calendar_exposure": r["battery_calendar_exposure"],
                "health_pct": r["battery_health_pct"],
                "initial_health_pct": r["battery_initial_health_pct"],
                "initial_temperature_c": r["battery_initial_temperature_c"],
                "capacity_kwh": r["battery_capacity_kwh"],
            })
    return pd.DataFrame(rows)


def summarize_days(frame) -> pd.DataFrame:
    """Summarize daily exposures and targets, never mix scenario histories."""
    result = []
    for (scenario, seed), group in frame.groupby(["scenario", "seed"], sort=True):
        health = float(group.iloc[0].initial_health_pct)
        temperature = float(group.iloc[0].initial_temperature_c)
        cycle, calendar, throughput = 0.0, 0.0, 0.0
        for day, rows in group.groupby("day", sort=True):
            end = rows.iloc[-1]
            result.append({
                "scenario": scenario, "seed": seed, "day": day,
                "start_health_pct": health, "end_health_pct": end.health_pct,
                "start_temperature_c": temperature,
                "peak_temperature_c": rows.temperature_c.max(),
                "mean_ambient_c": rows.ambient_temperature_c.mean(),
                "max_c_rate": rows.c_rate.max(),
                "cycle_exposure": end.cycle_exposure-cycle,
                "calendar_exposure": end.calendar_exposure-calendar,
                "throughput_kwh": end.throughput_kwh-throughput,
            })
            health, temperature = end.health_pct, end.temperature_c
            cycle, calendar, throughput = end.cycle_exposure, end.calendar_exposure, end.throughput_kwh
    return pd.DataFrame(result)
