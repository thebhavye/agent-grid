"""
AgentGrid - Phase 1 synthetic dataset generator.

Produces one simulated 24h day of smart-home microgrid data at 5-min
resolution: solar generation, battery state, household load, EV charging
trajectory, and an appliance roster (essential + deferable).

Schema is documented in README.md. Format: TWO related tables, not one flat
table - a 288-row timeseries can't hold an 8-row appliance roster without
either duplicating system state per appliance or losing appliance
granularity, so this returns a dict of two DataFrames instead:
    {"timeseries": <288 rows>, "appliances": <6-10 rows>}
Join them only where an agent/allocator actually needs both.

Usage:
    from data.generate_dataset import generate_dataset
    tables = generate_dataset(seed=42)
    timeseries_df = tables["timeseries"]
    appliances_df = tables["appliances"]

Run directly for a quick sanity check + plot:
    python data/generate_dataset.py
"""

import numpy as np
import pandas as pd

N_STEPS = 288          # 24h at 5-min resolution
STEP_MIN = 5
MINUTES_PER_DAY = 24 * 60


# ---------------------------------------------------------------------------
# Solar generation
# ---------------------------------------------------------------------------

def _generate_solar(rng, minutes):
    """Bell curve peaking at solar noon, zero outside daylight, plus
    randomized multiplicative cloud dips (0.3-0.7x for 10-30 min) so
    urgency scoring later has real volatility to react to, not a smooth
    sine wave."""
    peak_kw = rng.uniform(4.5, 6.0)
    solar_noon_min = 12 * 60 + rng.uniform(-20, 20)
    sigma = rng.uniform(140, 170)  # controls day length / steepness

    gen = peak_kw * np.exp(-0.5 * ((minutes - solar_noon_min) / sigma) ** 2)

    # hard-clip pre-sunrise / post-sunset (bell tails would otherwise
    # trickle small positive values all night)
    sunrise = solar_noon_min - rng.uniform(340, 370)
    sunset = solar_noon_min + rng.uniform(340, 370)
    gen[(minutes < sunrise) | (minutes > sunset)] = 0.0

    # cloud dips: a handful of random events during daylight hours
    daylight_idx = np.where(gen > 0.05)[0]
    if len(daylight_idx) > 0:
        n_dips = rng.integers(2, 6)
        for _ in range(n_dips):
            start = rng.choice(daylight_idx)
            duration_steps = rng.integers(2, 7)  # 10-30 min in 5-min steps
            end = min(start + duration_steps, N_STEPS)
            dip_factor = rng.uniform(0.3, 0.7)
            gen[start:end] *= dip_factor

    # small sensor-level noise
    gen += rng.normal(0, 0.05, size=N_STEPS)
    return np.clip(gen, 0, None)


# ---------------------------------------------------------------------------
# Household load
# ---------------------------------------------------------------------------

def _generate_load(rng, minutes):
    """Two peaks (7-9am, 6-9pm), low overnight trough, random noise.
    Independent of the solar curve on purpose."""
    base = 0.6  # overnight trough kw

    morning = 1.8 * np.exp(-0.5 * ((minutes - rng.uniform(460, 480)) / 45) ** 2)
    evening = 2.6 * np.exp(-0.5 * ((minutes - rng.uniform(1140, 1170)) / 60) ** 2)

    load = base + morning + evening
    load += rng.normal(0, 0.15, size=N_STEPS)
    return np.clip(load, 0.1, None)


# ---------------------------------------------------------------------------
# EV trajectory
# ---------------------------------------------------------------------------

def _snap_to_grid(minute_value):
    """Round a raw minute-of-day value to the nearest 5-min grid step, so
    a reported event time (e.g. ev_departure_min) always matches the exact
    grid point where the plugged/unplugged transition actually happens."""
    return float(round(minute_value / STEP_MIN) * STEP_MIN)


def _generate_ev(rng, minutes):
    """Randomize across a few plausible plug-in/departure scenarios rather
    than one fixed pattern: overnight charger, work-from-home midday
    charger, or evening-only top-up.

    Modeled as a list of (start_min, end_min, start_soc, target_soc)
    plugged-in segments within this single 24h window. The overnight
    scenario needs TWO segments - the tail of the previous night's charge
    (00:00 -> early morning departure) and the new evening plug-in
    (evening -> midnight) - otherwise the reported departure time has no
    matching plugged-in interval in the data at all.
    """
    scenario = rng.choice(["overnight", "midday", "evening_topup"])

    if scenario == "overnight":
        departure_min = _snap_to_grid(rng.uniform(6 * 60 + 30, 8 * 60 + 30))
        evening_plug_in_min = _snap_to_grid(rng.uniform(19 * 60, 21 * 60 + 30))
        segments = [
            # tail end of previous night's charge, already mostly full
            (0.0, departure_min, rng.uniform(70, 95), rng.uniform(90, 100)),
            # plugs back in that evening after the day's driving
            (evening_plug_in_min, float(MINUTES_PER_DAY), rng.uniform(20, 45), rng.uniform(85, 100)),
        ]
        departure_min_reported = departure_min
    elif scenario == "midday":
        plug_in_min = _snap_to_grid(rng.uniform(9 * 60, 10 * 60 + 30))
        departure_min = _snap_to_grid(rng.uniform(16 * 60, 17 * 60 + 30))
        segments = [(plug_in_min, departure_min, rng.uniform(35, 60), rng.uniform(80, 95))]
        departure_min_reported = departure_min
    else:  # evening_topup
        plug_in_min = _snap_to_grid(rng.uniform(17 * 60 + 30, 19 * 60))
        departure_min = _snap_to_grid(rng.uniform(21 * 60, 22 * 60 + 30))
        segments = [(plug_in_min, departure_min, rng.uniform(50, 70), rng.uniform(75, 90))]
        departure_min_reported = departure_min

    plugged_in = np.zeros(N_STEPS, dtype=bool)
    soc = np.full(N_STEPS, np.nan)

    for seg_start, seg_end, start_soc, target_soc in segments:
        mask = (minutes >= seg_start) & (minutes < seg_end)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        plugged_in[idx] = True
        charge_steps = np.arange(len(idx))
        soc_curve = start_soc + (target_soc - start_soc) * np.minimum(
            charge_steps / max(len(idx) * 0.8, 1), 1.0
        )
        soc[idx] = soc_curve

    # gaps (idle/away, or before the first segment) hold flat at the
    # nearest known value - forward-fill from the last charging step,
    # back-fill anything before the very first segment starts
    soc = pd.Series(soc).ffill().bfill().to_numpy()

    # measurement noise only while actually charging - flat means flat,
    # per the documented "flat while idle/away" contract
    noise = rng.normal(0, 0.3, size=N_STEPS)
    soc = soc + noise * plugged_in
    soc = np.clip(soc, 0, 100)

    departure_min_reported = int(round(departure_min_reported)) % MINUTES_PER_DAY
    return soc, plugged_in, departure_min_reported


# ---------------------------------------------------------------------------
# Battery state (home battery, not EV)
# ---------------------------------------------------------------------------

def _generate_battery(rng, minutes, solar_kw, load_kw):
    start_soc = rng.uniform(35, 65)
    soc = np.zeros(N_STEPS, dtype=float)
    soc[0] = start_soc

    capacity_kwh = rng.uniform(10, 14)
    charge_eff, discharge_eff = 0.95, 0.95

    for i in range(1, N_STEPS):
        surplus_kw = solar_kw[i] - load_kw[i]
        delta_kwh = surplus_kw * (STEP_MIN / 60)
        if delta_kwh > 0:
            soc[i] = soc[i - 1] + (delta_kwh * charge_eff / capacity_kwh) * 100
        else:
            soc[i] = soc[i - 1] + (delta_kwh / discharge_eff / capacity_kwh) * 100
        soc[i] = np.clip(soc[i], 5, 100)

    soc += rng.normal(0, 0.2, size=N_STEPS)
    soc = np.clip(soc, 0, 100)

    # battery health: near-constant across one simulated day, small noise
    health_base = rng.uniform(88, 99)
    health = health_base + rng.normal(0, 0.3, size=N_STEPS)
    health = np.clip(health, 0, 100)

    return soc, health


# ---------------------------------------------------------------------------
# Appliance roster
# ---------------------------------------------------------------------------

def _appliance_roster(rng):
    """6-10 appliances split essential (priority 1-2, deferable=False) vs
    flexible (priority 3-5, deferable=True), each with a rough 'usual'
    run window for the flexible ones (None for essentials, since they
    run all day)."""
    roster = [
        # essential - always-on, no usual window
        dict(appliance_id="fridge", appliance_priority=1, appliance_deferable=False,
             appliance_power_kw=round(rng.uniform(0.10, 0.20), 3),
             appliance_usual_start_min=None, appliance_usual_end_min=None),
        dict(appliance_id="lighting", appliance_priority=2, appliance_deferable=False,
             appliance_power_kw=round(rng.uniform(0.15, 0.35), 3),
             appliance_usual_start_min=None, appliance_usual_end_min=None),
        dict(appliance_id="security_system", appliance_priority=1, appliance_deferable=False,
             appliance_power_kw=round(rng.uniform(0.03, 0.08), 3),
             appliance_usual_start_min=None, appliance_usual_end_min=None),
        # flexible - rough usual-run windows (minutes from midnight)
        dict(appliance_id="ev_charger", appliance_priority=3, appliance_deferable=True,
             appliance_power_kw=round(rng.uniform(3.0, 7.0), 2),
             appliance_usual_start_min=19 * 60, appliance_usual_end_min=7 * 60),
        dict(appliance_id="dishwasher", appliance_priority=4, appliance_deferable=True,
             appliance_power_kw=round(rng.uniform(1.0, 1.5), 2),
             appliance_usual_start_min=20 * 60, appliance_usual_end_min=22 * 60),
        dict(appliance_id="washing_machine", appliance_priority=4, appliance_deferable=True,
             appliance_power_kw=round(rng.uniform(0.8, 1.3), 2),
             appliance_usual_start_min=9 * 60, appliance_usual_end_min=17 * 60),
        dict(appliance_id="water_heater", appliance_priority=3, appliance_deferable=True,
             appliance_power_kw=round(rng.uniform(1.5, 3.0), 2),
             appliance_usual_start_min=6 * 60, appliance_usual_end_min=8 * 60),
        dict(appliance_id="pool_pump", appliance_priority=5, appliance_deferable=True,
             appliance_power_kw=round(rng.uniform(0.7, 1.2), 2),
             appliance_usual_start_min=11 * 60, appliance_usual_end_min=15 * 60),
    ]
    return roster


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_dataset(seed=None):
    """Generate one simulated 24h day of AgentGrid data.

    Returns
    -------
    dict[str, pandas.DataFrame]
        {"timeseries": <288 rows>, "appliances": <6-10 rows>}. See README.md
        for the full column contract.
    """
    rng = np.random.default_rng(seed)

    minutes = np.arange(N_STEPS) * STEP_MIN
    timestamps = [f"{m // 60:02d}:{m % 60:02d}" for m in minutes]

    solar_kw = _generate_solar(rng, minutes)
    load_kw = _generate_load(rng, minutes)
    battery_soc, battery_health = _generate_battery(rng, minutes, solar_kw, load_kw)
    ev_soc, ev_plugged_in, ev_departure_min = _generate_ev(rng, minutes)
    roster = _appliance_roster(rng)

    timeseries = pd.DataFrame({
        "timestamp": timestamps,
        "solar_gen_kw": np.round(solar_kw, 3),
        "battery_soc_pct": np.round(battery_soc, 2),
        "battery_health_pct": np.round(battery_health, 2),
        "household_load_kw": np.round(load_kw, 3),
        "ev_soc_pct": np.round(ev_soc, 2),
        "ev_departure_min": ev_departure_min,
        "ev_plugged_in": ev_plugged_in,
    })

    appliance_cols = [
        "appliance_id", "appliance_priority", "appliance_deferable",
        "appliance_power_kw", "appliance_usual_start_min", "appliance_usual_end_min",
    ]
    appliances_df = pd.DataFrame(roster)[appliance_cols]

    return {"timeseries": timeseries, "appliances": appliances_df}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tables = generate_dataset(seed=1)
    ts = tables["timeseries"]
    appliances = tables["appliances"]

    print("timeseries:", ts.shape)
    print(ts.head())
    print(ts.dtypes)
    print()
    print("appliances:", appliances.shape)
    print(appliances)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(ts["timestamp"], ts["solar_gen_kw"], color="orange", label="solar_gen_kw")
    axes[0].plot(ts["timestamp"], ts["household_load_kw"], color="steelblue", label="household_load_kw")
    axes[0].set_ylabel("kW")
    axes[0].legend()
    axes[0].set_title("Solar generation vs household load")

    axes[1].plot(ts["timestamp"], ts["battery_soc_pct"], color="green", label="battery_soc_pct")
    axes[1].plot(ts["timestamp"], ts["battery_health_pct"], color="gray", linestyle="--", label="battery_health_pct")
    axes[1].set_ylabel("%")
    axes[1].legend()
    axes[1].set_title("Battery SOC / health")

    axes[2].plot(ts["timestamp"], ts["ev_soc_pct"], color="purple", label="ev_soc_pct")
    axes[2].fill_between(ts["timestamp"], 0, 100, where=ts["ev_plugged_in"],
                          color="purple", alpha=0.1, label="plugged in")
    axes[2].set_ylabel("%")
    axes[2].legend()
    axes[2].set_title("EV SOC over the day")

    for ax in axes:
        ax.set_xticks(ts["timestamp"][::24])
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("sanity_plot.png", dpi=120)
    print("Saved sanity_plot.png")