"""
Tests for data.generate_dataset.

Plain assert-based script (mirrors agents/test_agents.py convention) rather
than pytest, so no new dependency for the team to install.

Run from repo root:
    python -m data.test_generate_dataset
"""

import sys

import numpy as np
import pandas as pd

from data.generate_dataset import generate_dataset, N_STEPS

REQUIRED_TIMESERIES_COLS = [
    "timestamp", "solar_gen_kw", "battery_soc_pct", "battery_health_pct",
    "household_load_kw", "ev_soc_pct", "ev_departure_min", "ev_plugged_in",
]
REQUIRED_APPLIANCE_COLS = [
    "appliance_id", "appliance_priority", "appliance_deferable",
    "appliance_power_kw", "appliance_usual_start_min", "appliance_usual_end_min",
]

_failures = []


def check(name, condition):
    """Record and print pass/fail for one test, without stopping the run."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        _failures.append(name)


_OVERNIGHT_SCENARIOS = ("overnight", "midday", "evening_topup")


def _generate_overnight_tables(seed=1):
    """Generate a dataset with the overnight EV scenario forced (test-only).

    The generator picks its EV scenario via rng.choice(["overnight",
    "midday", "evening_topup"]). Patch np.random.default_rng so that exact
    draw returns "overnight" regardless of seed luck; all other RNG draws
    (cloud dips, noise, capacities) pass through untouched. Restores the
    original default_rng before returning, and changes no production code.
    """
    real_default_rng = np.random.default_rng

    class _OvernightRngProxy:
        """Pass-through wrapper forcing only the EV-scenario draw.

        numpy's Generator disallows monkeypatching `choice` on the
        instance, so wrap it instead: every method delegates to the real
        RNG except the scenario draw, which returns "overnight".
        """

        def __init__(self, rng):
            self._rng = rng

        def choice(self, a, *cargs, **ckwargs):
            try:
                if tuple(a) == _OVERNIGHT_SCENARIOS:
                    return "overnight"
            except (TypeError, ValueError):
                pass
            return self._rng.choice(a, *cargs, **ckwargs)

        def __getattr__(self, name):
            return getattr(self.__dict__["_rng"], name)

    def _overnight_rng(*args, **kwargs):
        return _OvernightRngProxy(real_default_rng(*args, **kwargs))

    np.random.default_rng = _overnight_rng
    try:
        return generate_dataset(seed=seed)
    finally:
        np.random.default_rng = real_default_rng


def main():
    # ---- basic shape / return type -------------------------------------
    tables = generate_dataset(seed=42)
    check("returns a dict", isinstance(tables, dict))
    check("dict has exactly 'timeseries' and 'appliances' keys",
          set(tables.keys()) == {"timeseries", "appliances"})

    ts = tables["timeseries"]
    ap = tables["appliances"]
    check("timeseries is a DataFrame", isinstance(ts, pd.DataFrame))
    check("appliances is a DataFrame", isinstance(ap, pd.DataFrame))

    check("timeseries has 288 rows", len(ts) == N_STEPS == 288)
    check("appliances has 6-10 rows", 6 <= len(ap) <= 10)

    check("timeseries has all required columns",
          all(c in ts.columns for c in REQUIRED_TIMESERIES_COLS))
    check("appliances has all required columns",
          all(c in ap.columns for c in REQUIRED_APPLIANCE_COLS))

    # ---- timestamps -------------------------------------------------------
    check("288 unique timestamps", ts["timestamp"].nunique() == 288)
    check("timestamps start at 00:00 and end at 23:55",
          ts["timestamp"].iloc[0] == "00:00" and ts["timestamp"].iloc[-1] == "23:55")

    # ---- no missing data where it shouldn't be ----------------------------
    check("no nulls in timeseries", ts.isnull().sum().sum() == 0)
    check("no nulls in appliance id/priority/deferable/power",
          ap[["appliance_id", "appliance_priority",
              "appliance_deferable", "appliance_power_kw"]].isnull().sum().sum() == 0)

    # ---- value ranges -------------------------------------------------------
    check("solar_gen_kw >= 0", (ts["solar_gen_kw"] >= 0).all())
    check("solar_gen_kw has a real peak (not flat/broken)", ts["solar_gen_kw"].max() > 1.0)
    check("solar_gen_kw is ~0 at midnight (both ends of day)",
          ts["solar_gen_kw"].iloc[0] < 0.5 and ts["solar_gen_kw"].iloc[-1] < 0.5)
    check("household_load_kw >= 0", (ts["household_load_kw"] >= 0).all())
    check("battery_soc_pct within 0-100",
          ts["battery_soc_pct"].between(0, 100).all())
    check("battery_health_pct within 0-100",
          ts["battery_health_pct"].between(0, 100).all())
    check("ev_soc_pct within 0-100", ts["ev_soc_pct"].between(0, 100).all())
    check("ev_departure_min within 0-1439",
          ts["ev_departure_min"].between(0, 1439).all())
    check("ev_plugged_in is boolean dtype", ts["ev_plugged_in"].dtype == bool)
    check("EV is plugged in for at least part of the day",
          ts["ev_plugged_in"].sum() > 0)

    # ---- appliance roster correctness -------------------------------------
    check("appliance_priority within 1-5",
          ap["appliance_priority"].between(1, 5).all())
    check("priority 1-2 appliances are all non-deferable",
          (ap.loc[ap["appliance_priority"] <= 2, "appliance_deferable"] == False).all())  # noqa: E712
    check("priority 3-5 appliances are all deferable",
          (ap.loc[ap["appliance_priority"] >= 3, "appliance_deferable"] == True).all())  # noqa: E712
    check("appliance_power_kw > 0 for every appliance",
          (ap["appliance_power_kw"] > 0).all())
    check("appliance_id values are unique",
          ap["appliance_id"].nunique() == len(ap))
    check("essential appliances have no usual window (NaN)",
          ap.loc[~ap["appliance_deferable"], "appliance_usual_start_min"].isnull().all())
    check("flexible appliances have a usual window (not NaN)",
          ap.loc[ap["appliance_deferable"], "appliance_usual_start_min"].notnull().all())
    check("at least one essential and one flexible appliance present",
          (~ap["appliance_deferable"]).any() and ap["appliance_deferable"].any())

    # ---- reproducibility ----------------------------------------------------
    tables_a = generate_dataset(seed=123)
    tables_b = generate_dataset(seed=123)
    tables_c = generate_dataset(seed=999)
    check("same seed produces identical timeseries",
          tables_a["timeseries"].equals(tables_b["timeseries"]))
    check("same seed produces identical appliances",
          tables_a["appliances"].equals(tables_b["appliances"]))
    check("different seed produces different timeseries",
          not tables_a["timeseries"].equals(tables_c["timeseries"]))

    # ---- runs without a seed too --------------------------------------------
    try:
        unseeded = generate_dataset()
        check("runs without a seed and returns valid shape",
              len(unseeded["timeseries"]) == 288 and 6 <= len(unseeded["appliances"]) <= 10)
    except Exception as e:
        check(f"runs without a seed (raised {type(e).__name__})", False)

    # ---- seed=0 regression: falsy seed must not be treated as "no seed" ----
    tables_zero_a = generate_dataset(seed=0)
    tables_zero_b = generate_dataset(seed=0)
    check("seed=0 is reproducible (not silently treated as unseeded)",
          tables_zero_a["timeseries"].equals(tables_zero_b["timeseries"]))

    # ---- timestamp ordering / spacing --------------------------------------
    parsed_minutes = ts["timestamp"].apply(
        lambda s: int(s[:2]) * 60 + int(s[3:5])
    )
    check("timestamps strictly increasing", parsed_minutes.is_monotonic_increasing)
    check("timestamps are exactly 5 minutes apart",
          (parsed_minutes.diff().dropna() == 5).all())

    # ---- schema boundary: appliances table must not carry a timestamp -----
    check("appliances table has no timestamp column (two-table contract)",
          "timestamp" not in ap.columns)

    # ---- EV departure time is a single constant-per-day value -------------
    check("ev_departure_min is constant across the whole day",
          ts["ev_departure_min"].nunique() == 1)

    # ---- dtypes -------------------------------------------------------------
    check("solar_gen_kw is float dtype",
          pd.api.types.is_float_dtype(ts["solar_gen_kw"]))
    check("appliance_priority is integer dtype",
          pd.api.types.is_integer_dtype(ap["appliance_priority"]))
    check("appliance_deferable is boolean dtype",
          pd.api.types.is_bool_dtype(ap["appliance_deferable"]))

    # ---- cloud dips actually present (not a smooth sine) -------------------
    # a real dip shows up as a sharp negative jump followed by recovery
    # during daylight hours, not just gradual rise/fall
    solar_diffs = ts["solar_gen_kw"].diff().dropna()
    check("solar curve has at least one sharp dip (cloud noise present)",
          (solar_diffs < -0.15).any())

    # ---- household load has two distinct peaks (morning + evening) ---------
    morning_window = ts[(parsed_minutes >= 6 * 60) & (parsed_minutes <= 10 * 60)]
    evening_window = ts[(parsed_minutes >= 17 * 60) & (parsed_minutes <= 21 * 60)]
    overnight_window = ts[(parsed_minutes >= 1 * 60) & (parsed_minutes <= 4 * 60)]
    check("household load morning peak exceeds overnight trough",
          morning_window["household_load_kw"].max() >
          overnight_window["household_load_kw"].max())
    check("household load evening peak exceeds overnight trough",
          evening_window["household_load_kw"].max() >
          overnight_window["household_load_kw"].max())

    # ---- EV idle SOC must be genuinely flat (regression: noise used to be
    # applied even while unplugged, contradicting the documented contract) --
    plugged = ts["ev_plugged_in"].to_numpy()
    soc_vals = ts["ev_soc_pct"].to_numpy()
    idle_flat = True
    i = 0
    while i < len(plugged):
        if not plugged[i]:
            j = i
            while j < len(plugged) and not plugged[j]:
                j += 1
            if soc_vals[i:j].max() - soc_vals[i:j].min() > 1e-6:
                idle_flat = False
                break
            i = j
        else:
            i += 1
    check("ev_soc_pct is exactly flat during every idle/unplugged stretch", idle_flat)

    # ---- overnight EV scenario: departure_min must correspond to a real
    # plugged->unplugged transition in the data (regression: it used to be
    # a phantom value with no matching interval at all) ----------------------
    # Forced overnight tables (Issue #11): never depend on which scenario
    # seed=42 happens to produce, otherwise these checks silently skip.
    overnight_tables = _generate_overnight_tables(seed=1)
    ots = overnight_tables["timeseries"]
    check("forced overnight dataset is actually overnight (plugged in at midnight)",
          bool(ots["ev_plugged_in"].iloc[0]))
    o_parsed_minutes = ots["timestamp"].apply(
        lambda s: int(s[:2]) * 60 + int(s[3:5])
    )
    o_dep = int(ots["ev_departure_min"].iloc[0])
    # find the actual grid index where departure occurs, using real
    # timestamp minutes rather than floor-dividing dep (dep isn't
    # necessarily grid-aligned, so floor division can land one bucket early)
    o_after_dep_idx = int(np.searchsorted(o_parsed_minutes.to_numpy(), o_dep, side="left"))
    check("overnight scenario: EV is plugged in on the step just before departure_min",
          o_after_dep_idx == 0 or bool(ots["ev_plugged_in"].iloc[o_after_dep_idx - 1]))
    check("overnight scenario: EV is unplugged on the step at/after departure_min",
          o_after_dep_idx >= N_STEPS or not bool(ots["ev_plugged_in"].iloc[o_after_dep_idx]))

    # ---- summary --------------------------------------------------------------
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()