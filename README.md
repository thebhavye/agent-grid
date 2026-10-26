# AgentGrid

Multi-agent simulated smart-home microgrid — urgency bidding, hybrid arbitration and closed-loop evaluation, orchestrated with LangGraph. PINN battery health, SHAP explainability and an edge LLM are planned (see Roadmap).

AgentGrid is a simulated smart-home microgrid where autonomous agents (solar, battery, EV, appliances) score their own power urgency and a shared allocator arbitrates limited solar power between them.

This repository is private pending a planned patent filing.

## Roadmap

| # | Phase | Status |
|---|-------|--------|
| 1 | Urgency-scoring agents | DONE (feature selection planned: #86) |
| 2 | LangGraph multi-agent bidding + arbiter | DONE: hybrid arbiter + deterministic graph with closed-loop record parity |
| 3 | Synthetic dataset, digital twin + closed-loop evaluation | DONE: offline scenarios, paired hybrid vs baseline metrics |
| 4 | LLM-generated allocation explanations | PARTIAL: facts-first explanations shipped; LLM renderer awaits a privacy decision (#59) |
| 5 | PINN for battery SoH/thermal risk | opt-in synthetic degradation data and empirical baseline (#69); PINN remains #85 |
| 6 | Edge quantized LLM for fault interpretation | planned (#84); labelled fault data exists |
| 7 | SHAP explainability | planned (#83) |

An interactive demo dashboard (Streamlit) is also implemented; see [Dashboard](#dashboard-demo-ui).

## Deterministic Bidding Graph

`orchestration.graph.run_graph_closed_loop(twin, allocator, steps=288)` runs
independent `bid_<agent_id>` nodes, joins them at `battery_mode`, then executes
`arbiter` and `commit`. It uses the same allocation phases and records as
`run_closed_loop`; no LLM is involved and remote tracing is disabled.

```python
from orchestration.graph import run_graph_closed_loop
from scenarios import build_twin
from simulation.hybrid_arbiter import hybrid_allocate

records = run_graph_closed_loop(build_twin("clear_day", closed_loop=True), hybrid_allocate)
```

LangGraph is pinned in `requirements.txt`; its imports stay in `orchestration/`.
The plain runner remains available without importing the graph package.

## Running tests

From the repository root (CI uses Python 3.13):

Dependencies are pinned in `requirements.txt`; bump pins from a fresh Python 3.13 environment's `pip freeze` and rerun the offline suite and `ruff check .`.

```bash
python -m pip install -r requirements.txt
python -m pytest . -m "not live"
```

`-m "not live"` excludes tests marked `live`, which call the external weather
API. CI runs this offline suite on pushes and pull requests targeting `main`.

To run only the Digital Twin tests:

```bash
python -m pytest digital_twin/ -v -m "not live"
```

Optional diagnostic scripts (the dataset smoke script also runs within the full suite):

```bash
python -m agents.test_agents
python -m data.test_generate_dataset
```

## Generating the dataset

```python
from data.generate_dataset import generate_dataset
tables = generate_dataset(seed=42)
timeseries_df = tables["timeseries"]  # 288 rows, 5-min steps over 24h
appliances_df = tables["appliances"]  # 8-row appliance roster
```

Or run `python data/generate_dataset.py` directly for a sanity check + plot
(solar/load/EV SOC over the day, saved as `sanity_plot.png`).

## Ownership

- `agents/` — Ayush
- `data/` + integration — Bhavye


## Design history

- `decisions.md` — why key choices were made (allocator surplus policy, urgency formulas, missed-window tracking)
- `known-limitations.md` — honest list of what's hardcoded, unmodeled, or deferred to later phases

## Dataset Schema

**Format decision: two related tables, not one flat table.** A single 288-row timeseries table
can't hold an 8-row appliance roster without either duplicating system state 8x per timestamp
or losing appliance granularity. So the generator returns:

```python
{
    "timeseries": timeseries_df,   # 288 rows, one per 5-min step
    "appliances": appliances_df,   # 6-10 rows, one per appliance
}
```

Join them (`timeseries` cross `appliances`, or per-appliance filtered) only at the point where
an agent/allocator actually needs both — don't materialize the join in the stored dataset.

**`timeseries`** — 288 rows, one simulated day, 5-minute resolution (`00:00` → `23:55`)

| column                | type        | notes                                                            |
|------------------------|-------------|-------------------------------------------------------------------|
| `timestamp`            | str `HH:MM` | 288 unique values/day, 5-min steps                                |
| `solar_gen_kw`         | float       | bell curve peaking at solar noon, 0 outside daylight, cloud dips  |
| `battery_soc_pct`      | float 0–100 | charges on solar surplus, drains on evening peak                  |
| `battery_health_pct`   | float 0–100 | near-constant per day (slow degradation, not simulated per-day)   |
| `household_load_kw`    | float       | morning (7–9am) + evening (6–9pm) peaks, overnight trough          |
| `ev_soc_pct`           | float 0–100 | rises while plugged in, flat while idle/away                      |
| `ev_departure_min`     | int         | minutes from midnight; constant for the day, randomized per run   |
| `ev_plugged_in`        | bool        | derived from randomized plug-in/departure scenario                 |

**`appliances`** — 6–10 rows, one per appliance (static roster/contract, not a timeseries)

| column                  | type  | notes                                                  |
|-------------------------|-------|---------------------------------------------------------|
| `appliance_id`          | str   | e.g. `fridge`, `ev_charger`, `dishwasher`                |
| `appliance_priority`    | int   | 1 = critical … 5 = deferable                             |
| `appliance_deferable`   | bool  | False for priority 1–2, True for priority 3–5            |
| `appliance_power_kw`    | float | rated draw when active                                    |
| `appliance_usual_start_min` | int | minutes from midnight; rough usual-run window start (flexible appliances only, `None` for essentials) |
| `appliance_usual_end_min`   | int | minutes from midnight; rough usual-run window end (flexible appliances only, `None` for essentials)   |

Generator: `data/generate_dataset.py`, entry point `generate_dataset(seed=None) -> dict[str, pd.DataFrame]`.
Sanity plot: run `python data/generate_dataset.py` directly → solar/load/EV SOC over the day.

### Sensor fault datasets

`digital_twin.faults` adds optional, deterministic sensor faults without changing
the clean twin or generator rows. Configure `FaultConfig(enabled=True, seed=0,
faults=[...])` and pass rows (a list of dictionaries or a timeseries DataFrame)
to `generate_fault_dataset(rows, config, output_path="faults.csv")`.
Supported fault types are `dropout`, `nan`, `stuck`, `spike`, and `drift`.
Fixed windows use `start_step` and exclusive `end_step`; probabilistic faults use
`probability` and `duration_range`.
Apply rows in chronological step order; use a new injector to replay a run.
Probability events are generated as needed, with no fixed dataset-length cap.
If faults overlap on the same key, the first configured active fault wins.

The returned DataFrame has one row per input step and these columns:

- `step`
- `<key>_clean` — the original value
- `<key>_observed` — the value presented to an agent (or blank for dropout)
- `<key>_fault` — the fault type, or `none` when the sensor is clean

The optional `output_path` writes the same columns to CSV. With `enabled=False`,
`FaultInjector.apply()` is an exact observation pass-through and emits no labels.

## Offline Scenarios

The [scenario library](scenarios/README.md) provides reproducible twins using
saved Vellore archive weather. No credentials or network access are needed.

| Scenario | Historical weather date | Purpose |
| --- | --- | --- |
| `clear_day` | 2024-01-25 | Sunny-midday baseline |
| `monsoon_cloudy` | 2023-12-04 | Low, variable solar |
| `heatwave` | 2024-05-01 | Hot-day panel derating |
| `low_battery_start` | 2024-01-25 | Initial SOC at 20% reserve |
| `degraded_battery` | 2024-01-25 | Reduced capacity: 6 kWh |
| `overnight_ev_cloudy` | 2023-12-04 | Overnight EV on a cloudy day |
| `three_days` | 2024-01-25 repeated | 864 five-minute steps |
| `demo_sunny_home` | 2024-01-25 | Dashboard default: 60 m² array so the battery visibly charges and discharges |

```python
from scenarios import CATALOG, build_twin, list_scenarios

twin = build_twin("three_days")
rows = twin.run_full_day(steps=CATALOG["three_days"]["steps"])
```

The weather comes from [Open-Meteo historical reanalysis](https://open-meteo.com/en/docs/historical-weather-api),
not station measurements or synthetic curves. Source URLs and attribution
are recorded with the saved files. `three_days` repeats one weather day.
Explicit EV charging segments (`EVModelConfig(segments=...)`, #67) can model
cases such as a late EV arrival; no catalog scenario uses them yet.

## Digital Twin

In addition to the static synthetic dataset (`data/generate_dataset.py`), AgentGrid now has a live
digital twin module (`digital_twin/`) — a tick-by-tick simulator modelling the same system (solar,
battery, EV, appliances). The static generator pre-computes one 24h day of random curves up front;
the twin instead advances simulated time one `tick()` at a time, driving solar from weather data
through physics equations and evolving battery/EV/appliance state live. Both produce the exact same
timeseries row schema, so existing agents consume either source via `update_state(row)` with zero
changes.

Main components (`digital_twin/`):

- `weather_client.py` — Open-Meteo client for Vellore, India (lat 12.97, lon 79.16); hourly
  `shortwave_radiation` / `cloudcover` / `temperature_2m`, timezone `Asia/Kolkata`, no API key.
  Supports historical dates (repeatable tests) plus in-memory/file caching; network layer is
  mockable so unit tests never need internet.
- `solar_model.py` — simplified PV physics: `power = irradiance × area × efficiency / 1000` with
  −0.4%/°C derating above 25°C (crystalline-silicon STC assumption). Defaults: 20 m², 20%.
- `battery_model.py` — stateful SOC with ~90% round-trip efficiency (0.949 each way), SOC clamped
  to [0, 100], plus a deliberately simple first-order thermal model. NOT the Phase 5 PINN; health
  stays constant by default. Opt-in synthetic degradation supports the #69
  dataset and empirical baseline; it does not represent calibrated cell aging.
- `ev_model.py` / `appliance_model.py` — scenario/state-based behaviour (NOT detailed electrical
  physics), reusing the generator's scenario patterns (overnight/midday/evening-topup) as live
  stateful models. Deterministic under a fixed seed.
- `twin.py` — `DigitalTwin.tick(time_delta_min=5)` orchestrates all models and returns one row
  matching the dataset schema; 288 ticks = 24 simulated hours, no wall-clock sleeping.

```python
from datetime import datetime
from digital_twin import DigitalTwin, DigitalTwinConfig

twin = DigitalTwin(DigitalTwinConfig(start_datetime=datetime(2024, 1, 15, 0, 0)))
row = twin.tick(5)  # same keys as generate_dataset()["timeseries"] rows
```

Notes:

- `simulation/run_simulation.py` provides `run_twin_simulation(twin, allocator)`, which uses the
  twin's own appliance roster and EV charging limit. It advances 288 configured timesteps
  (24 hours at the default five-minute step), then allocates solar over the resulting rows.
  Those replay allocations do not feed back into battery, EV, or appliance state.
  Use `run_closed_loop` or `run_graph_closed_loop` for allocation-driven feedback.
- Solar/battery models are simplified physics approximations, honestly documented in each
  module's docstring — not high-fidelity models.
- Tests: `pytest digital_twin/ -m "not live"` (offline); the single `@pytest.mark.live` smoke test
  hits the real Open-Meteo API.

## Current Progress

### Battery Degradation Data

Opt-in `BatteryModelConfig(degradation=True)` adds illustrative cycle/calendar
aging without changing defaults. Generate deterministic multi-day closed-loop
data and a held-out empirical baseline using
`python -m battery_health.baseline --days 3 --seed 42 --output battery-output`.
See [assumptions, schema and evaluation limits](battery_health/README.md).
This is synthetic preparation data, not measured SoH, a PINN or a real-cell
remaining-life prediction. Health does not yet alter capacity or bidding.

### Optional Grid Connection

Grid operation is off by default. Pass `grid_config=GridConfig(...)` to
`run_closed_loop` (or the graph runner) to import for unmet loads and export
otherwise curtailed solar. Battery charging remains solar-only. Imports are
allocated against remaining demand by the same supplied allocator, preserving
its priority policy. Existing records and metrics are unchanged when disabled.

```python
from simulation.grid import GridConfig

# SYNTHETIC test values in currency-free units; not a real residential tariff.
grid = GridConfig(enabled=True, import_limit_kw=3, export_limit_kw=2, tariff=[
    {"start_min": 0, "end_min": 1440,
     "import_price_per_kwh": 2, "export_price_per_kwh": 0.5},
])
```

Tariffs cover the day exactly once, use exclusive ends, and may wrap midnight.
An interval is priced at its start band. Enabled run metrics additionally report
import/export kWh, net cost and solar self-consumption (zero for no solar).
Self-consumption includes solar stored in the battery and excludes exported or
curtailed solar; it is not a household self-sufficiency percentage. An actual
demo tariff and export rules still require an owner-supplied authoritative source.

- `agents/`: Solar, Battery, EV and Appliance agents turn their state into a
  validated bid (`demand_kw`, `urgency`); the battery also offers discharge supply.
- `simulation/`: the hybrid arbiter (priority 1, then 2, then urgency bands), the
  proportional baseline, battery mode selection, the closed-loop runner with
  sensor/device fault policies, outcome metrics and the paired comparison
  (`simulation/comparison_results.md`).
- `orchestration/`: the same closed-loop step as a LangGraph graph.
- `digital_twin/`, `scenarios/`, `data/`: the twin, offline scenarios and the
  static dataset generator.
- `explain/`: facts-first explanations derived from the arbiter's rules.
- `dashboard/`: the Streamlit demo UI; `docs/demo-video.md`: the recording script.

## Dashboard (demo UI)

An interactive Streamlit dashboard runs the closed-loop digital twin live on the
saved offline scenarios and shows the hybrid arbiter next to the proportional
baseline. Use a virtual environment, because Streamlit pins its own versions of
shared packages such as `starlette` and `protobuf`:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on macOS/Linux)
python -m pip install -r requirements-dashboard.txt
python -m streamlit run dashboard/app.py
```

Then open http://localhost:8501. The sidebar picks the scenario, the allocation
strategy and an optional fault to inject; the tabs show:

- **Overview**: power flows, battery/EV state of charge, where the power went, unmet demand
- **Agents & decisions**: every agent's bid (priority, urgency, requested vs received) at
  the selected moment, with a plain-English explanation generated from the arbiter's rules
- **Hybrid vs baseline**: all outcome metrics for both strategies on the same scenario
- **Faults**: the affected steps and any sensor substitutions the runner made

The data layer (`dashboard/data.py`) has no Streamlit dependency and is covered by
`dashboard/test_data.py` in the normal offline test suite.

## Visualization

Run `python viz/plot_comparison.py --scenario clear_day --output comparison.png`
for a paired **closed-loop** comparison. All saved scenario names are supported,
including `overnight_ev_cloudy` and the 864-step `three_days`. Panels show solar
and actual load delivery, battery SOC and modes, EV SOC/targets/departures with
full-interval plug-in shading, unmet energy, and outcome metrics. The two fresh
twins must have identical initial conditions and exogenous schedules. SOC is
shown at interval ends; unmet energy uses each record's actual timestep.
`plot_comparison(fresh_twin_factory, save_path=...)` also accepts a factory
for a 288-step experiment. Tests: `python -m pytest viz/ -q`.

Run `python viz/plot_day.py --seed 42` to save `day_plot.png` with solar vs allocated power, allocations by agent group, and dataset battery/EV SOC with EV plug-in shading. SOC comes from the static dataset, not allocation feedback. Run the headless tests with `python -m pytest viz/test_plot_day.py`.

## Demo Recording

The [demo script and readiness checklist](docs/demo-video.md) covers a 5-6 minute
offline walkthrough, dependency gates, fresh-environment checks and the manual
record/upload steps. It does not claim a video has been recorded or uploaded.
