# AgentGrid

Multi-agent simulated smart-home microgrid — LangGraph bidding, PINN battery health, SHAP explainability, edge LLM.

AgentGrid is a simulated smart-home microgrid where autonomous agents (solar, battery, EV, appliances) score their own power urgency and a shared allocator arbitrates limited solar power between them.

## Roadmap

| # | Phase | Status |
|---|-------|--------|
| 1 | Urgency-scoring agents | DONE |
| 2 | LangGraph multi-agent bidding + arbiter | not started |
| 3 | 24h synthetic dataset + baseline eval | DONE |
| 4 | LLM-generated allocation explanations | not started |
| 5 | PINN for battery SoH/thermal risk | not started |
| 6 | Edge quantized LLM for fault interpretation | not started |
| 7 | SHAP explainability | not started |

## Running tests

```bash
python -m agents.test_agents
python -m data.test_generate_dataset
python -m pytest test_baseline.py -v
python -m pytest test_simulation.py -v
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
- `simulation/` — Rahul

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

## Current Progress

- `agents/` holds the 4 agent classes (Solar, Battery, EV, Appliance) with
  working `update_state()`, `get_demand_kw()`, and `compute_urgency()`.
- `data/` generates the 24h synthetic dataset; `simulation/` runs the
  capped proportional baseline allocator over it.
- `test_agents.py` exercises the agents; `test_baseline.py` /
  `test_simulation.py` cover the allocator and end-to-end pipeline.