# AgentGrid — Decisions Log

## 2026-09-05 — Allocator surplus handling (issue #5)
Decision: surplus solar (available > total demand) is curtailed/discarded,
not tracked or exposed anywhere in allocate()'s output.
Why: no design yet for where surplus should go (battery priority charging,
export, curtailment) — deferred until that's needed.
Revisit: before/during Phase 2, since bidding/arbiter may need this signal.

## PR #7 — Battery urgency formula: squared, not linear
Decision: urgency = ((100 - SoC)/100)^2 instead of linear (1 - SoC/100).
Why: linear made 50% SoC feel too urgent (0.5 urgency); squared keeps
urgency low until near-empty.

## PR #7 — EV urgency: zero when at/above target SOC
Decision: EVAgent.compute_urgency() short-circuits to 0.0 when SOC gap is
already 0, instead of the original logic which could return nonzero urgency
even at 100% SOC.
Why: caught in Copilot review on PR #7 — urgency should never ask for
charging when there's no SOC deficit.

## PR #7 — Appliance deferable window: 4-hour ramp assumption
Decision: urgency ramps toward 1.0 as time-left-in-window shrinks, assuming
a standard 4-hour (240 min) maximum operating window for deferable
appliances.
Why: dataset doesn't specify a per-appliance ramp window, so a fixed
assumption was used as a starting point.
Revisit: if per-appliance window length becomes available in the dataset.

## PR #7 (follow-up) — Missed appliance window: silent vs logged
Decision: no logging framework built. Added a minimal in-memory
self.missed_windows list on ApplianceAgent (timestamp + appliance_id) for
debug/test visibility only — no printing, no file writes.
Why: real user-facing explanation of missed windows deferred to Phase 4
(LLM-generated allocation explanations); this is just enough to inspect
during testing now.

## Issue #10 — missed_windows is a current-timestep event buffer
Decision: `ApplianceAgent.missed_windows` is per-timestep/current-step
state, not cumulative run history.
Lifecycle:
- `update_state(row)` begins the new timestep and clears the previous
  timestep's events whenever the parsed timestep changes (so a repeated
  `update_state()` for the same timestep does not drop an unread event).
- `compute_urgency()` may append a missed-window event only when the
  current timestep is outside the appliance's usual window, and never
  duplicates the same appliance/timestep twice (guarded by a
  timestamp + appliance_id check, so repeated calls are safe).
- The current-step consumer/arbiter reads the event (via
  `missed_windows` or the `get_missed_windows()` copy accessor) before the
  next timestep begins; the next `update_state()` then clears it.
- Buffer holds at most one event per appliance per timestep, so a 288-step
  run stays bounded (no unbounded growth).
Why: bid packaging in Phase 2 will call `compute_urgency()` every timestep
and the arbiter will consume the signal — without dedup + clearing, the
same event would be recorded multiple times and grow without bound.
Scope note: `run_simulation` currently does not consume this signal because
it does not yet call `compute_urgency()` in its allocation loop; this is
intentional. The contract above is designed so Phase 2 can safely consume
it once urgency/bid packaging is integrated. Urgency formula unchanged.

## 2026-10-06 — Add a digital twin as a second data source, alongside the static dataset
Decision: build digital_twin/ as an ADDITIONAL live simulator (real physics + real weather data for solar), not a replacement for the existing static dataset generator. Both remain available; agents can consume either since both produce the same row schema.
Why: the static dataset is fast and good for automated testing/CI; the digital twin gives a more realistic, defensible "real-time" story for demos and faculty/patent discussions, without risking the already-tested Phase 1/3 pipeline. Keeping both avoids a disruptive rewrite.

## Phase 2 prep — Agent bid object (get_bid())
Decision: the agent-side bid is exactly
`{"agent_id", "demand_kw", "urgency"}`, packaged generically in
`Agent.get_bid()` from the existing `get_demand_kw()` /
`compute_urgency()` with no overrides and no timestamp/context field.
Why: the bid is per-agent/per-timestep state; timestep ownership stays
with the caller (which already holds the row), so stamping time inside
the bid would duplicate caller state and invite skew.

## 2026-10-06 — Digital twin model assumptions (REAL DATA vs SIMPLIFIED PHYSICS vs SCENARIO)
Decision: digital_twin/ splits the world into three honesty tiers. REAL EXTERNAL DATA: only
solar irradiance/cloudcover/temperature from Open-Meteo (Vellore 12.97, 79.16, Asia/Kolkata;
variables shortwave_radiation/cloudcover/temperature_2m; historical-date + cacheable so tests are
repeatable and offline). SIMPLIFIED PHYSICS: solar PV (20 m2, 20% efficiency, -0.4%/C above 25C
per c-Si STC convention) and battery (12 kWh midpoint of generator's 10-14 range, 90% round-trip
as 0.949 each way, first-order thermal decay — explicitly NOT the Phase 5 PINN, health constant).
SCENARIO-BASED: EV (overnight/midday/evening-topup segments mirroring generate_dataset.py as live
state) and appliances (roster + usual-window stochastic cycles, no electrical physics).
Why: keeps the twin defensible in demos/reviews without overstating fidelity; schema-identical rows
(tested literally against generate_dataset output) let agents consume either source unchanged, and
 run_simulation.py stays unwired until a separate integration task.

## Overnight load spike — ev_charger excluded from household_load_kw
Decision: the appliance roster still simulates the 5 kW ev_charger (state
visible in appliance_loads/active_appliances) but its power is excluded
from household_load_kw totals (EXCLUDED_FROM_HOUSEHOLD_TOTAL), matching
generate_dataset.py whose household load has no EV term.
Why: the charger fired overnight inside its 19:00-07:00 window and stacked
~5-7 kW onto household load while EVModel was separately charging the EV
(PR #4 double-count) — draining the home battery to 0% before sunrise.
After the fix the worst plausible overlap (washing_machine + pool_pump +
base/essentials, or water_heater + base) peaks at ~3 kW; a regression
test caps the 288-tick max below 4 kW.

## Digital twin repeatability — fixed default seed 42
Decision: EVModelConfig.seed and ApplianceModelConfig.seed default to 42
instead of None, so a bare DigitalTwinConfig is deterministic; a test
asserts two default-config twins produce identical 288-tick output.
Why: seed=None (random.Random(None)) made every default twin a different
day, so overnight spikes and test results were irreproducible.
Revisit: if a "random day" mode is ever wanted, require an explicit
seed=None opt-in rather than defaulting to it.

## 2026-10-14 — Digital twin vs generator solar scale divergence (issue #29)
Decision: Document the ~50% peak solar difference between the digital twin
(default 20 m², 20% efficiency, 4 kWp nameplate) and the dataset generator
(synthetic curve amplitude peak_kw ~4.5–6.0 kW, before cloud dips and noise). Do not recalibrate.

Measured ranges across seeds 1, 7, 42:
- Generator (data/generate_dataset.py): peak 5.35–5.73 kW, mean daylight 2.3–3.0 kW, CV 0.58–0.73
- Digital twin (default config, mocked/clear-sky weather peaking at 800 W/m²): peak 2.78 kW, mean daylight 1.93 kW, CV 0.41

Here daylight means samples with solar output > 0.1 kW, excluding the
generator's small positive nighttime sensor noise. CV is population standard
deviation divided by the mean over those samples; it includes the daily
curve's shape, not just cloud variability. These are mocked measurements,
not live-weather measurements. Reproduce from the repository root:

```python
import numpy as np
from data.generate_dataset import generate_dataset
from digital_twin.test_digital_twin import make_test_twin
from digital_twin.weather_client import _cache

for seed in (1, 7, 42):
    _cache.clear()
    sources = {
        "generator": generate_dataset(seed=seed)["timeseries"]["solar_gen_kw"].to_numpy(),
        "mocked twin": np.array([r["solar_gen_kw"] for r in make_test_twin(seed=seed).run_full_day()]),
    }
    for name, values in sources.items():
        daylight = values[values > 0.1]
        print(seed, name, values.max(), daylight.mean(), daylight.std() / daylight.mean())
```

Why the difference:
- Twin models a specific physical system: 20 m² × 20% = 4.0 kWp at STC (1000 W/m², 25°C).
  Under test/clear-sky conditions (800 W/m² peak irradiance, ~32°C ambient → ~57°C cell),
  temperature derating (-0.4%/°C) reduces output to ~2.78 kW actual peak. This is physically
  consistent with the model under the mocked conditions; it does not establish
  observed Vellore January output.
- Generator uses a synthetic bell curve with peak_kw ~ Uniform(4.5, 6.0) as its
  amplitude, not a nameplate rating. Cloud dips and sensor noise alter the actual
  peak. The generator does not specify panel area or model physical system sizing.
- Cloud-dip volatility: Generator explicitly adds multiplicative cloud dips (0.3–0.7× for
  10–30 min), giving CV ≈ 0.6–0.7. Twin relies on weather-source irradiance which includes
  natural cloud variability; the mocked/clear-sky test profile is a smooth bell curve (CV ≈ 0.4).
  With live Open-Meteo data, the twin would exhibit real cloud volatility.

Why not recalibrate:
- The twin's 20 m² / 20% / -0.4%/°C parameters are documented, defensible assumptions for a
  specific residential system (see digital_twin/solar_model.py:20-27 and decisions.md:75-83).
  Changing them to match the generator's synthetic peak would undermine the twin's "physically
  grounded" design principle and its value for demos/faculty review.
- The two sources serve different purposes: generator = fast, statistical, CI-friendly; twin =
  realistic, physics-based, defensible. Schema identity (same row keys/types) is the contract
  for agent compatibility, not value-range identity.
- Users needing a larger system can configure SolarModelConfig(panel_area_m2=30.0) or similar
  without changing defaults.

Revisit: if agent logic is found to implicitly depend on a specific solar peak range.
