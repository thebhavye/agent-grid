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

## 2026-10-15 — Phase 2 arbiter and closed-loop evaluation plan
Decision: Phase 2 will combine hybrid allocation, a battery that can supply
or consume power, and a closed-loop twin runner evaluated against the
proportional baseline. This is planned work, not current behavior.

- Hybrid allocation: satisfy critical/non-deferable loads first, then rank
  flexible demand using urgency bids and share power proportionally within
  equal-priority groups. Cap every allocation at requested power and
  enforce a timestep power balance; record unmet demand and curtailment.
- Battery as supply and demand: expose feasible discharge supply as well
  as charging demand, constrained by SOC/reserve, capacity, charge/discharge
  limits, and efficiency from the twin configuration. Select one mode per
  timestep so the battery cannot charge and discharge simultaneously.
  The existing demand-only get_bid() contract remains unchanged until an
  explicit supply/mode contract is introduced during implementation.
- Closed-loop runner: observe current state, collect bids, allocate, apply
  the delivered power for that interval, then advance the twin. Battery
  and EV SOC and appliance progress/completion must reflect actual served
  energy rather than independent precomputed trajectories. Avoid applying
  the twin's existing autonomous charging updates a second time.
- Fair baseline comparison: run the hybrid arbiter and proportional
  allocator separately from identical initial state, weather, seeds,
  appliance/EV scenarios, timestep, and physical constraints. Both runs
  must use the same closed-loop physics and battery feasibility rules;
  keep the current allocate() function as the proportional reference,
  with any battery-mode adapter shared and documented. Do not compare
  closed-loop outcomes against the existing open-loop replay as though
  its SOC were a result of baseline allocation.
- Score outcomes using critical unmet energy (kWh), EV SOC/target shortfall
  at departure, appliance completed/missed windows, curtailed solar (kWh),
  and battery SOC/reserve violations and throughput (kWh). Report paired
  results across fixed scenarios/seeds, including overnight EV windows;
  verify energy conservation and charge/discharge exclusivity before
  claiming an improvement.

Why: replaying the same precomputed SOC for every allocator measures only
power splits. Feedback is necessary to establish whether bidding improves
charging, service completion, and use of solar under the same constraints.
The existing run_simulation()/run_twin_simulation() replay paths remain
available for deterministic demand/allocation checks.

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

## Issue #25 — Twin roster source for run_simulation
Decision: use `run_twin_simulation(twin, allocator)` for twin-backed runs.
It reads the actual `twin._appliances.config` and `twin._ev.config` together,
converts the appliance specs to the generator's table columns, advances
the twin by 288 configured timesteps, and delegates to `run_simulation`.
This is 24 hours at the default five-minute step. Allocation does not feed
back into twin state.
Why: the twin's own roster preserves custom appliances and ratings, and
obtaining both configurations together prevents accidentally pairing it
with the wrong EV limit. The wrapper isolates access to the twin's model
configuration; callers do not need to extract private attributes.

```python
from simulation.baseline_allocator import allocate
from simulation.run_simulation import run_twin_simulation

results = run_twin_simulation(twin, allocate)
```

## Issue #25 — Authoritative twin EV rate and roster validation
Decision: the twin entry point takes EVAgent's rate from the same twin's
EVModelConfig.max_charge_kw (7.0 kW default, including custom values such
as 11.0 kW), not the roster's ev_charger appliance_power_kw (5.0 kW default).
Why: EVModel.tick() clamps simulated charging power with
EVModelConfig.max_charge_kw (ev_model.py), while the appliance rating is
EXCLUDED from household_load_kw (EXCLUDED_FROM_HOUSEHOLD_TOTAL) and never
drives twin physics — the two values are not interchangeable. EVAgent
demand scales linearly with its max, so matching the physics clamp keeps
agent demand consistent with the twin.

The lower-level `run_simulation` keeps the generator's roster-derived EV
rate when no override is supplied. Manually supplied twin rows require
both their matching roster and explicit EV rate; prefer the twin entry
point for new calls. Only `appliances_df=None` uses the legacy fallback.
Supplied records require unique nonempty appliance IDs and finite,
nonnegative power ratings. Invalid records raise errors instead of
creating unknown appliances or substituting default power. An empty
roster or one without ev_charger is allowed with an explicit positive,
finite EV rate: EVModel is configured independently of the household
roster. Without that override, a valid ev_charger row is required.

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

## Issue #26 — Fail-fast bid/input validation
Decision: Battery/EV agents raise ValueError on non-finite SOC/departure
(ApplianceAgent keeps its 1.0 kW graceful default for missing/NaN rated
power but raises on Inf), and baseline allocate() rejects NaN/Inf/negative
available power or demands instead of propagating NaN allocations.
Why: one bad sensor value silently corrupted the whole timestep (e.g. NaN
battery SOC -> demand 0.0 + urgency 1.0); Phase 2 bidding needs trustworthy
bids, so invalid input fails loudly at the source.

## Phase 2 — Hybrid arbiter
Decision: `hybrid_allocate(available_power_kw, bids, priorities)` serves
priority 1 fully before priority 2, sharing proportionally to demand within
each priority when solar is insufficient. Missing priorities default to 5. Flexible bids are sorted
by descending urgency, then lower priority number, then agent ID.
Urgencies within 1e-9 of the highest urgency in a band are treated as tied;
the fixed highest-value anchor prevents transitive chains of near-ties.
Within each band, equal-priority agents share proportionally to demand,
and lower priority numbers are served first. Agent ID orders arithmetic
deterministically, including when the input bids are reordered.
Every allocation is capped at demand and excess solar is curtailed.
Power validation and proportional sharing reuse the baseline helpers;
invalid urgency, duplicate IDs, and total-demand overflow fail fast.
This pure arbiter does not advance a twin or feed allocations into state;
closed-loop integration and outcome metrics remain separate work.

## Phase 2 — Battery supply and mode selection
Decision: the battery can bid for charging or report feasible discharge
supply, bounded by configured power limits, efficiency, and timestep
energy. The default 20% SOC reserve protects stored backup energy;
discharge stops at that floor. BatteryLimits defaults match the twin's
12 kWh, 0.5C/1C, and 0.949 efficiency configuration; C-rates are converted
to kW when importing a twin config. Existing demand/bid behavior is retained.
The shared pure choose_battery_mode helper caps charging demand by solar
surplus as well as headroom, leaving other loads' power available. At exact
solar/load balance it idles. It selects charge on positive solar surplus,
discharge on a load deficit, or idle when no feasible demand/supply remains.
Charge and discharge are mutually exclusive. The selector returns feasible
power bids; it does not apply power, run a closed-loop simulation, or mutate
twin state. Both baseline and hybrid callers can use the same policy.

## Phase 2 — Twin observe/commit step (issue #54)
Decision: `DigitalTwin.observe()` advances the EV and appliance exogenous
schedules once and records the interval without applying battery, EV charging,
or appliance runtime progress. `commit(delivered)` then applies either the
autonomous behavior (`delivered=None`) or caller-provided power and advances
the twin clock exactly once. Calling either method out of order is an error.

`DigitalTwinConfig(closed_loop=False)` preserves the legacy tick path and its
captured 288-row outputs. Closed-loop mode uses stable per-appliance RNG
streams and a fixed pair of schedule draws per appliance per interval, so
allocation changes cannot shift later random draws. A successful start draw
while an earlier cycle is still running is recorded as a blocked start.

Delivered battery power is signed and passes through the single battery model
update for the interval. EV delivery is non-negative, capped by the configured
charger rate and SOC headroom, and is rejected while unplugged. Flexible
appliance runtime decreases in proportion to served/desired power; essential
loads never complete. Closed-loop rows expose desired and served appliance
power plus the battery power actually applied. The runner and outcome metrics
remain separate follow-up work.

## Phase 2 — Closed-loop runner (issue #56)
Decision: `simulation.closed_loop.run_closed_loop()` is the shared execution
loop for proportional and hybrid allocator comparisons. It observes a fresh
closed-loop twin, updates the existing agents, derives battery charge/supply
feasibility from the twin configuration, selects charge/discharge/idle mode,
allocates power, and commits the signed battery plus load deliveries exactly
once. `proportional_adapter()` adapts the unchanged baseline allocator to the
same bid/priorities interface used by `hybrid_allocate()`.

Appliance bids use the twin observation's `appliance_desired_kw` values when
present; `ApplianceAgent` retains its rated-power behavior for legacy rows
without that optional key. Priority maps come from the twin roster, with the
EV charger priority applied to `ev_1` and the battery assigned priority 5.

Every record includes the allocator inputs and outputs, selected mode,
available and curtailed power, signed battery power, desired/served appliance
power, and post-commit SOC state. The runner rejects stale or non-closed-loop
twins and checks allocation caps, charge/discharge exclusivity, power balance,
and zero curtailment in discharge mode. Metrics and paired outcome scoring
remain separate follow-up work.

The fixed household base load participates as `household_base_load`, an
always-on priority-1 bid, so it is included in allocation and battery balance.
Physical limits and energy checks use unrounded SOC and solar power. EV
service is recorded separately as `ev_applied_kw`, capped by physical charging
headroom, and excluded from appliance totals to avoid counting it twice.
Post-commit balance and SOC checks verify actual delivery, not just requests.

## Issue #64 — Sensor-only fault injection and labelled export
Decision: implement faults as a deterministic observation wrapper around clean
twin or generator rows. The wrapper supports dropout, NaN, stuck, spike, and
drift faults, with fixed half-open step windows or seeded probability/duration
events. `generate_fault_dataset()` exports clean, observed, and per-key fault
columns without changing the source rows or simulation physics.
Why: Phase 6 needs repeatable labelled bad observations, while agents must keep
their existing fail-fast validation behavior. Fault response policy and faults
that alter device physics remain out of scope for this layer.

Fault configurations canonicalize aliases and validate integer boundaries,
durations and finite numeric parameters before use. Probability events extend
as chronological observations arrive, without a fixed horizon; expired events
are discarded. The first configured active fault wins on overlap. A stuck
sensor without a prior good sample freezes its first available reading.

## Phase 4 — Facts-first allocation explanations
Decision: ship a deterministic, offline facts builder and template first.
External disclosure of simulated step data has not been approved; the
render_llm entry point always warns and returns the template, without
invoking even an injected client. No LLM SDK or credentials are required.
Enabling an external renderer needs a separate owner privacy decision,
facts-only JSON input, fake-client tests and numeric-claim validation at
six decimal places; that integration and its number checker are deferred.

The builder accepts an explicit keyword-only strategy (hybrid by default,
or baseline) and verifies allocations against that policy before attaching
reasons. Missing allocation keys mean zero. Facts retain full numeric
precision; template kW values display six decimal places. Agent output is
sorted by ID, and no inputs are mutated. Zero demand has served_fraction
1.0. Missed-window events are the agent's existing reported events, not
proof of an incomplete physical appliance cycle. They override the reason.
Baseline shortages use shared_proportionally; baseline priority metadata
and finite urgency values do not undergo hybrid range/type checks.
Partially served flexible singletons reflect earlier-group precedence;
proportional sharing applies to actual peer groups (or a lone group with
no earlier service). Urgency bands use the arbiter's fixed-anchor tolerance.
Hybrid reasons describe
critical precedence, proportional groups and deferred flexible groups;
the fixed deferred_lower_urgency code includes priority tie-breaks, so
its text says earlier urgency/priority groups rather than asserting a
strict urgency difference. The demo uses the current generator/arbiter
contracts and makes no claims about closed-loop outcomes or superiority.

Critical reasons follow the separate priority-1 and priority-2 pools: a
priority-2 singleton limited by earlier priority-1 service reports
critical_first. Actual priority-2 peers still report proportional sharing
within their own pool; no sharing across critical priorities is implied.

## Issue #67 - Explicit EV charging segments

Decision: EVModelConfig accepts an optional non-empty list of EVSegment values,
with plug_in_min, departure_min, start_soc and target_soc. Custom schedules
override scenario selection and skip all random segment draws. EVModel.scenario
reports "custom" without adding a random-selectable EVScenario enum member.
The seed still controls the existing autonomous charging noise. segments=None
keeps scenario selection, RNG draws, charging and the #54 snapshot unchanged.

Intervals are half-open, sorted and non-overlapping, with finite numeric times
0 <= plug_in_min < departure_min <= 1440 and finite SOC values in [0, 100].
Adjacent intervals are allowed, and targets need not exceed starting SOC.
Configuration and model copy supplied intervals so callers cannot mutate an
active schedule through their original list or segment objects. The original
start_min/end_min model accessors remain available as aliases.

Follow the existing overnight scenario's day-bounded convention for a single
23:30-to-06:00 connection:

```python
EVModelConfig(segments=[
    EVSegment(0, 360, 15, 80),
    EVSegment(1410, 1440, 15, 80),
])
```

The midnight tail comes first because intervals are sorted by minute of day.
A wrapping pair such as (1410, 360) is invalid, not silently reordered.
SOC resets to 15% on the actual evening plug-in, not on the continuous midnight
connection. Departure reporting retains the first interval's rounded end modulo
1440 (06:00 here). Observe/commit still uses actual delivered power, charging
efficiency, maximum power and capacity limits; autonomous mode retains its
existing per-interval ramp. reset() preserves custom intervals and replays the
same seeded behavior. The late_ev_arrival scenario catalog integration is a
follow-up once the scenario library is merged, not a dependency of this change.
