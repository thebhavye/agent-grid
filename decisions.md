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

Review clarification: "single overnight segment" is implemented as one logical
connection represented by two day-bounded intervals, not a wrapping EVSegment.
This reconciles the explicit 0 <= start < end <= 1440 constraint with overnight
charging. This interpretation was accepted at merge review (2026-10-16).

The custom scenario sentinel is deliberately a string rather than an extra
random-selectable enum member. Consumers must not assume .value is available
for custom schedules. EVSegment's former start_min/end_min constructor keyword
names are not supported; positional construction and read-only aliases remain.

## Issue #63 — Saved offline weather and stress scenarios
Decision: provide seven named scenarios through CATALOG, list_scenarios()
and build_twin(name, closed_loop=False), using existing twin configuration
APIs. Each build creates fresh models and an offline WeatherClient session
reading a saved JSON payload, with an instance-local cache independent of
live weather requests. No twin physics, EV model or runner behavior changes.

Saved Vellore profiles come from Open-Meteo's historical archive for
2024-01-25 (sunny midday), 2023-12-04 (overcast monsoon) and 2024-05-01
(hot pre-monsoon). They are historical reanalysis, not station measurements;
hourly values are unchanged, and request URLs, retrieval date, units and
CC BY attribution are retained. No synthetic fallback data was needed.

The low-battery scenario starts at 20% SOC. Reduced capacity (6 versus
12 kWh) is an explicitly assumed degradation proxy, not a health/aging
model. Three-day runs use 864 five-minute steps and repeat one saved weather
day, rather than implying varying multi-day historical weather. The hot-day
test isolates derating using identical irradiance at 25 C ambient as a
counterfactual control. Explicit late-EV-arrival segments remain issue #67.

## Issue #57 - Outcome metrics and paired comparison

Decision: score committed closed-loop records and compare separate twins using
identical initial state, physics, seeds, timestep, and the existing synthetic
weather fixture. Weather caches are isolated; no network is used. Before scoring,
compare solar, EV connection/departure/target series and all drawn appliance cycle
starts, including initialization draws. Accepted starts and later demand can
legitimately differ with allocations; these are outcomes, not fairness inputs.

The runner adds observational metadata only: priorities, reserve, segment target,
real departures, cycle completion/deadline events, blocked-start increments and
start draws. No bids, allocation policies, RNG draws or physics are changed.
A regression test compares all previous record fields with telemetry disabled.

Energy uses each record's dt_min. Unmet load includes base load and EV once,
excluding the battery storage bid and solar. Battery energies are electrical
terminal kWh, positive battery_kw charging and negative discharging. EV shortfall
is summed percentage-point deficit at each real departure using the last
plugged interval's committed SOC and that segment's target; meeting it adds zero.
Continuous overnight connection at midnight and a truncated run are not
additional departures. Cycle misses are counted at window end even if a cycle
later completes; blocked starts are increments, not repeated cumulative totals.
Reserve counting tolerates only 1e-9 percentage points of floating-point noise.
Balance is recomputed from actual appliance/EV service, served base load, solar,
signed battery power and curtailment, with no EV display-load double counting.

Measured with python -m simulation.compare: seeds 1, 7, 42 across overnight,
midday and evening_topup, each 288 five-minute steps, two strategies per pair.
All 18 runs have balance errors below 1e-6 kW (maximum 1.088018564e-14 kW)
and zero battery reserve violations. Full hybrid/baseline/delta values for all
12 metrics are saved in simulation/comparison_results.md and the PR description.
The following deltas are hybrid minus baseline:

| Seed | EV scenario | Critical unmet delta (kWh) | Flexible unmet delta (kWh) | Departure shortfall delta (pp) |
|---|---|---|---|---|
| 1 | overnight | -1.273225817 | 3.984209366 | 0.29066547 |
| 1 | midday | -4.720696896 | 38.60869739 | 3.967829883 |
| 1 | evening_topup | -0.9575598701 | 3.624287337 | 1.191216758 |
| 7 | overnight | -1.330639928 | 3.120719413 | 0.7139311619 |
| 7 | midday | -4.331048275 | 49.51918374 | 2.629671815 |
| 7 | evening_topup | -0.7538255529 | 2.317095053 | 1.01350414 |
| 42 | overnight | -1.352072885 | 3.063224573 | 1.074269437 |
| 42 | midday | -3.430880178 | 10.11872226 | 3.453560144 |
| 42 | evening_topup | -0.7258181343 | 2.850443566 | 1.006303087 |

Hybrid protects critical loads better in all nine pairs but has higher flexible
unmet energy and EV departure shortfall in every pair; there is no overall
superiority claim. Three seeds, one synthetic weather day and a one-day horizon
are limited evidence. The overnight evening segment's next morning departure is
outside the horizon, and allocation-dependent requests need not total equally.

## Issue #68 — Runner fault-handling policy and device faults
Decision: the closed-loop runner owns a configurable
`on_bad_reading` policy with values `"raise"` (default), `"hold_last"`, and
`"exclude"`. Agents stay untouched and keep their own fail-fast validation;
the runner sanitizes observations before calling `update_state`. A bad reading
is a missing key, a NaN/inf value, or a payload that does not match the key's
schema (for example a number where an `HH:MM` timestamp belongs). `raise`
rejects a bad watched reading in the runner rather than relying on the agents,
because agents substitute silent defaults (a missing SOC becomes 50%) that
would let a hidden fault masquerade as a good step.

Why agents keep fail-fast behavior: the runner is the only layer that knows
whether a run wants to survive sensor noise or halt on it. Moving the policy
into the agents would silently weaken the sensor contracts used elsewhere
(tests, dataset generation, standalone agent use). Keeping the decision at the
runner boundary lets `"raise"` remain the unambiguous default for development
and CI, while exploratory or resilient runs opt in explicitly.

Under `hold_last`, a substituted agent reuses its last good value for each
affected key until the fault clears, but only for `max_hold_steps` consecutive
steps (default 3); after that bound the agent is excluded for that step. Under
`exclude`, an affected agent bids zero for that step, so its demand goes
unserved; the clean demand is still recorded (`demand_kw` plus an explicit
`unserved_kw` map) rather than erased by the zero bid. Every substitution is
logged in the step record as `{agent_id, key, policy, observed, used}`, so
exports and tests can audit exactly what was replaced and why. Held values are
bounded by the clean observation so a stale reading cannot request delivery the
real devices reject: EV and appliance bids are capped by the clean demand, and
battery supply/headroom by the clean twin SOC, so the power-balance,
charge/discharge-exclusivity and reserve invariants hold even mid-hold.

Device faults extend the #64 injector in `digital_twin/faults.py` and are
applied through the twin `commit()` path so they change physics rather than
just observations: `inverter_trip` delivers zero solar for its window while
irradiance is unchanged, `charger_fault` caps an EV to zero accepted power
while plugged in, and `battery_derate` scales the battery's charge and
discharge ceiling by `factor`. They are labelled in the step record under
`device_faults`, deterministic by seed, and validated at configuration time.
Both fault families default off; with them disabled the runner emits records
identical to a run that never passed fault configuration.

## Issue #65 - Deterministic LangGraph bidding orchestration

Decision: compile a real StateGraph with one independent bid node per agent,
a join before battery_mode, then arbiter and commit. Nodes exchange typed
observation, bids, mode, available power, allocation, delivery and record state.
Concurrent bids merge by agent ID; original agent order is restored before
floating-point summation so thread completion order cannot change allocations.

The existing runner's phases are extracted into shared private helpers, not
duplicated or retuned. Both runners retain identical limits, fault policies,
invariants, RNG behavior and record metadata. LangGraph 1.2.15 is pinned, imports
are restricted to orchestration, and no LLM or remote checkpointer is used.
Remote tracing is explicitly disabled even when enabled in the environment.
Tests compare exact full-day records for both allocators, seeds 1/7/42 and all
three built-in EV scenarios, plus graph topology, roster changes, reset and faults.
This wraps the measured #57 trade-offs; it does not claim improved outcomes.

## Issue #69 - Opt-in synthetic degradation and daily empirical baseline

Use cycle aging weighted by actual terminal throughput, C-rate and cell
temperature, plus calendar aging weighted by elapsed days, mean SOC squared
and temperature. Every coefficient is a named BatteryModelConfig field with
units and an explicit assumption label in its docstring. Defaults: cycle fade
0.0002 health percentage points/kWh, calendar fade 0.001 percentage points/day,
C-rate stress 1, temperature reference 25 C, e-fold temperature scale 10 C.
These are illustrative assumptions, not chemistry-calibrated literature values.
Health is monotonic and floored at zero; this task does not shrink capacity or
change bidding. Disabled degradation retains exact legacy fixture/record output.

The seeded dataset uses real committed delivery for three saved scenarios over
multiple days, with per-step terminal energy, SOC, temperature, ambient and
cumulative aging exposures, plus separate daily summaries. Saved weather/EV
schedules repeat each day; appliance streams advance. Targets are daily health
percentage and peak cell temperature. Remaining useful life is not inferred.

Measured command: python -m battery_health.baseline --days 3 --seed 42.
Train on demo_sunny_home and degraded_battery (6 days); hold out all heatwave
data (3 days). Health MAE: 1.2789769243681803e-13 percentage points. Peak
temperature MAE: 0.41239561050337414 C. These are conditional daily fits using
the day's operation, not forecasts of unknown future behaviour. Health fits
the generating exposure terms, so near-zero error is expected by construction,
not evidence of real-cell generalization. A follow-up PINN must use comparable
inputs and this held-out split, report an ablation and avoid a safety claim.

## Issue #61 - Optional grid import and export

GridConfig is off by default. Enabled settlement caps import/export and uses
validated, full-day tariff bands with exclusive ends and midnight wrapping.
Prices are non-negative currency-free per-kWh values; examples are explicitly
synthetic until an owner supplies authoritative residential tariff/export rules.
Each interval uses its starting band's price, including non-five-minute steps.

After solar/battery allocation, run the same allocator on remaining load demand
with capped grid import. Do not import into the battery or export battery energy.
Export only otherwise curtailed solar. The runner checks solar + discharge +
import = actual load delivery + charge + export + curtailment, and preserves
the existing SOC checks. Ignore sub-1e-9 numerical residuals only at enabled
settlement boundaries, not by changing the arbiter or baseline.

Only enabled records gain four settlement fields. Only those runs gain grid
metrics: import/export energy, net cost and on-site solar use (including storage,
excluding exports/curtailment). Disabled records and the original metrics remain
exactly unchanged. Shared runner phases preserve graph parity. This implements
the exported-surplus revisit from issue #5 without changing off-grid defaults.

## Issue #66 - Paired closed-loop comparison plots

Use two fresh, equivalent offline twins with unchanged hybrid/proportional
allocators. Verify initial conditions, exogenous schedules and power balance
before rendering. Plot actual load delivery (including base load and EV once),
post-commit SOC at interval ends, targets/departures, per-interval unmet kWh,
battery modes and score_run outcomes. Interval shading uses stairs and actual
dt_min edges, including the final interval. Catalog scenarios retain their
configured run length; factory inputs run 288 steps. Agg and finally-close
keep the CLI headless and release figures even after a save error. No physics,
policy, data or metrics are changed to improve the visual comparison.


## Issues #59, #83, #84, #85 and #86 — Optional research modules

### #59: facts-first rendering and privacy

The existing facts/template implementation stays the default. An injected
renderer is callable only with `enabled=True`; an API key alone does not enable
external disclosure. No external API was called for this change. Only computed
facts are supplied to the client. A shared numeric-token guard compares claims
at six-decimal precision and falls back on unsupported numbers, empty/non-text
output or exceptions. Numeric consistency does not establish semantic truth;
spelled-out numbers and misuse of an otherwise valid number are limitations.
Optional SHAP factors are labelled surrogate contributions, not arbiter causes.
This extends the earlier #59 decision without granting blanket external API
approval or changing the deterministic reason codes.

### #83: surrogate fidelity

Pre-decision features are captured only when explicitly requested by the runner;
the dashboard requests them. Default runner records remain unchanged. Legacy
record reconstruction is approximate because stored SOC is post-commit. The
histogram gradient boosting surrogate uses a non-decreasing urgency constraint;
SHAP additivity and selected flexible-agent monotonicity cases are tested.
Training: clear_day and monsoon_cloudy, seeds 1/7/42. Holdout: heatwave, seed 13.
Observed held-out R² **0.982160**, served-fraction MAE **0.021095** (10549 training rows; 1772 holdout rows).
This measures imitation of the hybrid arbiter, not causal influence or
outcome superiority. The optional dashboard panel shows top-three factors,
model fidelity and a chosen agent/step. No baseline/fault explanation claim is
made. Full report: `docs/evaluations/shap-surrogate.json`.

### #86: selected-feature urgency versus existing formulas

A separate multi-output random-forest surrogate ranks candidate inputs against
critical unmet, flexible unmet and EV departure shortfall (training-standard-
deviation scaling). Permutation importance is held-out MSE increase. Training
uses clear_day, monsoon_cloudy and demo_sunny_home, seeds 1/7/42. Validation uses
heatwave and demo_sunny_home, seeds 13/29. The battery's validation therefore
holds out seeds, not a scenario: heatwave has no charging rows. The one-hour
solar forecast is persistence, not future saved weather. Recent unmet is an
exponentially decayed one-hour history updated after each decision. Candidate
health is constant here; zero importance does not establish irrelevance.

Selected positively ranked inputs (at most three per type):
- appliance: recent_unmet_kwh, hour_sin, solar_forecast_kw
- battery: hour_sin, soc_gap, recent_unmet_kwh
- ev: recent_unmet_kwh, solar_forecast_kw

The candidate gives equal weight to these normalized inputs; it is a deliberately
simple mapping, not an optimized learned controller. The comparison uses fresh
paired twins and the fairness/scoring checks in `simulation/compare.py`, with
seeds 5/19/31 separate from ranking. Delta is selected minus current:

| Scenario | Seed | Critical unmet Δ kWh | Flexible unmet Δ kWh | EV shortfall Δ pp |
|---|---|---|---|---|
| clear_day | 5 | 0.000000 | 8.437841 | -3.441505 |
| clear_day | 19 | 0.000000 | 10.693338 | 0.989938 |
| clear_day | 31 | 0.000000 | -0.297322 | -2.467343 |
| monsoon_cloudy | 5 | 0.000000 | 0.000000 | 0.000000 |
| monsoon_cloudy | 19 | 0.000000 | 0.000000 | 0.000000 |
| monsoon_cloudy | 31 | 0.000000 | 0.000000 | 0.000000 |
| heatwave | 5 | 0.000000 | -0.068702 | -2.744760 |
| heatwave | 19 | 0.000000 | 10.462952 | 0.442174 |
| heatwave | 31 | 0.000000 | 8.734503 | -2.658333 |

Decision: **keep existing urgency formulas as the default**. Results are mixed,
with sizeable flexible-unmet regressions and some EV regressions. This does not
justify promotion. The run count is small, step rows are correlated, and
permutation spread is not a run-level confidence interval. All rankings and
paired metrics are in `docs/evaluations/feature-selection.json`.

### #85: battery dataset, empirical baseline and PINN ablation

Minimum #69 prerequisites are included: opt-in diagnostic degradation, multi-day
closed-loop data, daily summaries, and an empirical least-squares baseline.
Health never scales dispatch or nominal capacity in this experiment. The thermal
recurrence is unchanged. Coefficients are **assumptions**, not calibrated cell
chemistry: cycle fade 0.02 percentage points/EFC (absolute terminal throughput /
twice nominal capacity), calendar fade 0.0001 percentage points/hour, linear
C-rate coefficient 1, linear mean-SOC coefficient 1, reference temperature 25 °C,
and doubling per 10 °C. All are named `BatteryModelConfig` fields. Higher C-rate,
temperature and SOC accelerate fade in paired tests. Default degradation off
matches original records/scores, checked against the pre-change main commit.
These coefficients have no measured source and must not be used for lifetime
claims. No remaining-useful-life target is selected.

The small CPU network predicts one-step health loss and temperature change from
operation features; its loss includes empirical aging/thermal residuals plus a
monotonic-health penalty. Training uses clear_day/monsoon_cloudy, three days,
seed 42. Holdout uses heatwave, three days, seed 13. Same-network, same-seed
ablation has zero physics weight. These noise-free labels largely duplicate
the physics targets, limiting what the ablation establishes. Baseline/physics
inputs are derived from pre-step state and applied power, never next-state labels.

| Model | SoH MAE (pp, step) | Temperature MAE (°C, step) | Day-end SoH MAE (pp) | Daily peak temperature MAE (°C) |
|---|---|---|---|---|
| Empirical baseline | 5.9211895e-16 | 8.635068e-17 | 1.7053026e-13 | 0 |
| PINN | 6.8724535e-06 | 0.1057941 | 0.0019448583 | 0.015466318 |
| Without physics | 6.8676088e-06 | 0.11407207 | 0.0019714142 | 0.030218047 |

The PINN **does not beat the empirical baseline**. Baseline residuals are near
floating-point precision because it fits the generating equations. Training
both networks plus dataset generation took 2.60 seconds on this cloud CPU;
weights reproduce via `python -m battery_health.train`, so no weights are
committed. Daily health errors accumulate one-step losses from observed state;
temperatures are also conditioned on observed state, not autonomous forecasts.
The opt-in dashboard panel shows the selected run's predicted/twin trajectories
and labelled risks. Thresholds (45 °C or 1 °C/min rise) are demonstration
assumptions, not manufacturer limits. A constructed hot high-C-rate prediction
and separate rapid-rise case exercise the flags. No real-cell accuracy claim.
Full report: `docs/evaluations/battery-health.json`.

### #84: local fault advice and remaining measurement gap

The advisor uses compact contiguous fault events, labelled sources and fallback
guidance for all eight current fault types. Coincident unmet energy is reported
without claiming causal impact. GGUF loading is lazy/optional and local;
failures and invented readings fall back to rules. Tests use injected models,
never a network download. Model weights are ignored by git. Setup and a model
licence reference are in `docs/research.md`.

On this Linux x86_64 cloud host, the rules-only benchmark used approximately
11 MiB peak process RSS and microsecond-scale advice latency. No GGUF was
available: model name, quantization and tokens/s are **not measured** (null in
`docs/evaluations/fault-advisor.json`). This is not the requested demo-laptop
model benchmark. `python -m edge_llm.benchmark --model /path/model.gguf` reports
cold loading, latency, peak RSS, model metadata and approximate output tokens/s.
The real-model laptop benchmark remains an explicit acceptance gap; #84 must
not be considered fully closed on the strength of fake-model tests alone.

### PR #95 review follow-up

Cache the local advisor resource by model path and cache advice together with
its source by event facts, so dashboard widget reruns do not reload weights or
repeat inference. Serialize access to the shared model to keep llama.cpp calls
and source attribution consistent across concurrent sessions. Benchmark token
counts and throughput are null whenever rule-based fallback is returned, even
if a model loaded successfully; peak RSS uses the platform's native units.
Diagnostic cycle fade uses actual transferred terminal energy at SOC clamps,
and invalid degradation timesteps are rejected before state mutation. These
changes leave default simulation behavior unchanged.
