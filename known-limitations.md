# AgentGrid — Known Limitations

- Surplus solar is discarded, not tracked anywhere in allocator output
  (see decisions.md, issue #5).
- Appliance missed-window tracking is a per-timestep in-memory buffer
  (self.missed_windows, cleared on update_state, deduplicated per
  timestep — see decisions.md, issue #10), not persisted, logged to
  file, or user-facing.
- EV target SOC is hardcoded to 100% in get_demand_kw() and
  compute_urgency() — no per-row or per-agent target field exists in the
  dataset yet.
- Appliance deferable window ramp assumes a fixed 4-hour (240 min) window
  for all deferable appliances — not derived per-appliance from the
  dataset.
- compute_urgency() has no cross-agent normalization — each agent's
  urgency is independently scaled to [0,1] but not compared or weighted
  against other agents yet. Relevant once Phase 2 bidding needs to rank
  agents against each other.
- Battery/EV max charge rate constants (BATTERY_MAX_CHARGE_KW, EV max
  charge kw) are hardcoded, not derived from a shared config or the
  dataset generator's own capacity assumptions.
- Battery/EV agents reject non-finite state values (SOC, departure, timestamp)
  with ValueError (fail fast at the source, issue #26); finite negative SOC
  is accepted and produces proportional demand. The baseline allocator
  rejects non-finite or negative power values (available/demand) with
  ValueError. ApplianceAgent still substitutes 1.0 kW for missing/NaN rated
  power (graceful default, not an error).
- EVAgent.ev_plugged_in bool() coercion mistreats string values like
  "False" as truthy; not reachable via generator output (real bools), only
  hand-built test rows.
- String timestamps are parsed permissively rather than strictly validated
  as HH:MM: malformed strings leave the parsed time unset (EV urgency
  becomes 0.0; appliance window checks treat the time as unknown), while
  out-of-range integer components such as "25:90" are accepted. EVAgent's
  non-finite numeric timestamp rejection does not validate string clocks.
- Current static and twin-backed allocation runs are open-loop: allocations
  do not drive battery/EV SOC or appliance completion. Outcome comparisons
  require the planned Phase 2 closed-loop twin runner (see decisions.md).
- EVAgent has a dead code path (minutes_left <= 0, ev_agent.py:59-60):
  unreachable with real data (the generator unplugs exactly at the
  departure minute), untested.
- Duplication/drift risk: overnight-wrap arithmetic is duplicated across
  ev_agent.py and appliance_agent.py with different variable names;
  BATTERY_MAX_CHARGE_KW and the EV legacy fallback constant coincidentally
  share the value 5.0 with unrelated meanings; README documents
  appliance_usual_start_min as int but the actual DataFrame dtype is
  float64 (NaN forces float).
- Twin weather cache is module-global and unbounded (no TTL): shared across
  clients, grows on multi-location runs, and stale forecasts never refresh
  without an explicit clear_cache().
- DigitalTwin.simulated_minutes ignores per-tick dt (step_count * default);
  variable-dt runs report the wrong elapsed time.
- Default twin seed 42 always yields the evening_topup EV scenario; the
  overnight path is only exercised via explicit-scenario tests.
- Twin battery clamps to [0, 100] (hits 0.0 on some seeds) while the
  generator clamps to [5, 100] + noise (floor ~4.5).
- Twin EV holds target SOC after the 80% charge phase and gates noise on
  charging_kw > 0; the generator ramps through the full phase with noise
  while plugged.
- Dead code/lint (no linter configured): unused `field` imports in
  twin/battery/appliance models, unused `import time` in weather_client,
  dead `prev_plugged` (ev_model), `_solar_initialized` (twin),
  `next_cycle_start` (appliance state); `1440` hard-coded in 5 places;
  the value 5.0 now has three unrelated meanings (battery max, legacy EV
  fallback, twin ev_charger rating).
- Degenerate roster edge: an ApplianceAgent with usual_start == usual_end
  is always "outside window" and logs a missed-window event every timestep.
- `.pytest_cache/` is not in .gitignore (untracked locally, but one
  `git add -A` would sweep it in).
- Digital twin solar peak is ~50% below generator peak (issue #29):
  twin default (20 m², 20%, 4 kWp) peaks at ~2.78 kW under 800 W/m² clear-sky
  conditions with temperature derating; generator peaks at 5.35–5.73 kW
  (synthetic peak_kw ~ Uniform(4.5, 6.0)). Twin also lacks generator's
  explicit cloud-dip volatility (twin CV ≈ 0.41 vs generator CV ≈ 0.58–0.73
  in mocked tests, using output > 0.1 kW and population standard deviation;
  see decisions.md for reproduction). CV includes the daily curve shape,
  not only cloud variability. Live-cloud volatility was not measured.
  This is a modeling-choice divergence, not a bug:
  twin = specific physical system + real weather; generator = statistical
  synthetic solar curves + synthetic clouds. Schema compatibility
  is maintained; users can configure a larger panel_area_m2 if needed.
