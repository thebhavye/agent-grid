# AgentGrid — Known Limitations

- There is no grid connection: solar beyond load and battery headroom is
  curtailed (reported as curtailed_kw in closed-loop records), and load the
  solar and battery cannot cover is unmet. Grid import/export is issue #61.
- Appliance missed-window tracking is a per-timestep in-memory buffer
  (self.missed_windows, cleared on update_state, deduplicated per
  timestep — see decisions.md, issue #10), not persisted, logged to
  file, or user-facing.
- EV target SOC and ready-by time are configurable per agent or per row
  (issue #62); the generator does not emit preference columns yet. Defaults
  remain 100% SOC by the reported departure. Twin charging segments are
  configured separately and are not changed by agent preferences.
- Appliance deferable window ramp assumes a fixed 4-hour (240 min) window
  for all deferable appliances — not derived per-appliance from the
  dataset.
- compute_urgency() has no cross-agent normalization: each agent type scales
  urgency to [0,1] with its own formula, yet the hybrid arbiter ranks
  flexible agents of different types by these raw values.
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
- run_simulation() and run_twin_simulation() replays are open-loop:
  allocations do not drive battery/EV SOC or appliance completion. Use
  run_closed_loop() or run_graph_closed_loop() for outcome comparisons.
- The hybrid arbiter is not uniformly better than the proportional
  baseline: in the paired comparison it reduces critical unmet energy but
  increases flexible unmet energy and EV departure shortfall
  (simulation/comparison_results.md).
- simulation/compare.py builds its paired runs from the mocked test-weather
  fixture (importing helpers from digital_twin/test_digital_twin.py) rather
  than the saved scenario library, so its published results are not on
  real weather.
- The dashboard runs the plain closed-loop runner, not the LangGraph graph
  (their records are tested to be identical). Under injected sensor faults,
  explanations for affected steps can fall back to describing outcomes without
  policy reasons.
- EVAgent.compute_urgency() has a dead code path (minutes_left <= 0):
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
- Default twin seed 42 always yields the evening_topup EV scenario; the
  overnight path is only exercised via explicit-scenario tests.
- Twin battery clamps to [0, 100] (hits 0.0 on some seeds) while the
  generator clamps to [5, 100] + noise (floor ~4.5).
- Twin EV holds target SOC after the 80% charge phase and gates noise on
  charging_kw > 0; the generator ramps through the full phase with noise
  while plugged.
- `1440` is hard-coded in 5 places;
  the value 5.0 now has three unrelated meanings (battery max, legacy EV
  fallback, twin ev_charger rating).
- Degenerate roster edge: an ApplianceAgent with usual_start == usual_end
  is always "outside window" and logs a missed-window event every timestep.
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
  The demo_sunny_home scenario does exactly this (60 m²) so the battery
  cycles visibly; the other scenarios keep the 20 m² default.
