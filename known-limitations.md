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
- Battery/EV agents don't validate NaN input — NaN SOC produces
  contradictory demand/urgency pairs (e.g. Battery demand 0.0 with urgency
  1.0); not reachable with current generator output, which never emits NaN.
- EVAgent.ev_plugged_in bool() coercion mistreats string values like
  "False" as truthy; not reachable via generator output (real bools), only
  hand-built test rows.
- EVAgent has a dead code path (minutes_left <= 0, ev_agent.py:59-60):
  unreachable with real data (the generator unplugs exactly at the
  departure minute), untested.
- Duplication/drift risk: overnight-wrap arithmetic is duplicated across
  ev_agent.py and appliance_agent.py with different variable names;
  BATTERY_MAX_CHARGE_KW and the EV legacy fallback constant coincidentally
  share the value 5.0 with unrelated meanings; README documents
  appliance_usual_start_min as int but the actual DataFrame dtype is
  float64 (NaN forces float).
