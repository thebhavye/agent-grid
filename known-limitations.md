# AgentGrid — Known Limitations

- Surplus solar is discarded, not tracked anywhere in allocator output
  (see decisions.md, issue #5).
- Appliance missed-window tracking is in-memory only (self.missed_windows),
  not persisted, logged to file, or user-facing.
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
