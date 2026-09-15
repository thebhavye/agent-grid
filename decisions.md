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
