# Demo video script and recording checklist (Issue #82)

Status: script prepared; video **not recorded or uploaded**. No demo URL exists
in this document. Requested deadline: **16 October 2026, 23:50 IST**.
Recording/upload and adding the resulting URL to the project-document appendix
(point 16b) require the presenter. That external document is not in this repo.

## Readiness gates

- [ ] PR #80 (dashboard) is merged and present in the recording checkout.
- [ ] Issue #81's `demo_sunny_home` is present and is the initial selection.
      PR #88 is stacked on #80; merge/retarget in that order, not by copying code.
- [ ] Decide whether to include the optional graph segment: #65 / PR #87 must
      be present and tested on the recording commit. Otherwise skip it.
- [ ] Choose a clean checkout and record `git rev-parse HEAD` and
      `git status --short`. Do not record a mixture of uncommitted branches.
- [ ] Create a fresh Python 3.13 virtual environment. Do not reuse global Python.
- [ ] Install `requirements-dashboard.txt`, run the complete offline suite on
      the exact recording commit, and run the pinned Ruff check. Save real results.
- [ ] Start the dashboard, open <http://localhost:8501>, and confirm it renders.
- [ ] Confirm Overview, Agents & decisions, Hybrid vs baseline, and Faults tabs
      work. Change a scenario, strategy and inspected step before recording.
- [ ] Check KPI text and values fit at the chosen browser zoom and 1920x1080
      capture size. The current dashboard has two rows of three KPIs, not six
      tiles on one row. Check legends, tables and explanations for clipping.
- [ ] Test every fault preset and both sensor policies; restore No faults,
      Hybrid arbiter, `demo_sunny_home`, and the intended initial slider position.
- [ ] Close unrelated apps/tabs, hide notifications, secrets and terminal paths
      where appropriate, and test microphone/system audio. Record clean 1080p.

### Fresh-environment commands (PowerShell, in a new recording checkout)

```powershell
git rev-parse HEAD
git status --short
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest . -m "not live"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --browser.gatherUsageStats=false
```

Use a new checkout if `.venv` already exists; do not overwrite someone else's
environment. The last command is the virtualenv equivalent of
`python -m streamlit run dashboard/app.py`. No weather API or LLM credentials
are needed. If a readiness gate fails, stop and fix it before recording.

## Run of show (5:35 with graph, 5:05 without)

Speak naturally; timings include moving the pointer and waiting for the charts.
Read values from the recorded run rather than memorizing old measurements.

### 0:00-0:30 - The household problem

**Show:** sidebar and scenario description, no faults, Hybrid arbiter.

**Say:** "This home has solar panels, a battery, an electric vehicle and several
appliances. Their requests can exceed the available power. AgentGrid models how
we share that power: which loads receive it, what stays unmet, and how those
decisions affect the devices over time. This is a simulation, not a deployed
home controller."

### 0:30-1:15 - Digital twin and saved weather

**Show:** Overview power-flow and SOC charts over the full day.

**Say:** "Each five-minute interval uses saved Vellore weather from Open-Meteo's
historical archive. These profiles are historical reanalysis, not readings from
a sensor installed at this house. The simulation runs offline. Allocated power
feeds back into battery and EV SOC and appliance progress. The sunny demo uses
a deliberately larger 60-square-metre array: 12 kWp at the model's standard
20-percent efficiency, not a promise of 12 kW throughout the day. Its 12-kWh
battery starts half full. The original scenarios retain their original sizing."

### 1:15-2:15 - Urgency scoring and bids

**Show:** Agents & decisions; inspect 12:30, then 19:30. Point to the battery
mode and per-agent priority, urgency, requested power and received power.

**Say:** "Each agent submits its demand and an urgency score between zero and
one. Priority and urgency are separate: critical loads have a protected tier,
while flexible loads compete by urgency and deterministic tie-breaks. The
battery also has physical charge and discharge limits and a reserve. At midday
there can be enough solar to charge it. At night it can supply loads until the
reserve limits its discharge. Notice the actual requests in the table: an EV
that is away, or an inactive appliance, must not be described as a deferred
positive request. Let's compare the requested and received columns rather than
assume every appliance is always running."

### 2:15-3:00 - One real arbitration decision

**Show:** 19:30 on the default no-fault hybrid run, its explanation and bids.

**Say:** "Hybrid serves priority 1 before priority 2, sharing within each
critical pool if necessary, then serves flexible urgency groups. The text here
is generated from computed facts and policy rules, not an external LLM. For
this specific step, read the solar, battery supply and actual served loads.
Check that the power adds up. This is one observed decision, not evidence that
hybrid wins every outcome."

**Verified rehearsal example, recompute on the recording commit:** at 19:30,
`demo_sunny_home` with hybrid and no faults has zero solar and 1.05 kW battery
discharge: base load 0.60, fridge 0.15, lighting 0.25, security 0.05 kW. EV and
flexible appliances request zero here; do not claim they were deferred.

### 3:00-4:00 - Paired evaluation

**Show:** switch strategy, return to hybrid, then Hybrid vs baseline tab.

**Say:** "These are separate closed-loop twins with the same initial state,
weather, seeds and schedule draws. Only allocation changes. We compare actual
outcomes, not a replay of an open-loop SOC trace. Read critical unmet energy,
flexible unmet energy, EV departure shortfall, curtailment and battery energy
from this table. Lower unmet energy or shortfall is better; more battery
throughput is not automatically better. In the tested sunny demo, hybrid
protects critical load better but leaves more flexible energy unmet. Equal or
worse outcomes stay in the table. This is limited simulation evidence, not a
general claim of optimality. No grid bill is shown unless grid support actually
exists and has been tested."

### 4:00-4:45 - Device and sensor robustness

**Show:** inverter trip preset, Faults tab around 12:00; then battery SOC sensor
dropout around 08:00 with Hold last good value. Show substitutions and balance.

**Say:** "An inverter trip removes delivered solar during its configured window;
the battery helps only within its available energy and limits. The SOC sensor
dropout is different: it corrupts an observation, not the physical battery.
The runner can hold a last good reading for a bounded number of steps, then
exclude the affected agent. Each substitution is logged. Truth and sanitized
agent observations remain separate, and power balance is checked throughout.
These are injected, labelled faults, not automatic fault detection or diagnosis."

### 4:45-5:15 - Optional LangGraph orchestration

**Include only when #65 is present on the tested recording commit.** The current
dashboard has no "How it works" tab; do not promise or search for one.
Show `orchestration/graph.py` and its independent bid-node fan-out and join:

```text
START -> independent bid_<agent_id> nodes -> battery_mode -> arbiter -> commit -> END
```

**Say:** "LangGraph orchestrates the same deterministic steps. Each bidding
node is independent; all bids join before battery mode, allocation and commit.
Tests require exact record equality with the ordinary runner. It is orchestration,
not LLM negotiation, and remote tracing is disabled."

If skipped, go straight to the roadmap at 4:45; the video ends around 5:05.

### 5:15-5:35 - Roadmap (4:45-5:05 without graph)

**Say:** "Next steps include PINN battery-health and thermal monitoring, a local
edge LLM for fault interpretation, and SHAP explanations. These are future work,
not capabilities demonstrated today. What we have shown is the offline twin,
deterministic bidding and arbitration, measured trade-offs and labelled faults."

## After recording (manual completion of Issue #82)

- [ ] Watch the full exported video; confirm readable 1080p output, audio,
      approximately 5-6 minute duration, and no exposed secrets or unrelated apps.
- [ ] Check every numerical claim against the exact commit/run shown. Re-record
      any statement that contradicts the displayed strategy, fault or timestep.
- [ ] Upload to the owner's approved destination and verify viewer access.
      This private repository and its simulation data must not be made public
      merely to obtain a demo link.
- [ ] Add the real, verified video URL to the project document appendix, point
      16b, with recording commit SHA and date. No placeholder URL counts.
- [ ] Confirm the upload/link deadline was actually met before claiming it.

Documentation alone does not close Issue #82's recorded-and-uploaded criterion.
