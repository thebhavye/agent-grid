# Optional research modules

Core simulation and tests need only `requirements.txt`. All examples below use
saved scenario weather, fixed seeds and local computation. They make no API
calls or automatic model downloads. Install extras in your virtual environment:

```sh
python -m pip install -r requirements-research.txt -r requirements-battery.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m explain.shap_surrogate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m explain.select_features
python -m battery_health.train --output outputs/battery-health.pt
python -m pytest . -m 'not live'
```

The thread limits avoid excessive BLAS/OpenMP concurrency on large hosts.
Training and evaluation reports from this change are in `docs/evaluations/`.
They are synthetic-twin results, not measurements of a real household or cell.

## Facts-first rendering (#59)

`render_template(facts)` remains available without dependencies. Optional
`render_llm(facts, client, enabled=True, model=...)` takes an explicitly supplied
callable `client(prompt, model=...) -> str`. A model can also be configured through
`AGENTGRID_EXPLAIN_MODEL`; no provider, endpoint or API key is chosen by the library.
The prompt contains instructions and only the computed facts JSON, not raw logs.
Without `enabled=True`, the client is never called. External disclosure remains
an owner decision; setting an API key alone does not enable it.

A response containing an unsupported numeric token (including scientific
notation), empty/non-text output, or a client exception falls back to the
six-decimal template. Numeric tokens are compared at six decimal places. This
cannot verify the meaning of a sentence, detect spelled-out numbers, or prove
that a supported number was used in the right context. Never treat the numeric
guard as a guarantee of factual correctness. Optional `surrogate_factors` facts
are labelled as model contributions rather than arbiter causes.

## SHAP surrogate (#83)

`build_training_frame(records)` uses captured pre-decision features when
`run_closed_loop(..., collect_features=True)` was requested. Legacy records can
be reconstructed, but their stored SOC is post-commit and battery charging bids
may have been capped: those inputs are approximate. Dashboard runs capture the
exact features. Solar agents, zero requests and non-charging battery rows are
excluded. Served fraction uses the allocator's actual requested power.

A small scikit-learn histogram gradient boosting model has a monotonic urgency
constraint. It trains on clear-day/cloudy scenarios, seeds 1/7/42, and reports
R²/MAE on heatwave, seed 13. `explain_step` checks SHAP additivity and returns all
contributions plus the top three. The dashboard's optional **Why (SHAP)** panel
shows them for a selected hybrid agent/step, with holdout fidelity visible.
It does not explain the proportional strategy or fault-modified inputs. Model
predictions can differ from the arbiter and are not causal explanations.

## Feature selection (#86)

`python -m explain.select_features` fits a separate outcome surrogate for each
agent type. Inputs are SOC gap, time pressure to departure/window end, window
width, priority pressure, battery health, time of day, a one-hour persistence
solar forecast, and an exponentially decayed history of unmet energy (one-hour
time constant). The forecast is explicitly a persistence assumption; it never
looks ahead into the saved weather. The history is updated after the decision.

Targets are the run's critical unmet kWh, flexible unmet kWh and EV departure
shortfall, scaled by their training standard deviations. Permutation importance
on validation data ranks inputs by increase in mean squared error. All rows
from a run stay together. Training uses clear_day, monsoon_cloudy and
demo_sunny_home, seeds 1/7/42. Ranking uses heatwave and demo_sunny_home, unseen
seeds 13/29: the sunny scenario is included because the heatwave day has no
battery charging requests. Thus battery validation holds out seeds, not a
scenario. The small number of independent runs and correlated per-step rows
limit conclusions; the permutation standard deviation is not a run-level
confidence interval. Battery health is constant in this experiment, so its
zero importance is not evidence that health is irrelevant.

At most three positively ranked features are selected per type. An equal-weight
urgency candidate uses those inputs (documented fixed scaling; not an optimized
policy). `simulation.compare.compare_urgency` pairs it against unchanged hybrid
urgency on fresh twins, using existing initial-state/schedule fairness checks,
scoring and power-balance validation. Seeds 5/19/31 are separate from ranking.
Results are mixed; **default agent formulas remain unchanged**.

## Battery health (#85)

The PINN builds on the merged #69 API without changing its aging physics,
dataset schema, baseline or runner telemetry. See [battery documentation](../battery_health/README.md)
for coefficient assumptions and columns. `daily_frame` uses `summarize_days`
and adds physics targets from known operation; they do not read measured
health/temperature labels. Its thermal recurrence assumes default catalog
thermal parameters.

`python -m battery_health.train` calls the unchanged `generate_battery_dataset`
with DEFAULT_SCENARIOS, three days and seed 42. PINN, no-physics ablation and
`fit_baseline` use the same data, train scenarios (demo_sunny_home,
degraded_battery), held-out scenario (heatwave), and targets (day-end health,
daily peak temperature). Inputs combine the baseline's health and thermal
features. These fits condition on the day's actual operating exposures; they
are not autonomous forecasts. Six training rows are particularly limited.
The PINN loses to the empirical baseline on both targets; the full numbers
are in `docs/evaluations/battery-health.json` and `decisions.md`.

The Battery health experiment dashboard uses existing battery_* records for
the selected run and shows daily health/peak-temperature fits. Predicted peak
risk uses a demonstration 45 °C threshold, not an instantaneous rate estimate.
The separate thermal-risk helper supports 1 °C/min rise warnings. Neither
threshold is a manufacturer safety limit. No weights or real-cell accuracy
claims are committed.

## Local maintenance advisor (#84)

`summarize_faults(records)` groups contiguous device/sensor labels, retaining
observed/used values and coincident unmet energy. Coincident energy is not a
causal estimate of fault impact. `FaultAdvisor.advise(event)` always returns
rules-only guidance when no model is supplied, a load/inference fails, or the
numeric guard rejects the generated text. Every current fault type has fallback
advice. Model loading is lazy; importing the package requires no llama.cpp.
The dashboard's **Advisor** panel labels each event's advice source.

To use a local model, install `requirements-edge.txt`, obtain a GGUF file and set
`AGENTGRID_GGUF_MODEL=/absolute/path/model.gguf`. One option is
[Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF),
`qwen2.5-1.5b-instruct-q4_k_m.gguf` (Apache-2.0; verify the model card and licence
before downloading). Download it manually from that page; no weights are
committed. llama-cpp-python may require a native compiler for installation.
Prompts contain compact event facts only; generation stays local.

```sh
python -m edge_llm.benchmark --model /absolute/path/model.gguf
```

This reports host, model metadata/quantization, cold load time, advice latency,
peak process RSS, output tokens and approximate tokens/s. Tokens/s includes
prompt processing in the advice interval. Run it on the actual demo laptop:
this cloud host has no GGUF file and cannot provide that measurement. The
committed report records **fallback only**, with model and tokens/s null. Real
model quality and laptop latency/memory remain an explicit acceptance gap for
#84; the offline fake-model tests are not a substitute.
