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

## Battery health (#85 and minimum #69 prerequisites)

`BatteryModelConfig(degradation=True)` opts into diagnostic empirical cycle and
calendar fade. Every coefficient is named and explicitly an assumption in
`battery_model.py`; none is presented as measured cell chemistry. Health does
not change nominal capacity, dispatch or arbiter behavior. With degradation off,
records and scores match the original code. The existing first-order thermal
model is unchanged.

`generate_battery_dataset(scenarios, days, seed)` produces per-step closed-loop
rows with pre/post SOC, signed terminal power, C-rate, cumulative throughput,
cell/ambient temperature and health. `daily_summary(frame)` produces day-end
health and peak temperature. Columns and units are documented in its docstring.
Save locally with `frame.to_csv('outputs/battery.csv', index=False)`. Outputs and
weights are ignored by git. Default dataset covers three scenarios and three
days each. No static open-loop SOC curve is used.

The baseline fits least squares on the same empirical aging and thermal terms.
The small CPU network predicts one-step health loss and temperature change,
with data loss plus discrete aging/thermal residuals and a penalty for increasing
health. The report compares it with the same network/seed without physics loss.
Training uses clear/cloudy, three days, seed 42; holdout uses heatwave, three days,
seed 13. Day-end health errors accumulate one-step predicted losses from the
observed start state; temperature predictions use observed previous state.
These are **not autonomous multi-day forecasts**. Physics targets largely
repeat the noise-free simulated labels; the empirical baseline has an inherent
advantage and the PINN does not beat it. No real-cell predictive claim is made.
Weights reproduce in seconds on this host, so none are committed.

The **Battery health experiment** dashboard toggle enables telemetry for the
selected run and shows predicted/twin health and temperature trends, with risk
flags at the selected moment. Demonstration warning thresholds are 45 °C or
1 °C/min rise. They are assumptions, not manufacturer safety thresholds.

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
