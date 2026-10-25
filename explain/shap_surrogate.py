"""Optional arbiter surrogate: fidelity is predictive, not causal evidence.

Run ``python -m explain.shap_surrogate`` for a scenario-held-out evaluation.
"""

import json

import numpy as np
import pandas as pd

from explain.features import FEATURES, record_features


def build_training_frame(records):
    """One positive-request row per step/agent; storage only bids in charge mode."""
    rows = []
    for step, record in enumerate(records):
        for name, agent in record["agents"].items():
            if (agent["demand_kw"] <= 0 or name == "solar_1"
                    or (name == "battery_1" and record["mode"] != "charge")):
                continue
            features = record_features(record, name)
            rows.append({**features, "step": step, "agent_id": name,
                         "served_fraction": agent["allocated_kw"] / features["requested_kw"]})
    return pd.DataFrame(rows)


def train_surrogate(frame, seed=42):
    """Train deterministic trees with a non-decreasing urgency constraint."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=15, learning_rate=.08, l2_regularization=.1,
        monotonic_cst=[1 if key == "urgency" else 0 for key in FEATURES], random_state=seed,
    )
    model.fit(frame[list(FEATURES)], frame.served_fraction)
    return model


def evaluate(model, frame):
    from sklearn.metrics import mean_absolute_error, r2_score
    predicted = model.predict(frame[list(FEATURES)])
    return {"r2": float(r2_score(frame.served_fraction, predicted)),
            "mae": float(mean_absolute_error(frame.served_fraction, predicted))}


def explain_step(model, record, agent_id):
    """Top three SHAP factors and full additive decomposition of a prediction."""
    import shap
    row = pd.DataFrame([record_features(record, agent_id)])[list(FEATURES)]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(row))[0]
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    prediction = float(model.predict(row)[0])
    if not np.isclose(values.sum(), prediction - expected, atol=1e-6):
        raise RuntimeError("SHAP additivity check failed")
    indices = np.argsort(-np.abs(values), kind="stable")[:3]
    return {"prediction": prediction, "expected_value": expected,
            "values": dict(zip(FEATURES, map(float, values))),
            "top_factors": [{"feature": FEATURES[i], "value": float(row.iloc[0, i]),
                             "shap_value": float(values[i])} for i in indices]}


def scenario_frame(scenarios=("clear_day", "monsoon_cloudy"), seeds=(1, 7, 42)):
    from scenarios import build_twin
    from simulation.closed_loop import run_closed_loop
    from simulation.hybrid_arbiter import hybrid_allocate
    frames = []
    for scenario in scenarios:
        for seed in seeds:
            twin = build_twin(scenario, closed_loop=True, seed=seed)
            records = run_closed_loop(twin, hybrid_allocate, collect_features=True)
            frame = build_training_frame(records)
            frame["scenario"], frame["seed"] = scenario, seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    training = scenario_frame()
    heldout = scenario_frame(("heatwave",), (13,))
    print(json.dumps({"training_rows": len(training), "heldout_rows": len(heldout),
                      "heldout_scenario": "heatwave", **evaluate(train_surrogate(training), heldout)},
                     indent=2))
