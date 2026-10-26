"""Reproducible closed-loop outcome feature ranking and paired urgency trial.

Run ``python -m explain.select_features`` with requirements-research.txt.
"""

import json

import numpy as np
import pandas as pd

from explain.features import CANDIDATES, agent_type
from simulation.compare import compare_urgency

OUTCOMES = ("critical_unmet_kwh", "flexible_unmet_kwh", "ev_shortfall_pct_at_departure")


def outcome_frame(scenarios=("clear_day", "monsoon_cloudy", "demo_sunny_home"), seeds=(1, 7, 42)):
    """Predict run outcomes from decision state; no allocation labels as inputs.

    Rows from a run are correlated. Validation must hold out entire scenarios
    and seeds, never randomly split rows. Ranking is associative, not causal.
    """
    from scenarios import build_twin
    from simulation.closed_loop import run_closed_loop
    from simulation.hybrid_arbiter import hybrid_allocate
    from simulation.metrics import score_run
    frames = []
    for scenario in scenarios:
        for seed in seeds:
            records = run_closed_loop(build_twin(scenario, closed_loop=True, seed=seed),
                                      hybrid_allocate, collect_features=True)
            score = score_run(records)
            rows = []
            for record in records:
                for name, features in record["decision_features"].items():
                    if features["requested_kw"] > 0:
                        rows.append({**features, "agent_type": agent_type(name),
                                     **{key: score[key] for key in OUTCOMES}})
            frame = pd.DataFrame(rows)
            frame["scenario"], frame["seed"] = scenario, seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def rank_features(training, validation, seed=42):
    """Held-out permutation MSE increase, separately for each agent type.

    Outcome columns are scaled by training standard deviation so EV percentage
    points cannot silently dominate the energy metrics. Negative importance is
    reported as measured. No automatic default policy promotion is performed.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    ranking = {}
    for kind in sorted(training.agent_type.unique()):
        left = training[training.agent_type == kind]
        right = validation[validation.agent_type == kind]
        if right.empty:
            continue
        scale = np.maximum(left[list(OUTCOMES)].std().to_numpy(), 1e-6)
        model = RandomForestRegressor(n_estimators=40, max_depth=6, min_samples_leaf=10,
                                      random_state=seed, n_jobs=1)
        model.fit(left[list(CANDIDATES)], left[list(OUTCOMES)].to_numpy() / scale)
        importance = permutation_importance(model, right[list(CANDIDATES)],
                                            right[list(OUTCOMES)].to_numpy() / scale,
                                            scoring="neg_mean_squared_error", n_repeats=3,
                                            random_state=seed, n_jobs=1)
        indices = np.argsort(-importance.importances_mean, kind="stable")
        ranking[kind] = [{"feature": CANDIDATES[i],
                          "importance": float(importance.importances_mean[i]),
                          "std": float(importance.importances_std[i])} for i in indices]
    return ranking


def selected_subset(ranking):
    """Select at most three positively ranked inputs; empty means retain default."""
    return {kind: [r["feature"] for r in rows if r["importance"] > 0][:3]
            for kind, rows in ranking.items()}


def selected_policy(subsets):
    """Simple equal-weight urgency candidate, never enabled implicitly.

    Scaling rules are assumptions. Low health adds pressure; persistence solar
    is scaled to the demo array and recent unmet to one kWh. This experiment
    does not claim to have optimized the selected-feature mapping.
    """
    def policy(agent_id, features):
        chosen = subsets.get(agent_type(agent_id), [])
        if features["priority"] <= 2 or not chosen:
            return features["urgency"]
        scores = []
        for name in chosen:
            value = features[name]
            if name == "battery_health":
                value = 1 - value
            elif name == "solar_forecast_kw":
                value /= 12
            scores.append(min(1., max(0., value)))
        return float(np.mean(scores))
    return policy


def main():
    ranking = rank_features(outcome_frame(), outcome_frame(("heatwave", "demo_sunny_home"), (13, 29)))
    subsets = selected_subset(ranking)
    # Separate comparison seeds from fitting/ranking seeds. All three outcome
    # columns plus the full metrics are retained in the report.
    paired = compare_urgency(selected_policy(subsets), seeds=(5, 19, 31))
    print(json.dumps({"ranking": ranking, "selected": subsets,
                      "comparison": paired.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
