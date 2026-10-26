"""Optional research tests run offline; core CI can skip missing extras."""

import numpy as np
import pandas as pd
import pytest

from explain.features import FEATURES
from explain.shap_surrogate import build_training_frame, scenario_frame
from scenarios import build_twin
from simulation.closed_loop import run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate


@pytest.fixture(scope="module")
def training():
    return scenario_frame(("demo_sunny_home", "clear_day"), (1,))


def test_feature_collection_preserves_all_default_records():
    original = run_closed_loop(build_twin("clear_day", closed_loop=True), hybrid_allocate)
    captured = run_closed_loop(build_twin("clear_day", closed_loop=True), hybrid_allocate,
                               collect_features=True)
    first = captured[0]["decision_features"]["ev_1"]
    assert first["recent_unmet_kwh"] == 0
    assert first["battery_soc_pct"] == 50
    assert captured[1]["decision_features"]["ev_1"]["battery_soc_pct"] == original[0]["battery_soc_pct"]
    for record in captured:
        record.pop("decision_features")
    assert captured == original


def test_surrogate_positive_demand_rows_only(training):
    assert np.isfinite(training[list(FEATURES)].to_numpy()).all()
    assert training.served_fraction.between(0, 1 + 1e-12).all()
    assert (training.requested_kw > 0).all()
    assert "battery_1" in set(training.agent_id)
    assert build_training_frame([]).empty


def test_shap_additivity_and_urgency_monotonicity(training):
    pytest.importorskip("sklearn")
    pytest.importorskip("shap")
    from explain.shap_surrogate import evaluate, explain_step, train_surrogate
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        model = train_surrogate(training)
        records = run_closed_loop(build_twin("demo_sunny_home", closed_loop=True), hybrid_allocate,
                                   collect_features=True)
        record = next(r for r in records if r["agents"]["ev_1"]["demand_kw"] > 0)
        result = explain_step(model, record, "ev_1")
        assert sum(result["values"].values()) == pytest.approx(
            result["prediction"] - result["expected_value"], abs=1e-6)
        assert len(result["top_factors"]) == 3
        frame = training[training.priority > 2].iloc[::25][list(FEATURES)].copy()
        low = frame.assign(urgency=0)
        high = frame.assign(urgency=1)
        assert (model.predict(high) >= model.predict(low) - 1e-12).all()
        assert evaluate(model, training)["mae"] < .15


def test_selected_urgency_is_explicit_and_paired():
    from explain.select_features import selected_policy
    from simulation.compare import compare_urgency
    neutral = selected_policy({})
    original = run_closed_loop(build_twin("clear_day", closed_loop=True), hybrid_allocate,
                               collect_features=True)
    same = run_closed_loop(build_twin("clear_day", closed_loop=True), hybrid_allocate,
                           collect_features=True, urgency_policy=neutral)
    assert same == original
    pairs = compare_urgency(selected_policy({"ev": ["soc_gap"]}), seeds=(1,), scenarios=("clear_day",))
    for metric in ("critical_unmet_kwh", "flexible_unmet_kwh", "ev_shortfall_pct_at_departure"):
        assert pairs[f"delta_{metric}"].iloc[0] == pytest.approx(
            pairs[f"selected_{metric}"].iloc[0] - pairs[f"current_{metric}"].iloc[0])
    assert pairs.selected_energy_balance_max_error_kw.iloc[0] < 1e-6


def test_feature_ranking_reproducible_with_held_out_data():
    pytest.importorskip("sklearn")
    from explain.features import CANDIDATES
    from explain.select_features import OUTCOMES, rank_features, selected_subset
    rng = np.random.default_rng(2)
    def frame():
        data = pd.DataFrame(rng.random((100, len(CANDIDATES))), columns=CANDIDATES)
        data["agent_type"] = "ev"
        for name in OUTCOMES:
            data[name] = data.soc_gap * 10
        return data
    left, right = frame(), frame()
    ranking = rank_features(left, right)
    assert ranking == rank_features(left, right)
    assert ranking["ev"][0]["feature"] == "soc_gap"
    assert selected_subset(ranking)["ev"][0] == "soc_gap"


@pytest.mark.parametrize("value", [float("nan"), -1, 2])
def test_urgency_hook_rejects_invalid_outputs(value):
    with pytest.raises(ValueError, match="urgency_policy"):
        run_closed_loop(build_twin("clear_day", closed_loop=True), hybrid_allocate,
                        steps=1, urgency_policy=lambda name, features: value)


def test_feature_capture_and_neutral_urgency_preserve_enabled_grid():
    from simulation.grid import GridConfig
    from simulation.metrics import score_run
    config = GridConfig(enabled=True, import_limit_kw=3, export_limit_kw=2, tariff=[{
        "start_min": 0, "end_min": 1440, "import_price_per_kwh": 2, "export_price_per_kwh": .5}])
    original = run_closed_loop(build_twin("demo_sunny_home", closed_loop=True), hybrid_allocate,
                               grid_config=config)
    captured = run_closed_loop(build_twin("demo_sunny_home", closed_loop=True), hybrid_allocate,
                               grid_config=config, collect_features=True,
                               urgency_policy=lambda name, features: features["urgency"])
    for record in captured:
        record.pop("decision_features")
    assert captured == original
    assert score_run(captured) == score_run(original)
