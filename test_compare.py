"""Deterministic offline paired comparisons and fairness failures."""

from copy import deepcopy

import pandas as pd
import pytest
import requests

from simulation import compare as comparison
from simulation.metrics import METRICS


@pytest.fixture
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("comparison attempted network access")
    monkeypatch.setattr(requests.sessions.Session, "request", fail)


def test_all_nine_pairs_offline_deterministic_and_complete(no_network):
    first = comparison.compare()
    second = comparison.compare()
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 9
    assert set(first.scenario) == {"overnight", "midday", "evening_topup"}
    assert set(first.seed) == {1, 7, 42}
    for metric in METRICS:
        assert first[f"delta_{metric}"].tolist() == (
            first[f"hybrid_{metric}"] - first[f"baseline_{metric}"]
        ).tolist()
    for method in ("hybrid", "baseline"):
        assert (first[f"{method}_energy_balance_max_error_kw"] < 1e-6).all()
        assert (first[f"{method}_battery_reserve_violation_steps"] == 0).all()
    rendered = comparison.results_markdown(first)
    assert len(rendered.splitlines()) == 2 + 9 * len(METRICS)


@pytest.mark.parametrize("field", ["solar_kw", "ev_plugged_in", "ev_departure_min",
                                   "ev_target_soc_pct", "drawn_cycle_starts", "dt_min"])
def test_schedule_mismatch_fails_before_scoring(monkeypatch, field, no_network):
    original = comparison.run_closed_loop
    calls = 0

    def changed(*args):
        nonlocal calls
        records = original(*args)
        calls += 1
        if calls == 2:
            records[0][field] = "different"
        return records

    monkeypatch.setattr(comparison, "run_closed_loop", changed)
    with pytest.raises(RuntimeError, match=field):
        comparison.compare(seeds=(42,), scenarios=("overnight",))


def test_initial_condition_mismatch_fails(monkeypatch):
    original = comparison._make_twin
    calls = 0

    def changed(*args):
        nonlocal calls
        twin = original(*args)
        calls += 1
        if calls == 2:
            twin._battery._soc_pct -= 1
        return twin

    monkeypatch.setattr(comparison, "_make_twin", changed)
    with pytest.raises(RuntimeError, match="initial conditions"):
        comparison.compare(seeds=(42,), scenarios=("overnight",))


def test_large_balance_error_fails(monkeypatch, no_network):
    original = comparison.score_run

    def bad_score(records):
        result = original(records)
        result["energy_balance_max_error_kw"] = 1e-6
        return result

    monkeypatch.setattr(comparison, "score_run", bad_score)
    with pytest.raises(RuntimeError, match="power balance failed"):
        comparison.compare(seeds=(42,), scenarios=("overnight",))


def test_fixture_weather_is_isolated_from_shared_cache(no_network):
    from digital_twin.weather_client import _cache
    before = deepcopy(_cache)
    twin = comparison._make_twin(1, "midday")
    comparison.run_closed_loop(twin, comparison.hybrid_allocate, steps=1)
    assert _cache == before
    assert twin._weather_client._session.urls


def test_different_record_lengths_rejected():
    with pytest.raises(RuntimeError, match="step counts"):
        comparison._assert_paired([], [{}])
