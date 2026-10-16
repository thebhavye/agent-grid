"""Issue #67: explicit day-bounded EV schedules and midnight continuity."""

import random
from datetime import datetime

import pytest

from digital_twin import EVModel, EVModelConfig, EVScenario, EVSegment
from digital_twin.test_digital_twin import make_test_twin


def overnight():
    # One overnight connection represented in the existing sorted day convention.
    return [EVSegment(0, 360, 15, 80), EVSegment(1410, 1440, 15, 80)]


@pytest.mark.parametrize("segments", [
    [], [None], [dict(plug_in_min=0, departure_min=60, start_soc=15, target_soc=80)],
    [EVSegment(60, 120, 15, 80), EVSegment(0, 30, 15, 80)],
    [EVSegment(0, 60, 15, 80), EVSegment(30, 90, 15, 80)],
])
def test_invalid_segment_lists(segments):
    with pytest.raises(ValueError, match="segments"):
        EVModelConfig(segments=segments)


@pytest.mark.parametrize("start,end", [
    (-1, 60), (60, 60), (120, 60), (0, 1441), (1410, 360),
    (float("nan"), 60), (0, float("nan")), (0, float("inf")),
    (float("-inf"), 60), (True, 60), (0, False), ("0", 60), (0, None),
])
def test_invalid_segment_times(start, end):
    with pytest.raises(ValueError, match="segments"):
        EVModelConfig(segments=[EVSegment(start, end, 15, 80)])


@pytest.mark.parametrize("field", ["start_soc", "target_soc"])
@pytest.mark.parametrize("value", [-0.01, 100.01, float("nan"), float("inf"),
                                  float("-inf"), True, "15", None])
def test_invalid_segment_soc(field, value):
    kwargs = {"plug_in_min": 0, "departure_min": 60, "start_soc": 15, "target_soc": 80}
    kwargs[field] = value
    with pytest.raises(ValueError, match="SOC|finite number"):
        EVModelConfig(segments=[EVSegment(**kwargs)])


def test_valid_boundaries_fractional_times_and_adjacent_intervals():
    segments = [EVSegment(0, 0.5, 0, 100), EVSegment(0.5, 1440, 100, 0)]
    model = EVModel(EVModelConfig(segments=segments))
    assert model.scenario == "custom"
    assert model.departure_min == 0  # Existing rounded first-departure reporting.
    assert segments[0].start_min == segments[0].plug_in_min
    assert segments[0].end_min == segments[0].departure_min


@pytest.mark.parametrize("seed", [0, 1, 7, 42, None])
def test_custom_skips_scenario_selection_and_segment_rng_draws(seed, monkeypatch):
    def fail(*args):
        pytest.fail("custom schedule called random segment generation")

    monkeypatch.setattr(EVModel, "_generate_segments", fail)
    model = EVModel(EVModelConfig(scenario=EVScenario.MIDDAY, seed=seed, segments=overnight()))
    assert model.scenario == "custom"
    assert model.departure_min == 360
    if seed is not None:
        assert model._rng.getstate() == random.Random(seed).getstate()


def test_custom_model_midnight_continuity_and_departure():
    model = EVModel(EVModelConfig(segments=overnight(), max_charge_kw=7))
    model.initialize_at(datetime(2024, 1, 15, 23, 25))
    observation = model.observe(5)
    assert observation["plugged_in"]
    assert observation["soc_pct"] == 15
    first = model.commit_delivered(5, 100)
    assert first["charging_kw"] == pytest.approx(7)
    assert first["soc_pct"] == pytest.approx(15 + 7 * 5 / 60 * 0.9 / 60 * 100)
    previous_soc = model.soc_pct
    # No delivery across midnight: SOC must neither reset nor charge autonomously.
    for _ in range(6):
        assert model.observe(5)["plugged_in"]
        assert model.commit_delivered(5, 0)["soc_pct"] == previous_soc
    assert model.current_minute == 0
    for _ in range(71):
        assert model.observe(5)["plugged_in"]
        model.commit_delivered(5, 0)
    assert model.current_minute == 355
    assert not model.observe(5)["plugged_in"]
    with pytest.raises(ValueError, match="unplugged"):
        model.commit_delivered(5, 1)
    assert model.commit_delivered(5, 0)["soc_pct"] == previous_soc


@pytest.mark.parametrize("closed_loop", [False, True])
def test_twin_overnight_exact_transition_labels_and_reset(closed_loop):
    twin = make_test_twin(
        start=datetime(2024, 1, 15, 23, 25), closed_loop=closed_loop,
        ev_config=EVModelConfig(segments=overnight()),
    )

    def run():
        rows = []
        observations = []
        for _ in range(81):
            if closed_loop:
                observation = twin.observe()
                observations.append(observation)
                rows.append(twin.commit({"ev_1": 2 if observation["ev_plugged_in"] else 0}))
            else:
                rows.append(twin.tick())
        return rows, observations

    first, observations = run()
    by_time = {row["timestamp"]: row for row in first}
    assert not by_time["23:25"]["ev_plugged_in"]
    assert by_time["23:30"]["ev_plugged_in"]
    assert by_time["23:55"]["ev_plugged_in"]
    assert by_time["00:00"]["ev_plugged_in"]
    assert by_time["05:55"]["ev_plugged_in"]
    assert not by_time["06:00"]["ev_plugged_in"]
    assert {row["ev_departure_min"] for row in first} == {360}
    if closed_loop:
        observed = {row["timestamp"]: row for row in observations}
        assert observed["23:30"]["ev_soc_pct"] == 15
        assert by_time["23:30"]["ev_soc_pct"] == 15.25
        assert observed["00:00"]["ev_soc_pct"] == by_time["23:55"]["ev_soc_pct"]
    else:
        assert by_time["23:30"]["ev_soc_pct"] == 15
    twin.reset()
    assert run() == (first, observations)


def test_custom_reset_and_input_isolation():
    supplied = [EVSegment(0, 60, 15, 80)]
    model = EVModel(EVModelConfig(segments=supplied, seed=42))
    supplied[0].start_soc = 90
    supplied.clear()
    first = [model.tick() for _ in range(20)]
    model.reset()
    assert [model.tick() for _ in range(20)] == first
    model.reset(seed=7)
    assert model.scenario == "custom"
    assert model.soc_pct == 15
    assert model.departure_min == 60


def test_gap_between_custom_segments_resets_soc_on_next_arrival():
    model = EVModel(EVModelConfig(segments=[
        EVSegment(0, 10, 90, 95), EVSegment(20, 30, 15, 80),
    ]))
    model.initialize_at(datetime(2024, 1, 15))
    assert not model.observe(10)["plugged_in"]
    assert model.observe(10)["soc_pct"] == 15


def test_custom_delivery_caps_at_full_soc():
    model = EVModel(EVModelConfig(segments=[EVSegment(0, 60, 99.9, 100)]))
    model.initialize_at(datetime(2024, 1, 15))
    assert model.observe()["soc_pct"] == 99.9
    result = model.commit_delivered(5, 100)
    assert result["soc_pct"] == pytest.approx(100)
    assert result["charging_kw"] == pytest.approx(0.8)


def test_builtin_scenario_enum_is_unchanged():
    assert [scenario.value for scenario in EVScenario] == ["overnight", "midday", "evening_topup"]
