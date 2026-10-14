"""Twin integration, roster validation, and backward compatibility."""

import pandas as pd
import pytest

import simulation.run_simulation as rs
from data.generate_dataset import generate_dataset
from digital_twin import ApplianceModelConfig, EVModelConfig, EVScenario
from digital_twin.appliance_model import ApplianceSpec
from digital_twin.test_digital_twin import (
    _clear_weather_cache as _clear_weather_cache,  # Reuse the autouse fixture too.
    make_test_twin,
)
from digital_twin.weather_client import _cache
from simulation.baseline_allocator import allocate


@pytest.mark.parametrize("ev_limit", [7.0, 11.0])
def test_twin_entry_point_uses_actual_ev_limit_end_to_end(ev_limit):
    ev_config = EVModelConfig(
        scenario=EVScenario.OVERNIGHT, seed=42, max_charge_kw=ev_limit,
    )
    rows = make_test_twin(ev_config=ev_config).run_full_day()
    twin = make_test_twin(ev_config=ev_config)
    demands_seen = []

    def capture(available, demands):
        demands_seen.append(demands)
        return allocate(available, demands)

    results = rs.run_twin_simulation(twin, capture)
    assert len(results) == len(demands_seen) == twin.step_count == 288
    assert any(row["ev_plugged_in"] and row["ev_soc_pct"] < 100 for row in rows)
    for row, demands, allocation in zip(rows, demands_seen, results):
        expected = ev_limit * (100 - row["ev_soc_pct"]) / 100 if row["ev_plugged_in"] else 0
        assert demands["ev_1"] == pytest.approx(expected)
        assert allocation == allocate(row["solar_gen_kw"], demands)


def test_twin_uses_real_per_appliance_agents_without_legacy_fallback(monkeypatch):
    monkeypatch.setattr(rs, "LEGACY_EV_MAX_CHARGE_KW_FALLBACK", 1234.5)
    twin = make_test_twin()
    expected_ids = {"solar_1", "battery_1", "ev_1"} | {
        f"appliance_{spec.appliance_id}"
        for spec in twin._appliances.config.appliances
        if spec.appliance_id != "ev_charger"
    }
    results = rs.run_twin_simulation(twin, allocate)
    assert len(expected_ids) == 10
    assert "appliance_1" not in expected_ids
    assert all(set(allocation) == expected_ids for allocation in results)


def test_twin_entry_point_preserves_custom_roster():
    config = ApplianceModelConfig(appliances=[
        ApplianceSpec("custom_load", 1, False, 0.37),
        ApplianceSpec("ev_charger", 3, True, 2.0),
    ])
    twin = make_test_twin(appliance_config=config)
    demands_seen = []

    def capture(available, demands):
        demands_seen.append(demands)
        return allocate(available, demands)

    rs.run_twin_simulation(twin, capture)
    for demands in demands_seen:
        assert set(demands) == {"solar_1", "battery_1", "ev_1", "appliance_custom_load"}
        assert demands["appliance_custom_load"] == 0.37


@pytest.mark.parametrize("roster", [[], [ApplianceSpec("fridge", 1, False, 0.15)]],
                         ids=["empty", "no-charger-row"])
def test_twin_roster_can_omit_charger_with_authoritative_ev_config(roster):
    twin = make_test_twin(appliance_config=ApplianceModelConfig(appliances=roster))
    results = rs.run_twin_simulation(twin, allocate)
    assert len(results) == 288
    assert set(results[0]) == {"solar_1", "battery_1", "ev_1"} | {
        f"appliance_{spec.appliance_id}" for spec in roster
    }


def test_default_twin_configs_are_resolved_from_models():
    twin = make_test_twin(ev_config=None, appliance_config=None)
    results = rs.run_twin_simulation(twin, allocate)
    assert len(results) == 288
    assert len(results[0]) == 10


def test_weather_fixture_starts_without_cached_payload():
    assert not _cache
    twin = make_test_twin()
    twin.tick()
    assert twin._weather_client._session.urls


@pytest.mark.parametrize("roster_kind", ["dataframe", "records", "config"])
def test_supported_roster_formats(roster_kind):
    config = ApplianceModelConfig()
    frame = rs.get_twin_appliances_df(config)
    roster = {"dataframe": frame, "records": frame.to_dict(orient="records"),
              "config": config}[roster_kind]
    agents = rs._build_agents(roster, ev_max_charge_kw=11.0)
    assert len(agents) == 10
    assert next(a for a in agents if a.agent_id == "ev_1").max_charge_kw == 11.0


def test_generator_path_still_uses_roster_ev_rating():
    tables = generate_dataset(seed=42)
    agents = rs._build_agents(tables["appliances"])
    ev = next(a for a in agents if a.agent_id == "ev_1")
    rating = tables["appliances"].set_index("appliance_id").loc["ev_charger", "appliance_power_kw"]
    assert ev.max_charge_kw == rating
    assert len(rs.run_simulation(tables, allocate)) == 288


def test_legacy_bare_rows_still_use_fallback():
    agents = rs._build_agents(None)
    assert {a.agent_id for a in agents} == {"solar_1", "battery_1", "ev_1", "appliance_1"}
    ev = next(a for a in agents if a.agent_id == "ev_1")
    assert ev.max_charge_kw == rs.LEGACY_EV_MAX_CHARGE_KW_FALLBACK
    assert len(rs.run_simulation([{"solar_gen_kw": 5.0}], allocate)) == 1


@pytest.mark.parametrize("bad_roster", [42, "not-a-roster", {"not": "a-roster"}, [1, 2, 3]],
                         ids=["integer", "string", "mapping", "non-dict-rows"])
@pytest.mark.parametrize("ev_limit", [None, 7.0], ids=["roster-rate", "explicit-rate"])
def test_invalid_non_none_roster_raises_instead_of_fallback(bad_roster, ev_limit):
    with pytest.raises(TypeError, match="roster"):
        rs.run_simulation([{}], allocate, appliances_df=bad_roster, ev_max_charge_kw=ev_limit)


@pytest.mark.parametrize("record", [
    {},
    {"appliance_id": "fridge"},
    {"appliance_id": "", "appliance_power_kw": 0.15},
    {"appliance_id": "fridge", "appliance_power_kw": None},
    {"appliance_id": "fridge", "appliance_power_kw": float("nan")},
    {"appliance_id": "fridge", "appliance_power_kw": float("inf")},
    {"appliance_id": "fridge", "appliance_power_kw": -1.0},
    {"appliance_id": "fridge", "appliance_power_kw": "invalid"},
], ids=["missing-fields", "missing-power", "empty-id", "none-power", "nan-power",
        "infinite-power", "negative-power", "nonnumeric-power"])
@pytest.mark.parametrize("as_frame", [False, True], ids=["records", "dataframe"])
def test_malformed_roster_rejected_even_with_ev_override(record, as_frame):
    roster = pd.DataFrame([record]) if as_frame else [record]
    with pytest.raises(ValueError, match="roster"):
        rs.run_simulation([{}], allocate, appliances_df=roster, ev_max_charge_kw=7.0)


def test_duplicate_appliance_ids_rejected():
    record = {"appliance_id": "fridge", "appliance_power_kw": 0.15}
    with pytest.raises(ValueError, match="duplicate"):
        rs._build_agents([record, record], ev_max_charge_kw=7.0)


@pytest.mark.parametrize("roster", [[], [{"appliance_id": "fridge", "appliance_power_kw": 0.15}]],
                         ids=["empty", "no-charger-row"])
def test_charger_required_without_ev_override(roster):
    with pytest.raises(ValueError, match="ev_charger"):
        rs._build_agents(roster)


@pytest.mark.parametrize("ev_limit", [0, -1, float("nan"), float("inf")])
def test_invalid_ev_override_rejected(ev_limit):
    with pytest.raises(ValueError, match="EV charging rate"):
        rs._build_agents([], ev_max_charge_kw=ev_limit)
