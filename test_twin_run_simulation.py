"""Twin-to-simulation integration (issue #25).

Twin rows run through run_simulation with real per-appliance agents and
the twin's authoritative EV rate (EVModelConfig.max_charge_kw), never the
LEGACY fallback.
"""

from datetime import datetime
from unittest.mock import patch

import simulation.run_simulation as rs
from data.generate_dataset import generate_dataset
from digital_twin import (
    ApplianceModelConfig,
    DigitalTwin,
    DigitalTwinConfig,
    EVModelConfig,
    EVScenario,
)
from digital_twin.weather_client import WeatherClient
from simulation.baseline_allocator import allocate
from simulation.run_simulation import (
    _build_agents,
    get_twin_appliances_df,
    get_twin_ev_max_charge_kw,
    run_simulation,
)


def _fake_weather_client():
    """Offline weather: one plausible day, no network."""

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, timeout=30):
            hours = [f"2024-01-15T{h:02d}:00" for h in range(24)]
            radiation = [
                round(max(0.0, 800.0 * (1.0 - ((h - 12.5) / 7.0) ** 2)), 1)
                if 6 <= h <= 18
                else 0.0
                for h in range(24)
            ]
            return FakeResponse(
                {
                    "hourly": {
                        "time": hours,
                        "shortwave_radiation": radiation,
                        "cloudcover": [10.0] * 24,
                        "temperature_2m": [28.0] * 24,
                    }
                }
            )

    return WeatherClient(session=FakeSession())


def _make_twin_and_rows(seed=42, ev_max_charge_kw=None):
    ev_config = EVModelConfig(
        scenario=EVScenario.OVERNIGHT,
        seed=seed,
        max_charge_kw=(
            ev_max_charge_kw
            if ev_max_charge_kw is not None
            else EVModelConfig().max_charge_kw
        ),
    )
    twin = DigitalTwin(
        DigitalTwinConfig(
            start_datetime=datetime(2024, 1, 15, 0, 0),
            weather_client=_fake_weather_client(),
            ev_config=ev_config,
            appliance_config=ApplianceModelConfig(seed=seed),
        )
    )
    rows = twin.run_full_day()
    appliances_df = get_twin_appliances_df(twin._appliances.config)
    ev_max = get_twin_ev_max_charge_kw(twin._ev.config)
    return twin, rows, appliances_df, ev_max


def _run_twin(rows, appliances_df, ev_max):
    return run_simulation(
        rows, allocate, appliances_df=appliances_df, ev_max_charge_kw=ev_max
    )


def test_twin_288_rows_complete_end_to_end():
    _, rows, appliances_df, ev_max = _make_twin_and_rows()
    assert len(rows) == 288
    results = _run_twin(rows, appliances_df, ev_max)
    assert len(results) == 288


def test_twin_uses_real_per_appliance_agents():
    _, rows, appliances_df, ev_max = _make_twin_and_rows()
    agents = _build_agents(appliances_df, ev_max_charge_kw=ev_max)
    ids = {a.agent_id for a in agents}
    assert "appliance_1" not in ids
    assert {"solar_1", "battery_1", "ev_1"}.issubset(ids)
    # 8-row twin roster minus ev_charger (served by ev_1) + 3 system agents.
    assert len(ids) == 10
    assert "appliance_fridge" in ids
    assert "appliance_dishwasher" in ids
    results = _run_twin(rows, appliances_df, ev_max)
    for allocation in results:
        assert set(allocation.keys()) == ids


def test_twin_ev_rate_matches_ev_model_config():
    twin, _, appliances_df, ev_max = _make_twin_and_rows()
    # Authoritative source is EVModelConfig (clamps EVModel.tick charging),
    # not the roster display rating (which is excluded from household load).
    assert ev_max == twin._ev.config.max_charge_kw
    assert ev_max == get_twin_ev_max_charge_kw(twin._ev.config)
    roster_power = next(
        s.power_kw
        for s in twin._appliances.config.appliances
        if s.appliance_id == "ev_charger"
    )
    assert ev_max != roster_power
    agents = _build_agents(appliances_df, ev_max_charge_kw=ev_max)
    ev = next(a for a in agents if a.agent_id == "ev_1")
    assert ev.max_charge_kw == ev_max


def test_twin_custom_ev_rate_preserved():
    _, rows, appliances_df, ev_max = _make_twin_and_rows(ev_max_charge_kw=11.0)
    assert ev_max == 11.0
    agents = _build_agents(appliances_df, ev_max_charge_kw=ev_max)
    ev = next(a for a in agents if a.agent_id == "ev_1")
    assert ev.max_charge_kw == 11.0
    results = _run_twin(rows, appliances_df, ev_max)
    assert len(results) == 288


def test_legacy_fallback_not_used_when_roster_present():
    _, rows, appliances_df, ev_max = _make_twin_and_rows()
    with patch.object(rs, "LEGACY_EV_MAX_CHARGE_KW_FALLBACK", 1234.5):
        agents = _build_agents(appliances_df, ev_max_charge_kw=ev_max)
        ev = next(a for a in agents if a.agent_id == "ev_1")
        assert ev.max_charge_kw != 1234.5
        results = _run_twin(rows, appliances_df, ev_max)
    assert "appliance_1" not in results[0]


def test_generator_path_still_compatible():
    tables = generate_dataset(seed=42)
    results = run_simulation(tables, allocate)
    assert len(results) == 288


def test_legacy_bare_rows_still_use_fallback():
    agents = _build_agents(None)
    ids = {a.agent_id for a in agents}
    assert ids == {"solar_1", "battery_1", "ev_1", "appliance_1"}
    ev = next(a for a in agents if a.agent_id == "ev_1")
    assert ev.max_charge_kw == rs.LEGACY_EV_MAX_CHARGE_KW_FALLBACK


def test_invalid_non_none_roster_raises_instead_of_fallback():
    import pytest

    for bad_roster in (42, "not-a-roster", {"not": "a-roster"}, [1, 2, 3]):
        try:
            _build_agents(bad_roster)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"invalid roster {bad_roster!r} silently fell back "
                "instead of raising"
            )
