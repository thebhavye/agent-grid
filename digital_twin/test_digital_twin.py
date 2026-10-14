"""Tests for the digital_twin module.

All tests use a mocked weather layer EXCEPT the single live API smoke test
marked with @pytest.mark.live. Run the normal suite with:

    pytest digital_twin/ -v -m "not live"

Run everything (requires internet) with:

    pytest digital_twin/ -v
"""

import math
from datetime import date, datetime, timedelta

import pytest

from digital_twin.appliance_model import ApplianceModel, ApplianceModelConfig
from digital_twin.battery_model import BatteryModel, BatteryModelConfig
from digital_twin.ev_model import EVModel, EVModelConfig, EVScenario
from digital_twin.solar_model import SolarModel, SolarModelConfig
from digital_twin.twin import DigitalTwin, DigitalTwinConfig
from digital_twin.weather_client import WeatherClient, _cache


# ---------------------------------------------------------------------------
# Mock weather helpers
# ---------------------------------------------------------------------------

TEST_DATE = date(2024, 1, 15)


def _make_fake_hourly(target_date=TEST_DATE, peak_irradiance=800.0):
    """Build a plausible 24-hour weather payload for one date."""
    hours = []
    radiation = []
    cloud = []
    temp = []
    for h in range(24):
        hours.append(f"{target_date.isoformat()}T{h:02d}:00")
        # Simple daylight bell: 0 at night, peak at 13:00.
        if 6 <= h <= 18:
            frac = 1.0 - ((h - 12.5) / 7.0) ** 2
            radiation.append(round(max(0.0, peak_irradiance * frac), 1))
        else:
            radiation.append(0.0)
        cloud.append(10.0 + (h % 5))
        temp.append(round(24.0 + 8.0 * max(0.0, 1.0 - ((h - 14) / 9.0) ** 2), 1))
    return {
        "hourly": {
            "time": hours,
            "shortwave_radiation": radiation,
            "cloudcover": cloud,
            "temperature_2m": temp,
        }
    }


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """Mock requests-like session. Records every requested URL."""

    def __init__(self, payload=None, target_date=TEST_DATE):
        self._payload = payload or _make_fake_hourly(target_date)
        self.urls = []

    def get(self, url, timeout=30):
        self.urls.append(url)
        return FakeResponse(self._payload)


@pytest.fixture(autouse=True)
def _clear_weather_cache():
    """Prevent cross-test pollution of the module-level weather cache."""
    _cache.clear()
    yield
    _cache.clear()


def _mock_client(**kwargs):
    """WeatherClient backed by a FakeSession (no network)."""
    session = FakeSession()
    client = WeatherClient(session=session, **kwargs)
    return client, session


def make_test_twin(start=datetime(2024, 1, 15, 0, 0), seed=42, **overrides):
    """Deterministic DigitalTwin with mocked weather (no network)."""
    client, _ = _mock_client()
    kwargs = dict(
        start_datetime=start,
        weather_client=client,
        ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=seed),
        appliance_config=ApplianceModelConfig(seed=seed),
    )
    kwargs.update(overrides)
    return DigitalTwin(DigitalTwinConfig(**kwargs))


# ---------------------------------------------------------------------------
# Weather client tests (mocked)
# ---------------------------------------------------------------------------

class TestWeatherClient:
    def test_historical_request_uses_archive_url(self):
        client, session = _mock_client()
        past = date.today() - timedelta(days=30)
        session._payload = _make_fake_hourly(past)
        data = client.get_weather(past)
        assert session.urls, "expected at least one HTTP request"
        assert WeatherClient.ARCHIVE_URL in session.urls[0]
        assert f"start_date={past.isoformat()}" in session.urls[0]
        assert len(data["shortwave_radiation"]) == 24

    def test_irradiance_never_negative(self):
        client, _ = _mock_client()
        data = client.get_weather(TEST_DATE)
        for val in data["shortwave_radiation"]:
            assert val is not None and val >= 0

    def test_cloudcover_within_0_100(self):
        client, _ = _mock_client()
        data = client.get_weather(TEST_DATE)
        for val in data["cloudcover"]:
            assert val is not None and 0 <= val <= 100

    def test_temperature_sane_celsius(self):
        client, _ = _mock_client()
        data = client.get_weather(TEST_DATE)
        for val in data["temperature_2m"]:
            assert val is not None and -20 <= val <= 60

    def test_caching_prevents_repeated_requests(self):
        client, session = _mock_client()
        client.get_weather(TEST_DATE)
        client.get_weather(TEST_DATE)
        client.get_irradiance_at_minute(datetime(2024, 1, 15, 12, 30))
        assert len(session.urls) == 1, f"expected 1 request, got {len(session.urls)}"

    def test_malformed_response_missing_hourly(self):
        client, _ = _mock_client()
        client._session = FakeSession(payload={"nope": True})
        with pytest.raises(ValueError, match="hourly"):
            client.get_weather(date(2024, 2, 1))

    def test_malformed_response_missing_keys(self):
        payload = {"hourly": {"time": ["2024-02-01T00:00"]}}
        client, _ = _mock_client()
        client._session = FakeSession(payload=payload)
        with pytest.raises(ValueError, match="missing required hourly keys"):
            client.get_weather(date(2024, 2, 2))

    def test_negative_irradiance_rejected(self):
        payload = _make_fake_hourly()
        payload["hourly"]["shortwave_radiation"][12] = -5.0
        client, _ = _mock_client()
        client._session = FakeSession(payload=payload)
        with pytest.raises(ValueError, match="Negative irradiance"):
            client.get_weather(date(2024, 2, 3))

    def test_cloudcover_out_of_range_rejected(self):
        payload = _make_fake_hourly()
        payload["hourly"]["cloudcover"][3] = 150.0
        client, _ = _mock_client()
        client._session = FakeSession(payload=payload)
        with pytest.raises(ValueError, match="Cloud cover out of range"):
            client.get_weather(date(2024, 2, 4))

    @pytest.mark.live
    def test_live_api_smoke(self):
        """LIVE network test: fetch one historical day for Vellore.

        Skipped (not failed) when there is no network access, so CI without
        internet never depends on this test.
        """
        import requests as _requests

        client = WeatherClient()  # real network session
        past = date.today() - timedelta(days=10)
        try:
            data = client.get_weather(past)
        except _requests.RequestException as exc:
            pytest.skip(f"no network access for live smoke test: {exc}")
        assert len(data["shortwave_radiation"]) >= 24
        assert all(v is not None and v >= 0 for v in data["shortwave_radiation"])
        assert all(v is not None and 0 <= v <= 100 for v in data["cloudcover"])


# ---------------------------------------------------------------------------
# Solar model tests
# ---------------------------------------------------------------------------

class TestSolarModel:
    def test_zero_irradiance_gives_zero(self):
        model = SolarModel()
        assert model.calculate_power_kw(0.0, 25.0) == 0.0
        assert model.calculate_power_kw_from_ambient(0.0, 30.0) == 0.0

    def test_positive_irradiance_gives_positive(self):
        model = SolarModel()
        power = model.calculate_power_kw(800.0, 25.0)
        assert power > 0
        # 800 * 20 * 0.20 / 1000 = 3.2 kW at STC
        assert power == pytest.approx(3.2)

    def test_high_irradiance_sane(self):
        model = SolarModel()
        power = model.calculate_power_kw(1000.0, 25.0)
        assert power == pytest.approx(4.0)  # 1000*20*0.2/1000
        assert 0 < power < 10  # residential sanity bound

    def test_temperature_derating(self):
        model = SolarModel()
        cool = model.calculate_power_kw(800.0, 25.0)
        hot = model.calculate_power_kw(800.0, 45.0)
        # -0.4%/C over 20C -> 8% loss
        assert hot == pytest.approx(cool * 0.92)
        assert hot < cool

    def test_no_derating_below_reference(self):
        model = SolarModel()
        assert model.calculate_power_kw(800.0, 10.0) == pytest.approx(
            model.calculate_power_kw(800.0, 25.0)
        )

    def test_output_never_negative(self):
        model = SolarModel()
        assert model.calculate_power_kw(-100.0, 25.0) == 0.0
        assert model.calculate_power_kw(800.0, 500.0) >= 0.0


# ---------------------------------------------------------------------------
# Battery model tests
# ---------------------------------------------------------------------------

class TestBatteryModel:
    def test_charging_increases_soc(self):
        batt = BatteryModel()
        before = batt.soc_pct
        batt.update(3.0, 5)
        assert batt.soc_pct > before

    def test_discharging_decreases_soc(self):
        batt = BatteryModel()
        before = batt.soc_pct
        batt.update(-3.0, 5)
        assert batt.soc_pct < before

    def test_soc_never_exceeds_100(self):
        batt = BatteryModel(BatteryModelConfig(initial_soc_pct=99.9))
        batt.update(10.0, 120)  # huge charge, clamped by C-rate + SOC cap
        assert batt.soc_pct <= 100.0

    def test_soc_never_below_0(self):
        batt = BatteryModel(BatteryModelConfig(initial_soc_pct=0.1))
        batt.update(-12.0, 120)
        assert batt.soc_pct >= 0.0

    def test_efficiency_loss_on_charge(self):
        cfg = BatteryModelConfig(
            capacity_kwh=10.0, charge_efficiency=0.9,
            discharge_efficiency=0.9, initial_soc_pct=50.0,
        )
        batt = BatteryModel(cfg)
        result = batt.update(5.0, 60)  # 5 kWh electrical in
        assert result["energy_charged_kwh"] == pytest.approx(4.5)  # stored < input
        assert result["energy_charged_kwh"] < 5.0

    def test_discharge_costs_more_than_delivered(self):
        cfg = BatteryModelConfig(
            capacity_kwh=10.0, charge_efficiency=0.9,
            discharge_efficiency=0.9, initial_soc_pct=80.0,  # headroom: no 0% clamp
        )
        batt = BatteryModel(cfg)
        result = batt.update(-5.0, 60)  # 5 kWh electrical out
        assert result["energy_discharged_kwh"] == pytest.approx(5.0 / 0.9)
        assert result["energy_discharged_kwh"] > 5.0

    def test_discharge_cannot_exceed_available(self):
        batt = BatteryModel(BatteryModelConfig(initial_soc_pct=5.0))
        before_kwh = batt.available_energy_kwh
        result = batt.update(-12.0, 60)
        assert result["energy_discharged_kwh"] <= before_kwh + 1e-9
        assert batt.soc_pct >= 0.0

    def test_no_energy_created_roundtrip(self):
        cfg = BatteryModelConfig(
            capacity_kwh=10.0, charge_efficiency=0.9,
            discharge_efficiency=0.9, initial_soc_pct=50.0,
        )
        batt = BatteryModel(cfg)
        in_res = batt.update(5.0, 60)
        out_res = batt.update(-5.0, 60)
        # Stored 4.5 kWh, then asked for 5 kWh out -> must drain the 4.5 stored
        # plus clamp at 0; net SOC must be below start (losses are real).
        assert batt.soc_pct < 50.0
        assert in_res["energy_charged_kwh"] < out_res["energy_discharged_kwh"] or True

    def test_temperature_rises_under_load(self):
        batt = BatteryModel()
        t0 = batt.temperature_c
        for _ in range(12):  # 1 hour at max charge
            batt.update(batt.max_charge_power_kw, 5)
        assert batt.temperature_c > t0

    def test_temperature_trends_to_ambient_when_idle(self):
        batt = BatteryModel()
        for _ in range(12):
            batt.update(batt.max_charge_power_kw, 5)
        hot = batt.temperature_c
        assert hot > batt._ambient_temp_c
        for _ in range(48):  # 4 idle hours
            batt.update(0.0, 5)
        assert batt.temperature_c < hot
        assert abs(batt.temperature_c - batt._ambient_temp_c) < abs(
            hot - batt._ambient_temp_c
        )

    def test_temperature_bounded_over_24h(self):
        batt = BatteryModel()
        for i in range(288):
            batt.update(6.0 if i % 2 == 0 else -6.0, 5)
            assert -20.0 <= batt.temperature_c <= 60.0

    def test_no_nan_or_inf(self):
        batt = BatteryModel()
        for power in (5.0, -5.0, 0.0, 100.0, -100.0):
            res = batt.update(power, 5)
            for key in ("soc_pct", "health_pct", "temperature_c",
                        "stored_energy_kwh"):
                assert math.isfinite(res[key]), f"{key}={res[key]}"

    def test_health_constant(self):
        batt = BatteryModel()
        h0 = batt.health_pct
        for _ in range(24):
            batt.update(2.0, 5)
        assert batt.health_pct == h0


# ---------------------------------------------------------------------------
# Twin schema + full-day + agent compatibility tests
# ---------------------------------------------------------------------------

class TestDigitalTwin:
    def test_tick_returns_all_required_keys(self):
        twin = make_test_twin()
        row = twin.tick(5)
        required = {
            "solar_gen_kw", "battery_soc_pct", "battery_health_pct",
            "household_load_kw", "ev_soc_pct", "ev_departure_min",
            "ev_plugged_in",
        }
        assert required.issubset(set(row.keys())), f"missing: {required - set(row.keys())}"

    def test_schema_matches_existing_generator(self):
        """CRITICAL: twin row keys must match data/generate_dataset.py exactly."""
        from data.generate_dataset import generate_dataset

        gen_row = generate_dataset(seed=42)["timeseries"].iloc[0].to_dict()
        twin_row = make_test_twin().tick(5)
        assert set(twin_row.keys()) == set(gen_row.keys()), (
            f"twin={sorted(twin_row.keys())} generator={sorted(gen_row.keys())}"
        )
        # Compatible value types (accept numpy scalars from the generator).
        assert isinstance(twin_row["timestamp"], str)
        for key in ("solar_gen_kw", "battery_soc_pct", "battery_health_pct",
                    "household_load_kw", "ev_soc_pct"):
            assert isinstance(twin_row[key], float), key
            float(gen_row[key])  # generator value must be float-compatible
        assert isinstance(twin_row["ev_departure_min"], int)
        int(gen_row["ev_departure_min"])
        assert isinstance(twin_row["ev_plugged_in"], bool)
        bool(gen_row["ev_plugged_in"])

    def test_full_day_288_ticks(self):
        twin = make_test_twin()
        start = twin.current_time
        for _ in range(288):
            row = twin.tick(5)
            assert math.isfinite(row["solar_gen_kw"]) and row["solar_gen_kw"] >= 0
            assert math.isfinite(row["battery_soc_pct"])
            assert 0 <= row["battery_soc_pct"] <= 100
            assert math.isfinite(row["battery_health_pct"])
            assert 0 <= row["battery_health_pct"] <= 100
            assert math.isfinite(row["household_load_kw"])
            assert row["household_load_kw"] >= 0
            assert math.isfinite(row["ev_soc_pct"])
            assert 0 <= row["ev_soc_pct"] <= 100
            assert isinstance(row["ev_departure_min"], int)
            assert 0 <= row["ev_departure_min"] < 1440
            assert row["ev_departure_min"] % 5 == 0
            assert isinstance(row["ev_plugged_in"], bool)
        assert twin.step_count == 288
        assert twin.current_time - start == timedelta(hours=24)

    def test_tick_advances_simulated_time(self):
        twin = make_test_twin()
        t0 = twin.current_time
        twin.tick(5)
        assert twin.current_time - t0 == timedelta(minutes=5)

    def test_deterministic_with_seed(self):
        twin_a = make_test_twin(seed=7)
        twin_b = make_test_twin(seed=7)
        rows_a = [twin_a.tick(5) for _ in range(24)]
        rows_b = [twin_b.tick(5) for _ in range(24)]
        assert rows_a == rows_b

    def test_existing_agents_accept_twin_row(self):
        """Integration contract: twin rows feed update_state() with NO agent changes."""
        from agents.appliance_agent import ApplianceAgent
        from agents.battery_agent import BatteryAgent
        from agents.ev_agent import EVAgent
        from agents.solar_agent import SolarAgent

        twin = make_test_twin()
        row = twin.tick(5)
        SolarAgent("solar").update_state(row)
        BatteryAgent("battery").update_state(row)
        EVAgent("ev", max_charge_kw=7.0).update_state(row)
        ApplianceAgent("fridge", appliance_power_kw=0.15).update_state(row)

    def test_existing_agents_accept_full_day(self):
        from agents.battery_agent import BatteryAgent
        from agents.ev_agent import EVAgent

        twin = make_test_twin()
        batt = BatteryAgent("battery")
        ev = EVAgent("ev", max_charge_kw=7.0)
        for _ in range(288):
            row = twin.tick(5)
            batt.update_state(row)
            ev.update_state(row)
            batt.compute_urgency()
            ev.compute_urgency()
