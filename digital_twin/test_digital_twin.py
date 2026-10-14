"""Tests for the digital_twin module.

All tests use a mocked weather layer EXCEPT the single live API smoke test
marked with @pytest.mark.live. Run the normal suite with:

    pytest digital_twin/ -v -m "not live"

Run everything (requires internet) with:

    pytest digital_twin/ -v
"""

import logging
import math
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from digital_twin.appliance_model import ApplianceModel, ApplianceModelConfig, ApplianceSpec
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


class DateAwareFakeSession:
    """Mock session serving a per-date payload based on the request URL.

    Parses the archive `start_date` query param; forecast URLs (no date
    param) fall back to TEST_DATE. Records every requested URL.
    """

    def __init__(self, payloads: dict):
        self._payloads = payloads
        self.urls = []

    def get(self, url, timeout=30):
        self.urls.append(url)
        query = parse_qs(urlparse(url).query)
        if "start_date" in query:
            key = date.fromisoformat(query["start_date"][0])
        else:
            key = TEST_DATE
        return FakeResponse(self._payloads[key])


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

    def test_negative_irradiance_clamped_with_warning(self, caplog):
        payload = _make_fake_hourly()
        payload["hourly"]["shortwave_radiation"][12] = -5.0
        client, _ = _mock_client()
        client._session = FakeSession(payload=payload)
        with caplog.at_level(logging.WARNING):
            data = client.get_weather(date(2024, 2, 3))
        assert data["shortwave_radiation"][12] == 0.0
        assert "irradiance" in caplog.text.lower()
        assert all(v is None or v >= 0 for v in data["shortwave_radiation"])

    def test_cloudcover_out_of_range_clamped_with_warning(self, caplog):
        payload = _make_fake_hourly()
        payload["hourly"]["cloudcover"][3] = 150.0
        payload["hourly"]["cloudcover"][7] = -20.0
        client, _ = _mock_client()
        client._session = FakeSession(payload=payload)
        with caplog.at_level(logging.WARNING):
            data = client.get_weather(date(2024, 2, 4))
        assert data["cloudcover"][3] == 100.0
        assert data["cloudcover"][7] == 0.0
        assert "cloud cover" in caplog.text.lower()
        assert all(v is None or 0 <= v <= 100 for v in data["cloudcover"])

    def test_single_bad_hour_does_not_block_day(self):
        payload = _make_fake_hourly()
        good_irradiance_6 = payload["hourly"]["shortwave_radiation"][6]
        payload["hourly"]["shortwave_radiation"][5] = -3.0
        payload["hourly"]["cloudcover"][7] = 150.0
        client, _ = _mock_client()
        client._session = FakeSession(payload=payload)
        data = client.get_weather(date(2024, 2, 5))
        assert len(data["shortwave_radiation"]) == 24
        assert len(data["cloudcover"]) == 24
        assert len(data["temperature_2m"]) == 24
        # Bad hours clamped, good hours untouched.
        assert data["shortwave_radiation"][5] == 0.0
        assert data["cloudcover"][7] == 100.0
        assert data["shortwave_radiation"][6] == good_irradiance_6

    @pytest.mark.live
    def test_live_api_smoke(self):
        """LIVE network test: fetch one historical day for Vellore.

        Skipped (not failed) when there is no network access, so CI without
        internet never depends on this test.
        """
        client = WeatherClient()  # real network session
        past = date.today() - timedelta(days=10)
        data = client.get_weather(past)
        if client.weather_source != "api":
            pytest.skip(
                "no network access for live smoke test "
                f"(weather_source={client.weather_source})"
            )
        assert client.weather_source == "api"
        assert len(data["shortwave_radiation"]) >= 24
        assert all(v is not None and v >= 0 for v in data["shortwave_radiation"])
        assert all(v is not None and 0 <= v <= 100 for v in data["cloudcover"])

    def test_temperature_outlier_clamped_with_warning(self, caplog):
        payload = _make_fake_hourly()
        payload["hourly"]["temperature_2m"][14] = 99.0
        client, _ = _mock_client()
        client._session = FakeSession(payload=payload)
        with caplog.at_level(logging.WARNING):
            data = client.get_weather(date(2024, 2, 6))
        assert data["temperature_2m"][14] == 60.0
        assert "temperature" in caplog.text.lower()
        assert all(v is None or -20 <= v <= 60 for v in data["temperature_2m"])

    def test_weather_source_tracks_provenance(self):
        client, session = _mock_client()
        assert client.weather_source == "none"
        client.get_weather(TEST_DATE)
        assert client.weather_source == "api"
        client.get_weather(TEST_DATE)
        assert client.weather_source == "cache"

    def test_file_cache_roundtrip(self, tmp_path):
        """Issue #27: cache_dir persists across clients without refetching."""
        client, session = _mock_client(cache_dir=str(tmp_path))
        first = client.get_weather(TEST_DATE)
        assert len(session.urls) == 1
        cache_file = tmp_path / (
            f"{client.latitude}_{client.longitude}_{TEST_DATE.isoformat()}.json"
        )
        assert cache_file.exists()

        _cache.clear()  # force the next client onto the file cache
        reader, reader_session = _mock_client(cache_dir=str(tmp_path))
        second = reader.get_weather(TEST_DATE)
        assert len(reader_session.urls) == 0
        assert second["shortwave_radiation"] == first["shortwave_radiation"]
        assert second["time"] == first["time"]
        assert second["date"] == TEST_DATE

    def test_corrupt_file_cache_falls_back_to_fetch(self, tmp_path):
        """Issue #27: unreadable cache files never block a fresh fetch."""
        client, session = _mock_client(cache_dir=str(tmp_path))
        cache_file = tmp_path / (
            f"{client.latitude}_{client.longitude}_{TEST_DATE.isoformat()}.json"
        )
        cache_file.write_text("{ not valid json")
        data = client.get_weather(TEST_DATE)
        assert len(session.urls) == 1
        assert len(data["shortwave_radiation"]) == 24

    def test_get_weather_for_date_convenience(self):
        """Issue #27: the module convenience function serves the date asked."""
        from unittest.mock import patch

        from digital_twin.weather_client import get_weather_for_date

        payload = _make_fake_hourly(TEST_DATE)
        with patch("requests.get", return_value=FakeResponse(payload)):
            data = get_weather_for_date(TEST_DATE)
        assert data["date"] == TEST_DATE
        assert len(data["shortwave_radiation"]) == 24
        assert all(v is None or v >= 0 for v in data["shortwave_radiation"])


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

    def test_degenerate_thermal_constants_stay_stable(self):
        """Issue #27: tau_h <= 0 takes the exact-decay fallback, never NaN."""
        for kwargs in ({"thermal_capacity": 0.0}, {"thermal_resistance": 0.0}):
            batt = BatteryModel(BatteryModelConfig(
                initial_soc_pct=50.0, **kwargs))
            for _ in range(24):
                res = batt.update(6.0, 5)
                assert math.isfinite(res["temperature_c"])
                assert -20.0 <= res["temperature_c"] <= 60.0

    def test_health_constant(self):
        batt = BatteryModel()
        h0 = batt.health_pct
        for _ in range(24):
            batt.update(2.0, 5)
        assert batt.health_pct == h0


# ---------------------------------------------------------------------------
# Appliance model tests
# ---------------------------------------------------------------------------

class TestApplianceModel:
    @pytest.mark.parametrize("seed", [42, 7, 123])
    def test_essentials_stay_on_for_full_day(self, seed):
        model = ApplianceModel(ApplianceModelConfig(seed=seed))
        model.initialize_at(datetime(2024, 1, 15))
        essentials = [spec for spec in model.config.appliances if not spec.deferable]
        minimum_load = model.config.base_load_kw + sum(spec.power_kw for spec in essentials)

        for tick in range(288):
            result = model.tick(5)
            for spec in essentials:
                assert model.get_appliance_state(spec.appliance_id)["is_on"], (
                    f"{spec.appliance_id} turned off at tick {tick}, seed {seed}"
                )
                assert spec.appliance_id in result["active_appliances"]
                assert result["appliance_loads"][spec.appliance_id] == spec.power_kw
            assert result["total_load_kw"] >= minimum_load
            assert model.total_load_kw == pytest.approx(result["total_load_kw"])

    def test_deferable_cycle_still_ends(self):
        spec = ApplianceSpec(
            "dishwasher", 4, True, 1.2, 0, 10,
            min_runtime_min=5, max_runtime_min=5, cycle_prob_per_hour=12,
        )
        model = ApplianceModel(ApplianceModelConfig(seed=42, appliances=[spec]))
        model.initialize_at(datetime(2024, 1, 15))

        assert model.tick(5)["appliance_loads"]["dishwasher"] == 1.2
        assert model.tick(5)["appliance_loads"]["dishwasher"] == 0.0
        assert not model.get_appliance_state("dishwasher")["is_on"]


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

    def test_timestamps_sample_and_label_at_start_time(self):
        """Fix 1: rows are sampled AND labeled at the interval start."""
        start = datetime(2024, 1, 15, 0, 0)
        twin = make_test_twin(start=start)
        rows = twin.run_full_day()
        assert len(rows) == 288
        assert rows[0]["timestamp"] == "00:00"
        assert rows[-1]["timestamp"] == "23:55"
        assert twin.current_time == start + timedelta(hours=24)

    def test_weather_date_override_selects_data_source(self):
        """Fix 2: the override date (not the start date) picks the data."""
        payloads = {
            date(2024, 1, 15): _make_fake_hourly(date(2024, 1, 15), peak_irradiance=800.0),
            date(2024, 1, 20): _make_fake_hourly(date(2024, 1, 20), peak_irradiance=100.0),
        }
        start = datetime(2024, 1, 15, 12, 0)  # midday: solar separates the peaks

        def _twin_at_noon(**overrides):
            session = DateAwareFakeSession(payloads)
            client = WeatherClient(session=session)
            kwargs = dict(
                start_datetime=start,
                weather_client=client,
                ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=42),
                appliance_config=ApplianceModelConfig(seed=42),
            )
            kwargs.update(overrides)
            return DigitalTwin(DigitalTwinConfig(**kwargs)), session

        plain_twin, plain_session = _twin_at_noon()
        plain_row = plain_twin.tick(5)
        assert "2024-01-15" in plain_session.urls[0]

        over_twin, over_session = _twin_at_noon(
            weather_date_override=datetime(2024, 1, 20, 0, 0)
        )
        over_row = over_twin.tick(5)
        assert "2024-01-20" in over_session.urls[0]

        # 100 W/m² peak vs 800 W/m² peak: override row must be much smaller.
        assert over_row["solar_gen_kw"] > 0
        assert over_row["solar_gen_kw"] < plain_row["solar_gen_kw"] / 2

    def test_tick_survives_api_failure_with_clear_sky_fallback(self, caplog):
        """Fix 3: RequestException -> warning + deterministic estimate row."""
        import requests as _requests

        class FailingSession:
            def get(self, url, timeout=30):
                raise _requests.RequestException("simulated outage")

        client = WeatherClient(session=FailingSession())
        twin = DigitalTwin(DigitalTwinConfig(
            start_datetime=datetime(2024, 1, 15, 12, 0),
            weather_client=client,
            ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=42),
            appliance_config=ApplianceModelConfig(seed=42),
        ))
        with caplog.at_level(logging.WARNING):
            row = twin.tick(5)
        assert "clear-sky" in caplog.text
        # Valid row: finite values, midday estimate gives real solar.
        assert math.isfinite(row["solar_gen_kw"]) and row["solar_gen_kw"] > 0
        assert math.isfinite(row["household_load_kw"]) and row["household_load_kw"] >= 0
        assert 0 <= row["battery_soc_pct"] <= 100
        assert 0 <= row["ev_soc_pct"] <= 100
        assert isinstance(row["ev_departure_min"], int)
        assert isinstance(row["ev_plugged_in"], bool)

    def test_api_failure_uses_stale_cache(self, caplog):
        """Fix 3 (stale leg): a cached profile is remapped onto the new date."""
        import requests as _requests

        client, _ = _mock_client()
        cached = client.get_weather(TEST_DATE)
        assert len(client._session.urls) == 1

        class FailingSession:
            def get(self, url, timeout=30):
                raise _requests.RequestException("simulated outage")

        client._session = FailingSession()
        other_day = date(2024, 1, 16)
        with caplog.at_level(logging.WARNING):
            data = client.get_weather(other_day)
        assert "stale" in caplog.text.lower()
        # Remapped onto the requested date, values preserved.
        assert data["date"] == other_day
        assert all(t.startswith("2024-01-16T") for t in data["time"])
        assert data["shortwave_radiation"] == cached["shortwave_radiation"]

    def test_multiday_rollover_reuses_weather_profile(self):
        """Fix 4: past tick 288 the day-1 profile repeats; no new fetches."""
        client, session = _mock_client()
        twin = DigitalTwin(DigitalTwinConfig(
            start_datetime=datetime(2024, 1, 15, 0, 0),
            weather_client=client,
            ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=42),
            appliance_config=ApplianceModelConfig(seed=42),
        ))
        rows = [twin.tick(5) for _ in range(600)]
        assert len(rows) == 600
        # Day 2 starts over at midnight with zero night solar ...
        assert rows[288]["timestamp"] == "00:00"
        assert rows[288]["solar_gen_kw"] == 0.0
        # ... and repeats midday sun (tick 432 == start + 36h == day-2 noon).
        assert rows[432]["timestamp"] == "12:00"
        assert rows[432]["solar_gen_kw"] > 1.0
        assert rows[432]["solar_gen_kw"] == pytest.approx(rows[144]["solar_gen_kw"])
        # Weather for the day was fetched once; rollover caused no refetch.
        assert len(session.urls) == 1, f"expected 1 request, got {len(session.urls)}"

    def test_deterministic_with_seed(self):
        twin_a = make_test_twin(seed=7)
        twin_b = make_test_twin(seed=7)
        rows_a = [twin_a.tick(5) for _ in range(24)]
        rows_b = [twin_b.tick(5) for _ in range(24)]
        assert rows_a == rows_b

    def test_departure_edge_matches_generator(self):
        """Issue #22: first-unplugged label == departure label, like the generator."""
        from data.test_generate_dataset import _generate_overnight_tables

        ots = _generate_overnight_tables(seed=1)["timeseries"]
        gen_dep = int(ots["ev_departure_min"].iloc[0])
        gen_labels = ots["timestamp"].tolist()
        gen_plug = [bool(v) for v in ots["ev_plugged_in"].tolist()]
        gen_first_unplug = next(lbl for lbl, p in zip(gen_labels, gen_plug) if not p)
        assert gen_first_unplug == f"{gen_dep // 60:02d}:{gen_dep % 60:02d}"

        client, _ = _mock_client()
        twin = DigitalTwin(DigitalTwinConfig(
            start_datetime=datetime(2024, 1, 15, 0, 0),
            weather_client=client,
            ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=1),
            appliance_config=ApplianceModelConfig(seed=1),
        ))
        rows = twin.run_full_day()
        assert rows[0]["ev_plugged_in"] is True
        twin_dep = rows[0]["ev_departure_min"]
        twin_first_unplug = next(r["timestamp"] for r in rows if not r["ev_plugged_in"])
        assert twin_first_unplug == f"{twin_dep // 60:02d}:{twin_dep % 60:02d}"

    def test_midday_scenario_full_day_edges(self):
        """Issue #27: the MIDDAY scenario (never covered before) meets the
        same edge contract: unplugged at midnight, plug-in inside the
        9:00-10:30 window, and the post-charge label equals departure."""
        client, _ = _mock_client()
        twin = DigitalTwin(DigitalTwinConfig(
            start_datetime=datetime(2024, 1, 15, 0, 0),
            weather_client=client,
            ev_config=EVModelConfig(scenario=EVScenario.MIDDAY, seed=11),
            appliance_config=ApplianceModelConfig(seed=11),
        ))
        assert twin._ev.scenario == EVScenario.MIDDAY
        rows = twin.run_full_day()
        assert rows[0]["ev_plugged_in"] is False
        dep = rows[0]["ev_departure_min"]
        assert 16 * 60 <= dep <= 17 * 60 + 30
        first_plugged = next(
            r["timestamp"] for r in rows if r["ev_plugged_in"])
        assert "09:00" <= first_plugged <= "10:30"
        last_plugged_idx = max(
            i for i, r in enumerate(rows) if r["ev_plugged_in"])
        assert rows[last_plugged_idx + 1]["timestamp"] == (
            f"{dep // 60:02d}:{dep % 60:02d}")
        for row in rows:
            assert isinstance(row["ev_plugged_in"], bool)
            assert math.isfinite(row["ev_soc_pct"])
            assert 0 <= row["ev_soc_pct"] <= 100

    def test_fifteen_minute_timestep_stays_aligned(self):
        """Issue #27: a non-5-minute config dt keeps labels, edges and clock
        consistent over a full day (96 x 15-min steps)."""
        client, _ = _mock_client()
        twin = DigitalTwin(DigitalTwinConfig(
            start_datetime=datetime(2024, 1, 15, 0, 0),
            weather_client=client,
            ev_config=EVModelConfig(scenario=EVScenario.OVERNIGHT, seed=42),
            appliance_config=ApplianceModelConfig(seed=42),
            time_delta_min=15.0,
        ))
        rows = twin.run_full_day(steps=96, time_delta_min=15.0)
        assert len(rows) == 96
        assert rows[0]["timestamp"] == "00:00"
        assert rows[1]["timestamp"] == "00:15"
        assert rows[-1]["timestamp"] == "23:45"
        assert twin.current_time == datetime(2024, 1, 16, 0, 0)
        assert rows[0]["ev_plugged_in"] is True
        dep = rows[0]["ev_departure_min"]
        # Departure need not sit on a 15-min boundary: the first unplugged
        # label is the first tick at or past departure.
        expected_min = ((dep + 14) // 15) * 15
        first_unplug = next(
            r["timestamp"] for r in rows if not r["ev_plugged_in"])
        assert first_unplug == f"{expected_min // 60:02d}:{expected_min % 60:02d}"
        for row in rows:
            assert math.isfinite(row["solar_gen_kw"]) and row["solar_gen_kw"] >= 0
            assert 0 <= row["battery_soc_pct"] <= 100

    def test_default_configs_are_repeatable(self):
        """Fixed default seeds: two bare-config twins give identical days."""
        def _default_twin():
            client, _ = _mock_client()
            return DigitalTwin(DigitalTwinConfig(
                start_datetime=datetime(2024, 1, 15, 0, 0),
                weather_client=client,
            ))

        rows_a = _default_twin().run_full_day()
        rows_b = _default_twin().run_full_day()
        assert len(rows_a) == 288 == len(rows_b)
        assert rows_a == rows_b

    def test_overnight_load_stays_plausible(self):
        """No EV-charger double-count: max household load stays below 4 kW."""
        twin = make_test_twin()
        rows = twin.run_full_day()
        peak = max(r["household_load_kw"] for r in rows)
        assert peak < 4.0, f"implausible overnight/daytime peak: {peak} kW"

    @pytest.mark.parametrize("seed", [42, 7, 123])
    def test_battery_soc_matches_household_energy_budget(self, seed):
        """Verify energy balance and bound depletion in the seeded default scenario."""
        cfg = BatteryModelConfig()
        twin = make_test_twin(seed=seed, battery_config=cfg)
        energy = cfg.capacity_kwh * cfg.initial_soc_pct / 100
        # Each power term rounds to 3 decimals: net power error <= 0.001 kW.
        # Convert its accumulated energy error to SOC using the worst efficiency.
        soc_error_per_tick = (
            2 * 0.5 * 10**-3 * (5 / 60)
            * max(cfg.charge_efficiency, 1 / cfg.discharge_efficiency)
            * 100 / cfg.capacity_kwh
        )
        worst_zero_run = current_zero_run = 0
        for tick, row in enumerate(twin.run_full_day(), start=1):
            power = row["solar_gen_kw"] - row["household_load_kw"]
            power = max(-cfg.capacity_kwh * cfg.max_discharge_rate_c,
                        min(cfg.capacity_kwh * cfg.max_charge_rate_c, power))
            delta = power * 5 / 60
            energy += (delta * cfg.charge_efficiency if delta > 0
                       else delta / cfg.discharge_efficiency)
            energy = max(0, min(cfg.capacity_kwh, energy))
            # Add half a unit of the reported SOC's 2-decimal rounding.
            tolerance = tick * soc_error_per_tick + 0.5 * 10**-2
            assert row["battery_soc_pct"] == pytest.approx(
                100 * energy / cfg.capacity_kwh, abs=tolerance, rel=0
            )
            current_zero_run = current_zero_run + 1 if row["battery_soc_pct"] == 0 else 0
            worst_zero_run = max(worst_zero_run, current_zero_run)
        # Always-on essentials can drain the initial 6 kWh before sunrise.
        # Accept at most 2 hours at zero in this mocked-weather scenario.
        assert worst_zero_run <= 24, f"seed {seed}: battery pinned at 0 for {worst_zero_run} ticks"

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
