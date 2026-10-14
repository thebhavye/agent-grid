"""Weather client for Vellore, India using Open-Meteo API.

Coordinates: latitude=12.97, longitude=79.16 (Vellore, Tamil Nadu)

Retrieves:
- Shortwave solar radiation / irradiance (W/m²)
- Cloud cover (%)
- Temperature (°C)

Supports:
A. Today's/recent weather data
B. A specific historical date (important for repeatable tests)

Implements caching so repeated requests for the same location/date do not
unnecessarily hit the API.

Design: The network layer can be mocked easily by passing a custom
`session` object with a `get` method (e.g., requests.Session or a test mock).
"""

import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests


logger = logging.getLogger(__name__)


# Module-level cache: key = (latitude, longitude, date_str), value = parsed data
_cache: dict[tuple[float, float, str], dict] = {}


class WeatherClient:
    """Client for fetching weather data from Open-Meteo API.

    API assumptions (based on https://open-meteo.com/en/docs):
    - Base URL: https://archive-api.open-meteo.com/v1/archive (historical)
      or https://api.open-meteo.com/v1/forecast (recent/forecast)
    - No API key required
    - Variables requested:
        - shortwave_radiation (W/m²) - solar irradiance
        - cloudcover (%) - cloud cover percentage
        - temperature_2m (°C) - air temperature at 2m
    - Timezone: Asia/Kolkata (IST, UTC+5:30)
    - Time resolution: hourly (we'll interpolate to 5-min if needed)
    """

    # Vellore, India coordinates
    DEFAULT_LATITUDE = 12.97
    DEFAULT_LONGITUDE = 79.16
    DEFAULT_TIMEZONE = "Asia/Kolkata"

    # Open-Meteo endpoints
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    # Variables to request
    HOURLY_VARIABLES = [
        "shortwave_radiation",
        "cloudcover",
        "temperature_2m",
    ]

    def __init__(
        self,
        latitude: float = DEFAULT_LATITUDE,
        longitude: float = DEFAULT_LONGITUDE,
        timezone: str = DEFAULT_TIMEZONE,
        session: Optional[Any] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            latitude: Latitude in degrees (default: Vellore 12.97)
            longitude: Longitude in degrees (default: Vellore 79.16)
            timezone: IANA timezone string (default: Asia/Kolkata)
            session: Optional requests.Session-like object with a `get` method.
                     If None, uses requests.get directly. Allows mocking in tests.
            cache_dir: Optional directory for persistent file-based caching.
                       If None, only in-memory caching is used.

        Attributes:
            weather_source: How the last get_weather() result was obtained:
                "api" (fresh fetch), "cache" (in-memory/file hit),
                "stale-cache" (older profile remapped after a fetch
                failure), "clear-sky-estimate" (synthetic fallback), or
                "none" (nothing fetched yet).
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self._tz = ZoneInfo(timezone)
        self._session = session or requests
        self.weather_source: str = "none"
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, target_date: date) -> tuple:
        """Generate cache key for a given date."""
        return (self.latitude, self.longitude, target_date.isoformat())

    def _get_cached(self, target_date: date) -> Optional[dict]:
        """Try to retrieve cached data for the given date."""
        key = self._cache_key(target_date)
        # Check in-memory cache first
        if key in _cache:
            return _cache[key]
        # Check file-based cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key[0]}_{key[1]}_{key[2]}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                    _cache[key] = data  # Populate in-memory cache
                    return data
                except (json.JSONDecodeError, OSError):
                    pass
        return None

    def _set_cached(self, target_date: date, data: dict) -> None:
        """Store data in cache."""
        key = self._cache_key(target_date)
        _cache[key] = data
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key[0]}_{key[1]}_{key[2]}.json"
            try:
                with open(cache_file, "w") as f:
                    json.dump(data, f)
            except OSError:
                pass  # Cache write failure is non-fatal

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Normalize a datetime to the client's configured timezone.

        - If dt is timezone-aware, convert it to the configured timezone.
        - If dt is naive, treat it as a wall-clock time in the configured timezone.
        Returns a timezone-aware datetime in the configured timezone.
        """
        if dt.tzinfo is not None:
            # Aware datetime: convert to configured timezone
            return dt.astimezone(self._tz)
        else:
            # Naive datetime: interpret as wall-clock in configured timezone
            return dt.replace(tzinfo=self._tz)

    def _today_in_tz(self) -> date:
        """Return today's date in the configured timezone."""
        return datetime.now(self._tz).date()

    def _build_url(self, target_date: date, is_historical: bool) -> str:
        """Build the API request URL."""
        base_url = self.ARCHIVE_URL if is_historical else self.FORECAST_URL
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "timezone": self.timezone,
        }
        if is_historical:
            params["start_date"] = target_date.isoformat()
            params["end_date"] = target_date.isoformat()
        else:
            # For forecast, we can request today + a few days
            params["forecast_days"] = 3
        return f"{base_url}?{urlencode(params)}"

    def _fetch_from_api(self, target_date: date) -> dict:
        """Fetch weather data from Open-Meteo API."""
        today = self._today_in_tz()
        is_historical = target_date < today

        url = self._build_url(target_date, is_historical)

        response = self._session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Validate response structure
        if "hourly" not in data:
            raise ValueError(f"API response missing 'hourly' data: {data}")

        hourly = data["hourly"]
        required_keys = set(self.HOURLY_VARIABLES) | {"time"}
        missing = required_keys - set(hourly.keys())
        if missing:
            raise ValueError(f"API response missing required hourly keys: {missing}")

        return data

    def get_weather(self, target_date: Optional[date] = None) -> dict:
        """Get weather data for a specific date.

        Fallback chain (a warning is logged whenever a fallback is used):
            1. Cached data for the date (fresh).
            2. Fresh fetch from the API.
            3. Stale cached profile for the same location (remapped onto
               the requested date) if the fetch fails.
            4. Deterministic clear-sky bell-curve estimate if nothing is
               cached. This keeps tick() alive when the API is down; it is
               an estimate, not observed weather.

        Fallback results are cached in-memory under the requested date so
        a dead API is not hammered on every tick; call clear_cache() to
        force a retry.

        Args:
            target_date: Date to fetch weather for. Defaults to today in the
                         configured timezone. Historical dates use the archive
                         API; today/future use the forecast API.

        Returns:
            Dict with keys:
                - "time": list of ISO format hour strings (e.g., "2024-01-15T00:00")
                - "shortwave_radiation": list of W/m² values (float)
                - "cloudcover": list of percentage values (float, 0-100)
                - "temperature_2m": list of Celsius values (float)
                - "date": the target date (date object)

        Raises:
            ValueError: If API response is structurally malformed (missing
                'hourly' data or required keys). Out-of-range values
                (irradiance, cloudcover, temperature) are clamped with a
                warning, never raised. Network failures never raise here;
                they trigger the fallback chain above.
        """
        if target_date is None:
            target_date = self._today_in_tz()

        # Try cache first
        cached = self._get_cached(target_date)
        if cached is not None:
            self.weather_source = "cache"
            return cached

        # Fetch from API, falling back gracefully when the API is down.
        try:
            raw_data = self._fetch_from_api(target_date)
        except requests.RequestException as exc:
            stale = self._get_stale_cached(target_date)
            if stale is not None:
                logger.warning(
                    "Weather API fetch failed for %s (%s); "
                    "falling back to stale cached profile.",
                    target_date, exc,
                )
                self.weather_source = "stale-cache"
                return stale
            logger.warning(
                "Weather API fetch failed for %s (%s); "
                "falling back to deterministic clear-sky estimate.",
                target_date, exc,
            )
            estimated = self._clear_sky_estimate(target_date)
            self._set_cached(target_date, estimated)
            self.weather_source = "clear-sky-estimate"
            return estimated

        # Sanitize data ranges (clamp + warn, never aborts the day)
        self._validate_data(raw_data["hourly"])

        # Package result
        result = {
            "time": raw_data["hourly"]["time"],
            "shortwave_radiation": raw_data["hourly"]["shortwave_radiation"],
            "cloudcover": raw_data["hourly"]["cloudcover"],
            "temperature_2m": raw_data["hourly"]["temperature_2m"],
            "date": target_date,
        }

        # Cache the result
        self._set_cached(target_date, result)
        self.weather_source = "api"
        return result

    def _get_stale_cached(self, target_date: date) -> Optional[dict]:
        """Return any cached profile for this location, remapped onto target_date.

        The hourly times keep their time-of-day but are re-dated to the
        requested date, so interpolation works as usual. The remapped
        profile is also cached under the requested date.
        """
        for key, data in _cache.items():
            if key[0] == self.latitude and key[1] == self.longitude:
                date_prefix = target_date.isoformat()
                remapped = dict(data)
                remapped["time"] = [
                    f"{date_prefix}T{t.split('T', 1)[1]}" for t in data["time"]
                ]
                remapped["date"] = target_date
                self._set_cached(target_date, remapped)
                return remapped
        return None

    def _clear_sky_estimate(self, target_date: date) -> dict:
        """Build a deterministic clear-sky weather estimate for a date.

        Daylight bell peaking at 12:30 (800 W/m²), zero at night, light
        10% cloud, mild 24-32°C temperature curve. No randomness: the same
        date always yields the same estimate. Only used when the API is
        unreachable and nothing is cached.
        """
        times = []
        radiation = []
        cloud = []
        temp = []
        for h in range(24):
            times.append(f"{target_date.isoformat()}T{h:02d}:00")
            if 6 <= h <= 18:
                frac = max(0.0, 1.0 - ((h - 12.5) / 7.0) ** 2)
                radiation.append(round(800.0 * frac, 1))
            else:
                radiation.append(0.0)
            cloud.append(10.0)
            temp.append(round(24.0 + 8.0 * max(0.0, 1.0 - ((h - 14) / 9.0) ** 2), 1))
        return {
            "time": times,
            "shortwave_radiation": radiation,
            "cloudcover": cloud,
            "temperature_2m": temp,
            "date": target_date,
        }

    def _validate_data(self, hourly: dict) -> None:
        """Sanitize weather data in place: clamp + warn per hour.

        Out-of-range values are clamped (never raise, never abort the day):
            - negative irradiance -> 0.0
            - cloudcover clamped to [0, 100]
            - temperature clamped to [-20, 60] (a reading outside that band
              indicates a broken sensor/API payload, so it is clamped and
              flagged loudly rather than trusted)
        """
        # Irradiance must not be negative: clamp to 0.
        irradiance = hourly.get("shortwave_radiation", [])
        for i, val in enumerate(irradiance):
            if val is not None and val < 0:
                logger.warning(
                    "Negative irradiance at index %d: %s W/m²; clamping to 0.",
                    i, val,
                )
                irradiance[i] = 0.0

        # Cloud cover must be 0-100%: clamp both ends.
        cloudcover = hourly.get("cloudcover", [])
        for i, val in enumerate(cloudcover):
            if val is not None and (val < 0 or val > 100):
                logger.warning(
                    "Cloud cover out of range [0,100] at index %d: %s%%; clamping.",
                    i, val,
                )
                cloudcover[i] = max(0.0, min(100.0, val))

        # Temperature sanity band: clamp extreme outliers.
        temperature = hourly.get("temperature_2m", [])
        for i, val in enumerate(temperature):
            if val is not None and (val < -20 or val > 60):
                logger.warning(
                    "Temperature out of range [-20,60] at index %d: %s°C; clamping.",
                    i, val,
                )
                temperature[i] = max(-20.0, min(60.0, val))

    def get_irradiance_at_minute(self, target_datetime: datetime) -> float:
        """Get solar irradiance (W/m²) interpolated to a specific minute.

        Args:
            target_datetime: The exact datetime to get irradiance for.
                             Timezone-aware datetimes are converted to the
                             client's configured timezone. Naive datetimes
                             are treated as wall-clock times in the configured
                             timezone.

        Returns:
            Irradiance in W/m² (float, >= 0).
        """
        normalized = self._normalize_datetime(target_datetime)
        target_date = normalized.date()
        weather = self.get_weather(target_date)

        # Parse hourly times as timezone-aware in the configured timezone
        hourly_times = [
            datetime.fromisoformat(t).replace(tzinfo=self._tz)
            for t in weather["time"]
        ]
        irradiance_values = weather["shortwave_radiation"]

        # Find the two hours that bracket our target time
        target_minute = normalized.replace(second=0, microsecond=0)

        for i in range(len(hourly_times) - 1):
            if hourly_times[i] <= target_minute < hourly_times[i + 1]:
                # Linear interpolation between the two hours
                t0 = hourly_times[i]
                t1 = hourly_times[i + 1]
                v0 = irradiance_values[i] if irradiance_values[i] is not None else 0.0
                v1 = irradiance_values[i + 1] if irradiance_values[i + 1] is not None else 0.0

                # Fraction of hour elapsed (0 to 1)
                frac = (target_minute - t0).total_seconds() / (t1 - t0).total_seconds()
                return max(0.0, v0 + frac * (v1 - v0))

        # If target is exactly on the last hour or beyond, return last value
        if hourly_times and target_minute >= hourly_times[-1]:
            val = irradiance_values[-1]
            return max(0.0, val if val is not None else 0.0)

        # Before first hour (shouldn't happen for same-day queries)
        if hourly_times:
            val = irradiance_values[0]
            return max(0.0, val if val is not None else 0.0)

        return 0.0

    def get_temperature_at_minute(self, target_datetime: datetime) -> float:
        """Get temperature (°C) interpolated to a specific minute.

        Args:
            target_datetime: The exact datetime to get temperature for.
                             Timezone-aware datetimes are converted to the
                             client's configured timezone. Naive datetimes
                             are treated as wall-clock times in the configured
                             timezone.
        """
        normalized = self._normalize_datetime(target_datetime)
        target_date = normalized.date()
        weather = self.get_weather(target_date)

        hourly_times = [
            datetime.fromisoformat(t).replace(tzinfo=self._tz)
            for t in weather["time"]
        ]
        temp_values = weather["temperature_2m"]

        target_minute = normalized.replace(second=0, microsecond=0)

        for i in range(len(hourly_times) - 1):
            if hourly_times[i] <= target_minute < hourly_times[i + 1]:
                t0 = hourly_times[i]
                t1 = hourly_times[i + 1]
                v0 = temp_values[i] if temp_values[i] is not None else 25.0
                v1 = temp_values[i + 1] if temp_values[i + 1] is not None else 25.0
                frac = (target_minute - t0).total_seconds() / (t1 - t0).total_seconds()
                return v0 + frac * (v1 - v0)

        if hourly_times and target_minute >= hourly_times[-1]:
            val = temp_values[-1]
            return val if val is not None else 25.0

        if hourly_times:
            val = temp_values[0]
            return val if val is not None else 25.0

        return 25.0

    def get_cloudcover_at_minute(self, target_datetime: datetime) -> float:
        """Get cloud cover (%) interpolated to a specific minute.

        Args:
            target_datetime: The exact datetime to get cloud cover for.
                             Timezone-aware datetimes are converted to the
                             client's configured timezone. Naive datetimes
                             are treated as wall-clock times in the configured
                             timezone.
        """
        normalized = self._normalize_datetime(target_datetime)
        target_date = normalized.date()
        weather = self.get_weather(target_date)

        hourly_times = [
            datetime.fromisoformat(t).replace(tzinfo=self._tz)
            for t in weather["time"]
        ]
        cloud_values = weather["cloudcover"]

        target_minute = normalized.replace(second=0, microsecond=0)

        for i in range(len(hourly_times) - 1):
            if hourly_times[i] <= target_minute < hourly_times[i + 1]:
                t0 = hourly_times[i]
                t1 = hourly_times[i + 1]
                v0 = cloud_values[i] if cloud_values[i] is not None else 0.0
                v1 = cloud_values[i + 1] if cloud_values[i + 1] is not None else 0.0
                frac = (target_minute - t0).total_seconds() / (t1 - t0).total_seconds()
                return max(0.0, min(100.0, v0 + frac * (v1 - v0)))

        if hourly_times and target_minute >= hourly_times[-1]:
            val = cloud_values[-1]
            return max(0.0, min(100.0, val if val is not None else 0.0))

        if hourly_times:
            val = cloud_values[0]
            return max(0.0, min(100.0, val if val is not None else 0.0))

        return 0.0

    def clear_cache(self) -> None:
        """Clear the in-memory cache. File cache is not affected."""
        global _cache
        _cache.clear()


# Convenience function for simple usage
def get_weather_for_date(target_date: date, latitude: float = WeatherClient.DEFAULT_LATITUDE,
                         longitude: float = WeatherClient.DEFAULT_LONGITUDE) -> dict:
    """Convenience function to get weather for a date with default Vellore location."""
    client = WeatherClient(latitude=latitude, longitude=longitude)
    return client.get_weather(target_date)
