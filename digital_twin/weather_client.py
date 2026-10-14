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
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests


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
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self._session = session or requests
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
        today = date.today()
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

        Args:
            target_date: Date to fetch weather for. Defaults to today.
                         Historical dates use the archive API; today/future
                         use the forecast API.

        Returns:
            Dict with keys:
                - "time": list of ISO format hour strings (e.g., "2024-01-15T00:00")
                - "shortwave_radiation": list of W/m² values (float)
                - "cloudcover": list of percentage values (float, 0-100)
                - "temperature_2m": list of Celsius values (float)
                - "date": the target date (date object)

        Raises:
            ValueError: If API response is malformed or validation fails.
            requests.RequestException: If network request fails.
        """
        if target_date is None:
            target_date = date.today()

        # Try cache first
        cached = self._get_cached(target_date)
        if cached is not None:
            return cached

        # Fetch from API
        raw_data = self._fetch_from_api(target_date)

        # Validate data ranges
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
        return result

    def _validate_data(self, hourly: dict) -> None:
        """Validate that weather data is physically plausible.

        Raises:
            ValueError: If any validation check fails.
        """
        # Irradiance should not be negative
        irradiance = hourly.get("shortwave_radiation", [])
        for i, val in enumerate(irradiance):
            if val is not None and val < 0:
                raise ValueError(f"Negative irradiance at index {i}: {val} W/m²")

        # Cloud cover should be 0-100%
        cloudcover = hourly.get("cloudcover", [])
        for i, val in enumerate(cloudcover):
            if val is not None and (val < 0 or val > 100):
                raise ValueError(f"Cloud cover out of range [0,100] at index {i}: {val}%")

        # Temperature should be reasonable for Vellore (roughly -10 to 50°C)
        # This is a sanity check, not a strict physical limit
        temperature = hourly.get("temperature_2m", [])
        for i, val in enumerate(temperature):
            if val is not None and (val < -20 or val > 60):
                raise ValueError(f"Temperature out of reasonable range at index {i}: {val}°C")

    def get_irradiance_at_minute(self, target_datetime: datetime) -> float:
        """Get solar irradiance (W/m²) interpolated to a specific minute.

        Args:
            target_datetime: The exact datetime to get irradiance for.

        Returns:
            Irradiance in W/m² (float, >= 0).
        """
        target_date = target_datetime.date()
        weather = self.get_weather(target_date)

        # Parse hourly times and find surrounding hours
        hourly_times = [datetime.fromisoformat(t) for t in weather["time"]]
        irradiance_values = weather["shortwave_radiation"]

        # Find the two hours that bracket our target time
        target_minute = target_datetime.replace(second=0, microsecond=0)

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
        """Get temperature (°C) interpolated to a specific minute."""
        target_date = target_datetime.date()
        weather = self.get_weather(target_date)

        hourly_times = [datetime.fromisoformat(t) for t in weather["time"]]
        temp_values = weather["temperature_2m"]

        target_minute = target_datetime.replace(second=0, microsecond=0)

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
        """Get cloud cover (%) interpolated to a specific minute."""
        target_date = target_datetime.date()
        weather = self.get_weather(target_date)

        hourly_times = [datetime.fromisoformat(t) for t in weather["time"]]
        cloud_values = weather["cloudcover"]

        target_minute = target_datetime.replace(second=0, microsecond=0)

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