# Offline twin scenarios

`list_scenarios()` returns the seven names in alphabetical order.
`build_twin(name, closed_loop=False)` returns a fresh `DigitalTwin`; an unknown
name raises `ValueError` listing the available choices. `CATALOG[name]` holds
the weather filename, start date, EV scenario/seed, appliance seed, battery
initial SOC/capacity, timestep and step count.

| Scenario | Weather source date | Steps | What it stresses |
| --- | --- | ---: | --- |
| `clear_day` | 2024-01-25 | 288 | Sunny-midday baseline; this real day also has morning/evening cloud |
| `monsoon_cloudy` | 2023-12-04 | 288 | Overcast monsoon profile with weak, variable irradiance |
| `heatwave` | 2024-05-01 | 288 | Hot pre-monsoon air temperatures and PV temperature derating |
| `low_battery_start` | 2024-01-25 | 288 | Battery starts at the 20% discharge reserve |
| `degraded_battery` | 2024-01-25 | 288 | 6 kWh capacity instead of 12 kWh |
| `overnight_ev_cloudy` | 2023-12-04 | 288 | Overnight EV charging on a low-solar day |
| `three_days` | 2024-01-25, repeated | 864 | Three simulated days with one repeated weather profile |

All scenarios use five-minute steps, EV seed 42 and appliance seed 42.
Unless listed otherwise, battery initial SOC is 50%, capacity is 12 kWh,
and the EV scenario is `midday`. `overnight_ev_cloudy` uses `overnight`.
Every build creates separate model state and RNGs.

The degraded scenario is an explicit capacity proxy, not a battery aging or
state-of-health model. Health remains the twin default; changing capacity
also changes the kW limits derived by the existing model from its C-rates.

## Saved weather and attribution

Weather data by [Open-Meteo](https://open-meteo.com/), retrieved on 2026-10-16
from its [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
for Vellore (requested latitude 12.97, longitude 79.16), timezone Asia/Kolkata.
These are provider-supplied **historical reanalysis** profiles, not direct
local weather-station observations. The default Best Match archive combines
underlying model datasets; no particular constituent model is claimed.
Returned grid coordinates (12.970123, 79.2), hourly units, request URL, date and
retrieval provenance are retained in each file.

| Saved file | Source date | Peak radiation (W/m2) | Maximum air temperature (C) |
| --- | --- | ---: | ---: |
| `weather/vellore_2024-01-25.json` | 2024-01-25 | 833 | 31.3 |
| `weather/vellore_2023-12-04.json` | 2023-12-04 | 104 | 24.5 |
| `weather/vellore_2024-05-01.json` | 2024-05-01 | 978 | 42.9 |

The hourly arrays are saved unchanged from the API response. Only JSON
formatting and an `_provenance` metadata field were added. No synthetic
weather is included. `hourly` contains `time`, `shortwave_radiation`,
`cloudcover` and `temperature_2m`, matching the existing WeatherClient.
The files can be reproduced using the exact archive URLs in `_provenance.url`;
normal builds/tests never fetch them again.

API data attribution: Open-Meteo and its underlying ECMWF/Copernicus data
providers, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
as described on the [provider's licence page](https://open-meteo.com/en/licence).

## Running offline

```python
from scenarios import CATALOG, build_twin, list_scenarios

print(list_scenarios())
name = "three_days"
twin = build_twin(name)
rows = twin.run_full_day(steps=CATALOG[name]["steps"])
assert len(rows) == 864
```

For allocation-driven runs:

```python
from scenarios import CATALOG, build_twin
from simulation.closed_loop import run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate

name = "overnight_ev_cloudy"
records = run_closed_loop(build_twin(name, closed_loop=True), hybrid_allocate,
                          steps=CATALOG[name]["steps"])
```

The local session only reads the bundled JSON. It passes the saved payload
through the existing weather validation, sanitization and interpolation.
Each client has its own memory cache, avoiding contamination from earlier
live requests for the same location/date. There is no persistent cache or
network dependency. Tests block requests and socket connections.

`three_days` repeats the source day's weather at the same local time each
day. The twin's existing EV schedule also repeats; battery/appliance state
continues according to existing model behavior. This is not three distinct
historical weather days. The source-day weather anchor is preserved on reset.

Derating tests compare the hot profile against **the same irradiance at 25 C
ambient**, using the existing NOCT cell-temperature approximation in both
cases. Comparing different days would conflate temperature and irradiance.

Future work: `late_ev_arrival` requires explicit EV charging segments (#67).
The current EV configuration only supports seeded scenarios; this library
does not change the EV model or implement that case.
