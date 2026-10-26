# Synthetic battery degradation baseline

This is Phase 5 preparation, not a trained PINN or a validated cell-life model.
All aging coefficients in `BatteryModelConfig` are **illustrative assumptions**.
No chemistry-specific published coefficient is asserted. Opt in with
`degradation=True`; default observations, physics and fixtures are unchanged.

Health declines by two additive terms, in percentage points:

- Cycle: `cycle_fade_pct_per_kwh * terminal_kWh * (1 + c_rate_stress*C) * stress`.
- Calendar: `calendar_fade_pct_per_day * dt_days * mean_SOC_fraction**2 * stress`.
- `stress = exp((cell_temperature - aging_reference_temp_c) / aging_temperature_scale_c)`.

Defaults are 0.0002 percentage points per terminal kWh, 0.001 percentage points
per day at full SOC, C-rate coefficient 1, reference 25 degrees C, and a
10-degree exponential scale. These are named demonstration parameters, not
fitted battery data. The exponential is bounded for numerical safety.
Actual transferred energy excludes unserved requests at SOC clamps. Health
is floored at zero. Health decline does **not** shrink capacity or change the
arbiter in this preparation task; the catalog's 6-kWh battery is a separate
capacity stress case. The existing first-order temperature model is retained.

```bash
python -m battery_health.baseline --days 3 --seed 42 --output battery-output
```

The optional output directory contains `steps.csv` and `days.csv`; generated
data is ignored by Git. Defaults cover sunny-home, degraded-capacity and
heatwave scenarios for three days each. Weather and EV schedules repeat the
saved day; appliance random streams continue across days. No network is used.

`generate_battery_dataset(scenarios, days, seed)` returns a DataFrame with:

| Columns | Meaning and units |
|---|---|
| scenario, seed, step, day, dt_min | Run identity, zero-based day and minutes |
| soc_pct, start_soc_pct | End/start SOC in percent |
| power_kw, c_rate | Actual terminal kW, positive charging; absolute kW/capacity |
| temperature_c, ambient_temperature_c | Cell and weather ambient degrees C |
| throughput_kwh | Cumulative absolute terminal energy |
| cycle_exposure, calendar_exposure | Cumulative terms before aging coefficients |
| health_pct, initial_health_pct | Synthetic health percentage, not measured SoH |
| initial_temperature_c, capacity_kwh | Initial cell temperature and fixed capacity |

`summarize_days(frame)` returns start/end health, start/peak cell temperature,
mean ambient, peak C-rate and daily exposure/throughput totals. The baseline
holds out an entire scenario rather than randomly splitting correlated steps.
Health is a least-squares fit on the same daily aging exposures; peak
temperature uses start temperature, mean ambient and peak C-rate. This predicts
the end of a day **conditional on that day's operating exposures**, not a blind
forecast of unknown future usage. A PINN comparison must use the same split,
inputs and target definitions. Near-zero synthetic health error is expected
because the fit recovers its generating coefficients; it is not real-cell skill.

Current targets are daily health percentage and peak cell temperature, as in
issue #69. Remaining useful life needs cell calibration and an owner decision;
this baseline supplies no RUL or certified thermal-safety claim.
