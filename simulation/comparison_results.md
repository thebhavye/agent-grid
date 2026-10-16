# Issue #57: paired closed-loop results

Command: `python -m simulation.compare`. Each run is 288 five-minute steps.
Weather is the existing synthetic offline test fixture, not historical weather.
Seeds and EV scenarios are paired with separate twins and isolated weather caches.
All twelve metrics for every pair are below; values use ten significant digits.
Energy is in kWh, SOC shortfall in percentage points, balance error in kW.
Count metrics count events (deadline misses may later complete).

| Seed | Scenario | Metric | Hybrid | Baseline | Hybrid - baseline |
|---|---|---|---|---|---|
| 1 | overnight | critical_unmet_kwh | 6.920283206 | 8.193509023 | -1.273225817 |
| 1 | overnight | priority_1_unmet_kwh | 5.104966117 | 6.242673541 | -1.137707424 |
| 1 | overnight | flexible_unmet_kwh | 22.8131159 | 18.82890654 | 3.984209366 |
| 1 | overnight | ev_shortfall_pct_at_departure | 1.460420328 | 1.169754858 | 0.29066547 |
| 1 | overnight | appliance_cycles_completed | 5 | 5 | 0 |
| 1 | overnight | appliance_cycles_incomplete_at_window_end | 3 | 3 | 0 |
| 1 | overnight | blocked_starts | 5 | 5 | 0 |
| 1 | overnight | curtailed_solar_kwh | 0 | 0 | 0 |
| 1 | overnight | battery_reserve_violation_steps | 0 | 0 | 0 |
| 1 | overnight | battery_charge_kwh | 5.728659496 | 7.029797166 | -1.30113767 |
| 1 | overnight | battery_discharge_kwh | 8.575636471 | 9.747442358 | -1.171805887 |
| 1 | overnight | energy_balance_max_error_kw | 5.329070518e-15 | 1.065814104e-14 | -5.329070518e-15 |
| 1 | midday | critical_unmet_kwh | 8.641865192 | 13.36256209 | -4.720696896 |
| 1 | midday | priority_1_unmet_kwh | 6.339860358 | 10.18099969 | -3.841139327 |
| 1 | midday | flexible_unmet_kwh | 57.25062987 | 18.64193248 | 38.60869739 |
| 1 | midday | ev_shortfall_pct_at_departure | 15.98559722 | 12.01776734 | 3.967829883 |
| 1 | midday | appliance_cycles_completed | 2 | 4 | -2 |
| 1 | midday | appliance_cycles_incomplete_at_window_end | 2 | 2 | 0 |
| 1 | midday | blocked_starts | 7 | 6 | 1 |
| 1 | midday | curtailed_solar_kwh | 0 | 6.661338148e-16 | -6.661338148e-16 |
| 1 | midday | battery_reserve_violation_steps | 0 | 0 | 0 |
| 1 | midday | battery_charge_kwh | 0 | 0.7194783476 | -0.7194783476 |
| 1 | midday | battery_discharge_kwh | 3.4164 | 4.064362919 | -0.6479629193 |
| 1 | midday | energy_balance_max_error_kw | 3.996802889e-15 | 3.996802889e-15 | 0 |
| 1 | evening_topup | critical_unmet_kwh | 7.689680236 | 8.647240106 | -0.9575598701 |
| 1 | evening_topup | priority_1_unmet_kwh | 5.701066117 | 6.588373414 | -0.8873072972 |
| 1 | evening_topup | flexible_unmet_kwh | 14.01626744 | 10.3919801 | 3.624287337 |
| 1 | evening_topup | ev_shortfall_pct_at_departure | 7.57199241 | 6.380775652 | 1.191216758 |
| 1 | evening_topup | appliance_cycles_completed | 5 | 5 | 0 |
| 1 | evening_topup | appliance_cycles_incomplete_at_window_end | 3 | 3 | 0 |
| 1 | evening_topup | blocked_starts | 5 | 5 | 0 |
| 1 | evening_topup | curtailed_solar_kwh | 7.401486831e-17 | 9.251858539e-17 | -1.850371708e-17 |
| 1 | evening_topup | battery_reserve_violation_steps | 0 | 0 | 0 |
| 1 | evening_topup | battery_charge_kwh | 5.645366076 | 6.946503746 | -1.30113767 |
| 1 | evening_topup | battery_discharge_kwh | 8.500622333 | 9.67242822 | -1.171805887 |
| 1 | evening_topup | energy_balance_max_error_kw | 5.329070518e-15 | 1.065814104e-14 | -5.329070518e-15 |
| 7 | overnight | critical_unmet_kwh | 9.056902882 | 10.38754281 | -1.330639928 |
| 7 | overnight | priority_1_unmet_kwh | 6.734955429 | 7.914318331 | -1.179362902 |
| 7 | overnight | flexible_unmet_kwh | 30.58292644 | 27.46220703 | 3.120719413 |
| 7 | overnight | ev_shortfall_pct_at_departure | 2.115883151 | 1.401951989 | 0.7139311619 |
| 7 | overnight | appliance_cycles_completed | 6 | 6 | 0 |
| 7 | overnight | appliance_cycles_incomplete_at_window_end | 3 | 3 | 0 |
| 7 | overnight | blocked_starts | 2 | 2 | 0 |
| 7 | overnight | curtailed_solar_kwh | 0 | 2.775557562e-17 | -2.775557562e-17 |
| 7 | overnight | battery_reserve_violation_steps | 0 | 0 | 0 |
| 7 | overnight | battery_charge_kwh | 3.700218618 | 4.720364929 | -1.020146311 |
| 7 | overnight | battery_discharge_kwh | 6.748820588 | 7.667565376 | -0.9187447875 |
| 7 | overnight | energy_balance_max_error_kw | 5.329070518e-15 | 5.329070518e-15 | 0 |
| 7 | midday | critical_unmet_kwh | 8.641865192 | 12.97291347 | -4.331048275 |
| 7 | midday | priority_1_unmet_kwh | 6.339860358 | 9.884124546 | -3.544264187 |
| 7 | midday | flexible_unmet_kwh | 68.16424193 | 18.64505819 | 49.51918374 |
| 7 | midday | ev_shortfall_pct_at_departure | 17.6226037 | 14.99293188 | 2.629671815 |
| 7 | midday | appliance_cycles_completed | 0 | 4 | -4 |
| 7 | midday | appliance_cycles_incomplete_at_window_end | 4 | 4 | 0 |
| 7 | midday | blocked_starts | 5 | 4 | 1 |
| 7 | midday | curtailed_solar_kwh | 0 | 4.163336342e-16 | -4.163336342e-16 |
| 7 | midday | battery_reserve_violation_steps | 0 | 0 | 0 |
| 7 | midday | battery_charge_kwh | 0.01452261414 | 0.3848918061 | -0.370369192 |
| 7 | midday | battery_discharge_kwh | 3.429479081 | 3.763033945 | -0.3335548647 |
| 7 | midday | energy_balance_max_error_kw | 3.996802889e-15 | 3.996802889e-15 | 0 |
| 7 | evening_topup | critical_unmet_kwh | 8.427497141 | 9.181322694 | -0.7538255529 |
| 7 | evening_topup | priority_1_unmet_kwh | 6.251383022 | 6.995293481 | -0.743910459 |
| 7 | evening_topup | flexible_unmet_kwh | 12.27511658 | 9.958021524 | 2.317095053 |
| 7 | evening_topup | ev_shortfall_pct_at_departure | 10.27914843 | 9.265644293 | 1.01350414 |
| 7 | evening_topup | appliance_cycles_completed | 6 | 6 | 0 |
| 7 | evening_topup | appliance_cycles_incomplete_at_window_end | 3 | 3 | 0 |
| 7 | evening_topup | blocked_starts | 2 | 2 | 0 |
| 7 | evening_topup | curtailed_solar_kwh | 0 | 4.625929269e-17 | -4.625929269e-17 |
| 7 | evening_topup | battery_reserve_violation_steps | 0 | 0 | 0 |
| 7 | evening_topup | battery_charge_kwh | 3.714741233 | 4.734887543 | -1.020146311 |
| 7 | evening_topup | battery_discharge_kwh | 6.761899669 | 7.680644456 | -0.9187447875 |
| 7 | evening_topup | energy_balance_max_error_kw | 5.329070518e-15 | 5.329070518e-15 | 0 |
| 42 | overnight | critical_unmet_kwh | 9.006080236 | 10.35815312 | -1.352072885 |
| 42 | overnight | priority_1_unmet_kwh | 6.704966117 | 7.891926188 | -1.186960071 |
| 42 | overnight | flexible_unmet_kwh | 28.02130472 | 24.95808014 | 3.063224573 |
| 42 | overnight | ev_shortfall_pct_at_departure | 12.33177442 | 11.25750498 | 1.074269437 |
| 42 | overnight | appliance_cycles_completed | 5 | 5 | 0 |
| 42 | overnight | appliance_cycles_incomplete_at_window_end | 2 | 2 | 0 |
| 42 | overnight | blocked_starts | 2 | 2 | 0 |
| 42 | overnight | curtailed_solar_kwh | 0 | 1.387778781e-17 | -1.387778781e-17 |
| 42 | overnight | battery_reserve_violation_steps | 0 | 0 | 0 |
| 42 | overnight | battery_charge_kwh | 6.971283723 | 7.709086366 | -0.7378026432 |
| 42 | overnight | battery_discharge_kwh | 9.694745092 | 10.35921089 | -0.6644657983 |
| 42 | overnight | energy_balance_max_error_kw | 1.065814104e-14 | 1.088018564e-14 | -2.220446049e-16 |
| 42 | midday | critical_unmet_kwh | 8.511865174 | 11.94274535 | -3.430880178 |
| 42 | midday | priority_1_unmet_kwh | 6.294084389 | 9.099234554 | -2.805150166 |
| 42 | midday | flexible_unmet_kwh | 25.80216816 | 15.6834459 | 10.11872226 |
| 42 | midday | ev_shortfall_pct_at_departure | 28.79324456 | 25.33968442 | 3.453560144 |
| 42 | midday | appliance_cycles_completed | 3 | 4 | -1 |
| 42 | midday | appliance_cycles_incomplete_at_window_end | 3 | 1 | 2 |
| 42 | midday | blocked_starts | 4 | 3 | 1 |
| 42 | midday | curtailed_solar_kwh | 0 | 2.775557562e-16 | -2.775557562e-16 |
| 42 | midday | battery_reserve_violation_steps | 0 | 0 | 0 |
| 42 | midday | battery_charge_kwh | 0.9042846967 | 2.646132084 | -1.741847388 |
| 42 | midday | battery_discharge_kwh | 4.230799702 | 5.799509201 | -1.568709499 |
| 42 | midday | energy_balance_max_error_kw | 3.996802889e-15 | 4.218847494e-15 | -2.220446049e-16 |
| 42 | evening_topup | critical_unmet_kwh | 7.077180236 | 7.80299837 | -0.7258181343 |
| 42 | evening_topup | priority_1_unmet_kwh | 5.23439945 | 5.945141615 | -0.7107421652 |
| 42 | evening_topup | flexible_unmet_kwh | 6.681252146 | 3.83080858 | 2.850443566 |
| 42 | evening_topup | ev_shortfall_pct_at_departure | 16.19552356 | 15.18922047 | 1.006303087 |
| 42 | evening_topup | appliance_cycles_completed | 5 | 5 | 0 |
| 42 | evening_topup | appliance_cycles_incomplete_at_window_end | 2 | 1 | 1 |
| 42 | evening_topup | blocked_starts | 2 | 2 | 0 |
| 42 | evening_topup | curtailed_solar_kwh | 0 | 1.850371708e-17 | -1.850371708e-17 |
| 42 | evening_topup | battery_reserve_violation_steps | 0 | 0 | 0 |
| 42 | evening_topup | battery_charge_kwh | 6.971283723 | 7.958202897 | -0.9869191749 |
| 42 | evening_topup | battery_discharge_kwh | 9.694745092 | 10.58356549 | -0.8888203958 |
| 42 | evening_topup | energy_balance_max_error_kw | 1.065814104e-14 | 1.088018564e-14 | -2.220446049e-16 |

Hybrid reduces critical unmet energy in all nine pairs, but increases flexible
unmet energy and EV departure shortfall in all nine. It is not uniformly better.
Requests and accepted cycles can diverge after different delivery decisions;
fairness checks compare exogenous draws, not allocation-dependent load histories.
No reserve violations occur. All 18 balance errors are below 1e-6 kW.
The overnight evening connection continues at midnight; it is not a departure
within this one-day horizon. Results therefore do not measure its next morning.
