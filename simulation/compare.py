"""Offline paired comparisons: ``python -m simulation.compare``."""

from dataclasses import asdict

import pandas as pd

from digital_twin.ev_model import EVModelConfig, EVScenario
from digital_twin.test_digital_twin import FakeSession, make_test_twin
from digital_twin.weather_client import WeatherClient
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.metrics import METRICS, score_run


class _FixtureWeatherClient(WeatherClient):
    """Use the existing fake session, isolated from global/file weather caches."""

    def __init__(self):
        self._profiles = {}
        super().__init__(session=FakeSession())

    def _get_cached(self, target_date):
        return self._profiles.get(target_date)

    def _set_cached(self, target_date, data):
        self._profiles[target_date] = data


def _make_twin(seed, scenario):
    return make_test_twin(
        seed=seed, closed_loop=True, weather_client=_FixtureWeatherClient(),
        ev_config=EVModelConfig(scenario=scenario, seed=seed),
    )


def _initial_conditions(twin):
    return {
        "start": twin.current_time, "dt": twin._time_delta_min,
        "battery_soc": twin._battery.soc_pct, "ev_soc": twin._ev.soc_pct,
        "weather_payload": twin._weather_client._session._payload,
        "ev_segments": [asdict(segment) for segment in twin._ev._segments],
        "appliance_states": {
            spec.appliance_id: dict(twin._appliances.get_appliance_state(spec.appliance_id))
            for spec in twin._appliances.config.appliances
        },
        "configs": [asdict(model.config) for model in
                    (twin._solar, twin._battery, twin._ev, twin._appliances)],
    }


def _assert_paired(hybrid, baseline):
    if len(hybrid) != len(baseline):
        raise RuntimeError("paired runs have different step counts")
    for index, (left, right) in enumerate(zip(hybrid, baseline)):
        for field in ("timestamp", "dt_min", "solar_kw", "ev_plugged_in",
                      "ev_departure_min", "ev_target_soc_pct", "drawn_cycle_starts"):
            if left[field] != right[field]:
                raise RuntimeError(f"paired exogenous schedule differs at step {index}: {field}")


def compare(seeds=(1, 7, 42), scenarios=tuple(EVScenario)) -> pd.DataFrame:
    """Report every metric for hybrid/baseline and their signed difference.

    Each pair uses separate twins and fake weather sessions. Drawn starts, not
    allocation-dependent accepted cycle starts, are the fairness criterion.
    Values are measured as-is; no claim of hybrid superiority is assumed.
    """
    scenarios = tuple(scenarios)
    rows = []
    for seed in seeds:
        for scenario in scenarios:
            scenario = EVScenario(scenario)
            hybrid_twin, baseline_twin = _make_twin(seed, scenario), _make_twin(seed, scenario)
            if _initial_conditions(hybrid_twin) != _initial_conditions(baseline_twin):
                raise RuntimeError("paired twins have different initial conditions")
            hybrid = run_closed_loop(hybrid_twin, hybrid_allocate)
            baseline = run_closed_loop(baseline_twin, proportional_adapter)
            _assert_paired(hybrid, baseline)
            scores = {"hybrid": score_run(hybrid), "baseline": score_run(baseline)}
            row = {"seed": seed, "scenario": scenario.value}
            for method, score in scores.items():
                if score["energy_balance_max_error_kw"] >= 1e-6:
                    raise RuntimeError(f"{method} power balance failed: seed={seed}, {scenario.value}")
                row.update({f"{method}_{key}": value for key, value in score.items()})
            row.update({f"delta_{key}": scores["hybrid"][key] - scores["baseline"][key]
                        for key in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def results_markdown(frame):
    """Render all metrics without truncating columns or requiring tabulate."""
    lines = ["| Seed | Scenario | Metric | Hybrid | Baseline | Hybrid - baseline |",
             "|---|---|---|---|---|---|"]
    for row in frame.to_dict("records"):
        for metric in METRICS:
            values = [format(row[f"{method}_{metric}"], ".10g")
                      for method in ("hybrid", "baseline", "delta")]
            lines.append(f"| {row['seed']} | {row['scenario']} | {metric} | "
                         + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(results_markdown(compare()))
