"""Fit synthetic daily health loss and peak temperature; no random row split."""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_health.dataset import generate_battery_dataset, summarize_days


DEFAULT_SCENARIOS = ("demo_sunny_home", "degraded_battery", "heatwave")


def fit_baseline(frame, held_out="heatwave") -> dict:
    """Hold out an entire scenario and report two least-squares MAEs.

    Health uses the same exposure terms as the synthetic generator, so near-
    zero error is expected and is NOT evidence of real-cell predictive skill.
    Temperature uses intercept, start temperature, mean ambient and peak C-rate.
    """
    daily = summarize_days(frame)
    train = daily[daily.scenario != held_out]
    test = daily[daily.scenario == held_out]
    if train.empty or test.empty:
        raise ValueError("need both training scenarios and a held-out scenario")
    x = train[["cycle_exposure", "calendar_exposure"]].to_numpy()
    coef = np.linalg.lstsq(x, (train.start_health_pct-train.end_health_pct).to_numpy(), rcond=None)[0]
    health = test.start_health_pct.to_numpy() - test[["cycle_exposure", "calendar_exposure"]].to_numpy() @ coef

    def thermal(rows):
        return np.column_stack((np.ones(len(rows)), rows.start_temperature_c,
                                rows.mean_ambient_c, rows.max_c_rate))

    tcoef = np.linalg.lstsq(thermal(train), train.peak_temperature_c, rcond=None)[0]
    temperature = thermal(test) @ tcoef
    return {
        "held_out_scenario": held_out, "train_days": len(train), "test_days": len(test),
        "health_mae_pct": float(np.mean(abs(health-test.end_health_pct.to_numpy()))),
        "temperature_mae_c": float(np.mean(abs(temperature-test.peak_temperature_c.to_numpy()))),
        "health_coefficients": coef.tolist(), "temperature_coefficients": tcoef.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = generate_battery_dataset(DEFAULT_SCENARIOS, args.days, args.seed)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output / "steps.csv", index=False)
        summarize_days(data).to_csv(args.output / "days.csv", index=False)
    print(json.dumps(fit_baseline(data), indent=2))
