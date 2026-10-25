"""Reproduce baseline/PINN/ablation: python -m battery_health.train."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from battery_health.dataset import EmpiricalBaseline, generate_battery_dataset
from battery_health.pinn import INPUTS, train


def evaluate(frame, predicted):
    actual = frame[["next_health_pct", "next_temperature_c"]].to_numpy()
    error = np.abs(predicted - actual).mean(axis=0)
    daily = frame[["scenario", "seed", "day", "health_pct", "temperature_c",
                   "next_health_pct", "next_temperature_c"]].copy()
    daily["pred_loss"] = frame.health_pct.to_numpy() - predicted[:, 0]
    daily["pred_temp"] = predicted[:, 1]
    health_errors, temp_errors = [], []
    for _, day in daily.groupby(["scenario", "seed", "day"]):
        health_errors.append(abs(day.health_pct.iloc[0] - day.pred_loss.sum() - day.next_health_pct.iloc[-1]))
        temp_errors.append(abs(day.pred_temp.max() - day.next_temperature_c.max()))
    return dict(soh_mae_pct=float(error[0]), temperature_mae_c=float(error[1]),
                next_day_soh_mae_pct=float(np.mean(health_errors)),
                daily_peak_temperature_mae_c=float(np.mean(temp_errors)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    training = generate_battery_dataset(("clear_day", "monsoon_cloudy"), days=3, seed=42)
    heldout = generate_battery_dataset(("heatwave",), days=3, seed=13)
    baseline = EmpiricalBaseline().fit(training)
    model = train(training, epochs=args.epochs)
    ablation = train(training, epochs=args.epochs, physics_weight=0)
    result = {"heldout": "heatwave, seed 13, three days", "training_rows": len(training),
              "epochs": args.epochs, "baseline": evaluate(heldout, baseline.predict(heldout)),
              "pinn": evaluate(heldout, model.predict(heldout)),
              "without_physics": evaluate(heldout, ablation.predict(heldout)),
              "elapsed_seconds": time.perf_counter() - started}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "inputs": INPUTS,
                    "epochs": args.epochs, "seed": 42}, args.output)


if __name__ == "__main__":
    main()
