"""Compare daily PINN and #69 baseline on identical data/split/targets."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from battery_health.baseline import DEFAULT_SCENARIOS, fit_baseline
from battery_health.dataset import generate_battery_dataset
from battery_health.pinn import INPUTS, daily_frame, train


def evaluate(frame, predicted):
    actual = frame[["end_health_pct", "peak_temperature_c"]].to_numpy()
    error = np.abs(predicted - actual).mean(axis=0)
    return dict(health_mae_pct=float(error[0]), temperature_mae_c=float(error[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    steps = generate_battery_dataset(DEFAULT_SCENARIOS, days=3, seed=42)
    daily = daily_frame(steps)
    training = daily[daily.scenario != "heatwave"]
    heldout = daily[daily.scenario == "heatwave"]
    model = train(training, epochs=args.epochs)
    ablation = train(training, epochs=args.epochs, physics_weight=0)
    result = {"heldout": "heatwave, seed 42, three days", "training_days": len(training),
              "targets": ["end_health_pct", "peak_temperature_c"],
              "epochs": args.epochs, "baseline": fit_baseline(steps),
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
