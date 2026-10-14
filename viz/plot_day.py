"""Plot one day of baseline allocations and static dataset state."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Support direct execution from the repository root or another directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_dataset import STEP_MIN, generate_dataset
from simulation.baseline_allocator import allocate
from simulation.run_simulation import run_simulation


def plot_day(seed=42, save_path="day_plot.png"):
    """Save baseline power allocation and dataset SOC for a seeded day.

    SOC is supplied by the static dataset, not updated by the allocator.
    """
    dataset = generate_dataset(seed=seed)
    ts = dataset["timeseries"]
    results = run_simulation(dataset, allocate)
    minutes = np.arange(len(ts)) * STEP_MIN
    solar = ts["solar_gen_kw"].to_numpy()
    total = np.array([sum(row.values()) for row in results])
    battery = [row["battery_1"] for row in results]
    ev = [row["ev_1"] for row in results]
    appliances = [
        sum(value for agent, value in row.items() if agent.startswith("appliance_"))
        for row in results
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    try:
        fig.suptitle(f"AgentGrid baseline — one day (seed {seed})")
        axes[0].plot(minutes, solar, color="darkorange", label="Solar generation")
        axes[0].plot(minutes, total, color="steelblue", label="Total allocated")
        axes[0].fill_between(
            minutes, total, solar, color="orange", alpha=0.25,
            label="Curtailed surplus",
        )
        axes[0].set_title("Solar generation and allocated power")
        axes[0].set_ylabel("Power (kW)")

        axes[1].stackplot(
            minutes, battery, ev, appliances,
            labels=["Battery", "EV", "Appliances"],
            colors=["seagreen", "mediumpurple", "steelblue"],
        )
        axes[1].set_title("Allocation by agent group")
        axes[1].set_ylabel("Allocated power (kW)")

        axes[2].plot(minutes, ts["battery_soc_pct"], color="seagreen", label="Battery SOC")
        axes[2].plot(minutes, ts["ev_soc_pct"], color="mediumpurple", label="EV SOC")
        # Each row's plug-in state covers the following five-minute interval.
        axes[2].stairs(
            ts["ev_plugged_in"].to_numpy(dtype=float) * 100,
            np.append(minutes, minutes[-1] + STEP_MIN),
            baseline=0, fill=True, linewidth=0,
            color="mediumpurple", alpha=0.15, label="EV plugged in",
        )
        axes[2].set_title("State of charge (static dataset; not allocation feedback)")
        axes[2].set_ylabel("SOC (%)")
        axes[2].set_ylim(0, 100)
        axes[2].set_xlabel("Time of day (HH:MM)")
        ticks = np.arange(0, 24 * 60, 120)
        axes[2].set_xticks(ticks, [f"{minute // 60:02d}:00" for minute in ticks])
        axes[2].set_xlim(0, 24 * 60)
        for ax in axes:
            ax.legend(loc="upper left", ncol=3, fontsize=9)
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot_day(seed=args.seed)
    print(f"Saved day_plot.png (seed {args.seed})")
