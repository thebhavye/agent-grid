"""Save an offline paired closed-loop comparison, not a dataset replay."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenarios import CATALOG, build_twin, list_scenarios
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.compare import _assert_paired, _initial_conditions
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.metrics import score_run


def _paired_runs(source):
    factory = (lambda: build_twin(source, closed_loop=True)) if isinstance(source, str) else source
    twins = [factory(), factory()]
    if twins[0] is twins[1]:
        raise ValueError("factory must return separate fresh twins")
    if _initial_conditions(twins[0]) != _initial_conditions(twins[1]):
        raise ValueError("factory returned different initial conditions")
    steps = CATALOG[source]["steps"] if isinstance(source, str) else 288
    runs = [run_closed_loop(twin, allocator, steps=steps) for twin, allocator in
            zip(twins, (hybrid_allocate, proportional_adapter))]
    _assert_paired(*runs)
    for records in runs:
        if score_run(records)["energy_balance_max_error_kw"] >= 1e-6:
            raise RuntimeError("paired plot power balance failed")
    return runs


def _unmet(records, critical):
    return [sum(max(0, agent["demand_kw"] - agent["allocated_kw"])
                for name, agent in row["agents"].items()
                if name not in {"battery_1", "solar_1"}
                and (row["priorities"][name] <= 2) == critical)
            * row["dt_min"] / 60 for row in records]


def plot_comparison(scenario_or_twin_factory="clear_day", save_path="comparison.png"):
    """Plot matched hybrid/baseline runs from an offline name or fresh-twin factory.

    Catalog names use their configured step count (including three_days).
    Factories use 288 steps and must return equivalent closed-loop twins.
    Power and shading cover full intervals; SOC points are post-commit values
    at interval ends. EV departures mark actual events, not the run boundary.
    """
    runs = _paired_runs(scenario_or_twin_factory)
    first = runs[0]
    edges = np.r_[0.0, np.cumsum([r["dt_min"] for r in first])]
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2.5, 2, 2, 1.2]})
    try:
        title = scenario_or_twin_factory if isinstance(scenario_or_twin_factory, str) else "factory"
        fig.suptitle(f"AgentGrid paired closed-loop comparison: {title}")
        axes[0].stairs([r["solar_kw"] for r in first], edges,
                       color="black", label="Solar")
        modes = axes[1].inset_axes([0, 0, 1, 0.16])
        for records, label, color in zip(runs, ("Hybrid", "Baseline"),
                                         ("steelblue", "darkorange")):
            delivered = [sum(r["appliance_served_kw"].values()) + r["ev_applied_kw"]
                         + r["agents"]["household_base_load"]["allocated_kw"] for r in records]
            axes[0].stairs(delivered, edges, color=color, label=f"{label} delivered")
            axes[1].plot(edges[1:], [r["battery_soc_pct"] for r in records],
                         color=color, label=f"{label} battery")
            axes[2].plot(edges[1:], [r["ev_soc_pct"] if r["ev_plugged_in"] else np.nan
                                     for r in records], color=color, label=f"{label} EV")
            axes[3].stairs(_unmet(records, True), edges, color=color,
                           label=f"{label} critical")
            axes[3].stairs(_unmet(records, False), edges, color=color, linestyle="--",
                           label=f"{label} flexible")
            base = 0 if label == "Hybrid" else 1
            for mode, mode_color in (("charge", "seagreen"), ("discharge", "firebrick"),
                                     ("idle", "lightgray")):
                modes.stairs([base + int(r["mode"] == mode) for r in records], edges,
                             baseline=base, fill=True, color=mode_color,
                             linewidth=0, label=mode if base == 0 else None)
        modes.set(xlim=(0, edges[-1]), ylim=(0, 2), yticks=[0.5, 1.5],
                  yticklabels=["Hybrid", "Baseline"], xticks=[])
        modes.tick_params(axis="y", labelsize=7)
        modes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=7)
        axes[1].plot(edges[1:], [r["battery_reserve_pct"] for r in first],
                     color="black", linestyle=":", label="Reserve")
        axes[1].set_ylim(-25, 105)
        axes[1].set_yticks([0, 20, 40, 60, 80, 100])
        axes[2].stairs([100 * r["ev_plugged_in"] for r in first], edges,
                       baseline=0, fill=True, color="steelblue", alpha=0.12,
                       linewidth=0, label="Plugged in")
        axes[2].stairs([r["ev_target_soc_pct"] if r["ev_target_soc_pct"] is not None
                        else np.nan for r in first], edges,
                       color="black", linestyle=":", label="Target")
        for i, row in enumerate(first):
            if row["ev_departures"]:
                axes[2].axvline(edges[i+1], color="firebrick", linestyle="--",
                                label="Departure" if not any(
                                    r["ev_departures"] for r in first[:i]) else None)
        axes[0].set_ylabel("Power (kW)")
        axes[1].set_ylabel("Battery SOC (%)")
        axes[2].set(ylabel="EV SOC (%)", ylim=(0, 105))
        axes[3].set_ylabel("Unmet per interval (kWh)")
        metrics = [("critical_unmet_kwh", "Critical unmet (kWh)"),
                   ("ev_shortfall_pct_at_departure", "EV shortfall (pp)"),
                   ("curtailed_solar_kwh", "Curtailed solar (kWh)")]
        scores = [score_run(records) for records in runs]
        if all("net_cost" in score for score in scores):
            metrics.append(("net_cost", "Net cost (tariff units)"))
        axes[4].axis("off")
        table = axes[4].table(
            cellText=[[name, *(f"{score[key]:.4f}" for score in scores)]
                      for key, name in metrics],
            colLabels=["Outcome", "Hybrid", "Baseline"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        start = int(first[0]["timestamp"][:2])*60 + int(first[0]["timestamp"][3:])
        ticks = np.arange(0, edges[-1]+1, 180)
        axes[3].set_xticks(ticks, [f"{int((start+t)//1440)}d "
                                  f"{int((start+t)%1440)//60:02d}:"
                                  f"{int((start+t)%60):02d}" for t in ticks])
        axes[3].set_xlabel("Elapsed day and time (HH:MM)")
        axes[3].tick_params(axis="x", labelbottom=True)
        for ax in axes[:4]:
            ax.set_xlim(0, edges[-1])
            ax.grid(alpha=0.2)
            ax.legend(loc="upper left", ncol=3, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(save_path, dpi=150)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list_scenarios(), default="clear_day")
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()
    plot_comparison(args.scenario, args.output)
    print(f"Saved {args.output}")
