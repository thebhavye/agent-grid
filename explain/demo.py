"""Offline generator-day demo: python -m explain.demo --seed 42 --time 19:30."""

import argparse
import json

from data.generate_dataset import generate_dataset
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.run_simulation import _build_agents

from .facts import build_step_facts
from .render import render_template


def generated_day_facts(seed=42, strategy="hybrid"):
    """Yield facts for each generator interval, using the existing agents."""
    from simulation.baseline_allocator import allocate

    dataset = generate_dataset(seed=seed)
    roster = dataset["appliances"]
    agents = _build_agents(roster)
    priorities = {f"appliance_{row.appliance_id}": int(row.appliance_priority)
                  for row in roster.itertuples()}
    priorities["ev_1"] = priorities.get("appliance_ev_charger", 3)
    priorities["battery_1"] = 5
    for row in dataset["timeseries"].to_dict(orient="records"):
        bids, missed = [], []
        for agent in agents:
            agent.update_state(row)
            bids.append(agent.get_bid())
            if hasattr(agent, "get_missed_windows"):
                missed.extend(agent.get_missed_windows())
        available = row["solar_gen_kw"]
        allocation = (hybrid_allocate(available, bids, priorities) if strategy == "hybrid"
                      else allocate(available, {b["agent_id"]: b["demand_kw"] for b in bids}))
        yield build_step_facts(row["timestamp"], available, bids, priorities,
                               allocation, missed, strategy=strategy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time", default="19:30")
    args = parser.parse_args()
    selected = None
    for facts in generated_day_facts(args.seed):
        if facts["timestamp"] == args.time:
            selected = facts
    if selected is None:
        parser.error("time must match a generated five-minute interval, e.g. 19:30")
    print(json.dumps(selected, indent=2, sort_keys=True))
    print(render_template(selected))


if __name__ == "__main__":
    main()
