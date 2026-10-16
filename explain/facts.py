"""Deterministic facts derived from validated allocations."""

import math

from simulation.baseline_allocator import _require_valid_kw, allocate
from simulation.hybrid_arbiter import hybrid_allocate


def build_step_facts(timestamp, available_kw, bids, priorities, allocation,
                     missed_windows=(), *, strategy="hybrid") -> dict:
    """Describe one hybrid or baseline allocation without changing inputs.

    The keyword-only strategy selects the actual allocation policy. Missing
    allocation keys mean zero; inconsistent allocations raise ValueError
    rather than receiving a fabricated policy explanation. Agent order is
    canonical. Values retain full precision; text displays kW to six decimals.
    A missed-window event may name an agent or use ApplianceAgent's
    {appliance_id, timestamp} event shape. It overrides the service reason.
    """
    if strategy not in ("hybrid", "baseline"):
        raise ValueError("strategy must be hybrid or baseline")
    available = _require_valid_kw(available_kw, "available_kw")
    bids = [dict(bid) for bid in bids]
    # Reuse the arbiter's validation, including urgency, IDs and priorities.
    expected = hybrid_allocate(available, bids, priorities)
    demands = {bid["agent_id"]: float(bid["demand_kw"]) for bid in bids}
    if strategy == "baseline":
        expected = allocate(available, demands)
    if set(allocation) - set(demands):
        raise ValueError("allocation contains unknown agents")
    actual = {}
    for agent_id, demand in demands.items():
        value = _require_valid_kw(allocation.get(agent_id, 0.0), agent_id)
        if value > demand or not math.isclose(value, expected[agent_id], rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"allocation for {agent_id!r} does not match {strategy} policy")
        actual[agent_id] = value
    total_allocated = math.fsum(actual.values())
    if total_allocated > available + 1e-9:
        raise ValueError("allocation exceeds available power")

    missed = set()
    for event in missed_windows:
        if isinstance(event, str):
            missed.add(event)
        else:
            agent_id = event.get("agent_id")
            if agent_id is None:
                agent_id = f"appliance_{event['appliance_id']}"
            missed.add(agent_id)
    if missed - set(demands):
        raise ValueError("missed window contains unknown agents")

    shortage = math.fsum(demands.values()) > available
    critical_served = any(priorities.get(key, 5) <= 2 and value > 0
                          for key, value in actual.items())
    flexible_served = any(priorities.get(key, 5) > 2 and value > 0
                          for key, value in actual.items())
    agents = []
    for bid in sorted(bids, key=lambda item: item["agent_id"]):
        key = bid["agent_id"]
        demand, served = demands[key], actual[key]
        priority = int(priorities.get(key, 5))
        if key in missed:
            reason = "missed_window"
        elif demand == 0:
            reason = "no_demand"
        elif served == demand:
            reason = ("critical_first" if strategy == "hybrid" and priority <= 2
                      and shortage else "fully_served")
        elif strategy == "baseline" or priority <= 2 or served > 0 or available == 0:
            reason = "shared_proportionally"
        elif critical_served and not flexible_served:
            reason = "critical_first"
        else:
            # This fixed code includes priority tie-breaks within urgency bands.
            reason = "deferred_lower_urgency"
        agents.append({"agent_id": key, "demand_kw": demand, "allocated_kw": served,
                       "served_fraction": served / demand if demand else 1.0,
                       "priority": priority, "urgency": float(bid["urgency"]),
                       "reason": reason})
    return {"timestamp": timestamp, "strategy": strategy, "available_kw": available,
            "total_demand_kw": math.fsum(demands.values()),
            "curtailed_kw": max(0.0, available - total_allocated), "agents": agents}
