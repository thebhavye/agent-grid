"""Priority and urgency based allocation of available solar power."""

import math
from itertools import groupby
from numbers import Integral

from .baseline_allocator import _require_valid_kw, allocate


def hybrid_allocate(available_power_kw, bids, priorities) -> dict[str, float]:
    """Serve priority 1 before priority 2, then flexible bids.

    Flexible bids follow in descending urgency order. Urgencies within
    1e-9 of the highest urgency in a band tie; within each band, lower
    priority numbers go first, then agent_id. Equal-priority bids in a
    band share proportionally. Missing priorities default to 5.

    Every bid receives an entry, including zero demands. Allocations are
    capped at demand, and surplus is curtailed. Invalid power, non-finite
    or out-of-range urgency, duplicate IDs, and demand overflow raise
    ValueError. Priorities must be integers from 1 through 5 (excluding
    booleans). Inputs are not modified; all returned values are floats.
    """
    remaining = _require_valid_kw(available_power_kw, "available_power_kw")
    demands = {}
    priority_one = []
    priority_two = []
    flexible = []
    for bid in bids:
        agent_id = bid["agent_id"]
        if agent_id in demands:
            raise ValueError(f"duplicate agent_id: {agent_id!r}")
        demand = _require_valid_kw(bid["demand_kw"], f"demand for {agent_id!r}")
        urgency = float(bid["urgency"])
        if not math.isfinite(urgency) or not 0.0 <= urgency <= 1.0:
            raise ValueError(f"urgency for {agent_id!r} must be finite and in [0, 1]")
        demands[agent_id] = demand
        priority = priorities.get(agent_id, 5)
        if isinstance(priority, bool) or not isinstance(priority, Integral) or not 1 <= priority <= 5:
            raise ValueError(f"priority for {agent_id!r} must be an integer from 1 through 5")
        if priority == 1:
            priority_one.append(agent_id)
        elif priority == 2:
            priority_two.append(agent_id)
        else:
            flexible.append((urgency, priority, agent_id))

    _require_valid_kw(sum(demands.values()), "total_demand")
    result = {agent_id: 0.0 for agent_id in sorted(demands)}
    groups = [sorted(priority_one), sorted(priority_two)]
    flexible.sort(key=lambda item: (-item[0], item[1], item[2]))
    index = 0
    while index < len(flexible):
        anchor = flexible[index][0]
        end = index + 1
        while end < len(flexible) and anchor - flexible[end][0] <= 1e-9:
            end += 1
        band = sorted(flexible[index:end], key=lambda item: (item[1], item[2]))
        for _, members in groupby(band, key=lambda item: item[1]):
            groups.append([item[2] for item in members])
        index = end

    for group in groups:
        if not group or remaining <= 0.0:
            continue
        allocated = allocate(remaining, {agent_id: demands[agent_id] for agent_id in group})
        result.update({agent_id: float(value) for agent_id, value in allocated.items()})
        remaining = max(0.0, remaining - sum(allocated.values()))
    return result
