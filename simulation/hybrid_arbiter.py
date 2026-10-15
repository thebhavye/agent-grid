"""Priority and urgency based allocation of available solar power."""

import math
from itertools import groupby

from .baseline_allocator import _require_valid_kw, allocate


def hybrid_allocate(available_power_kw, bids, priorities) -> dict[str, float]:
    """Serve priority 1-2 first, proportionally when critical power is short.

    Flexible bids follow in descending urgency order. Urgencies within
    1e-9 of the highest urgency in a band tie; within each band, lower
    priority numbers go first, then agent_id. Equal-priority bids in a
    band share proportionally. Missing priorities default to 5.

    Every bid receives an entry, including zero demands. Allocations are
    capped at demand, and surplus is curtailed. Invalid power, non-finite
    or out-of-range urgency, duplicate IDs, and demand overflow raise
    ValueError. Inputs are not modified.
    """
    remaining = _require_valid_kw(available_power_kw, "available_power_kw")
    demands = {}
    critical = []
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
        if priority <= 2:
            critical.append(agent_id)
        else:
            flexible.append((urgency, priority, agent_id))

    _require_valid_kw(sum(demands.values()), "total_demand")
    result = {agent_id: 0.0 for agent_id in sorted(demands)}
    groups = [sorted(critical)]
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
        result.update(allocated)
        remaining = max(0.0, remaining - sum(allocated.values()))
    return result
