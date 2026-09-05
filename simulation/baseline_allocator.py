"""Baseline power allocator.

This module intentionally contains no smart/priority logic.
It allocates available power in proportion to each agent's demand,
capped at what each agent requested: no agent ever receives more than
its demand. Surplus solar (available > total demand) is curtailed --
left unallocated -- since the system models no battery-priority charging,
export, or other surplus sink yet.
"""


def allocate(available_power_kw, agent_demands):
    """Split available power proportionally across all agents.

    Args:
        available_power_kw (float): Power available for this timestep.
        agent_demands (dict): Mapping of agent id -> requested power in kW.

    Returns:
        dict: Mapping of agent id -> allocated power in kW. Each value is
            <= the corresponding demand; the total equals
            min(available_power_kw, total demand), with any surplus
            curtailed (unallocated).
    """
    total_demand = sum(agent_demands.values())

    if total_demand == 0:
        return {agent_id: 0 for agent_id in agent_demands}

    if total_demand <= available_power_kw:
        # Surplus case: every agent gets exactly what it asked for;
        # leftover solar is curtailed, not fabricated as consumption.
        return {agent_id: float(demand) for agent_id, demand in agent_demands.items()}

    return {
        agent_id: available_power_kw * (demand / total_demand)
        for agent_id, demand in agent_demands.items()
    }
