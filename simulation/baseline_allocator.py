"""Baseline power allocator.

This module intentionally contains no smart/priority logic.
It allocates available power in proportion to each agent's demand.
"""


def allocate(available_power_kw, agent_demands):
    """Split available power proportionally across all agents.

    Args:
        available_power_kw (float): Power available for this timestep.
        agent_demands (dict): Mapping of agent id -> requested power in kW.

    Returns:
        dict: Mapping of agent id -> allocated power in kW.
    """
    total_demand = sum(agent_demands.values())

    if total_demand == 0:
        return {agent_id: 0 for agent_id in agent_demands}

    return {
        agent_id: available_power_kw * (demand / total_demand)
        for agent_id, demand in agent_demands.items()
    }
