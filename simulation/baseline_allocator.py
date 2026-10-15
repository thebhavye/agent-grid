import math


def _require_valid_kw(value, name):
    """Coerce to float, rejecting NaN/Inf/negatives loudly (issue #26).

    Physical power quantities must be finite and non-negative; anything
    else indicates a broken sensor row or agent bug, and failing here
    keeps NaN from silently poisoning every allocation downstream.
    """
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result


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

    Raises:
        ValueError: If available power or any demand is NaN, infinite,
            or negative, or if total demand overflows to infinity.
    """
    available_power_kw = _require_valid_kw(available_power_kw, "available_power_kw")
    agent_demands = {
        agent_id: _require_valid_kw(demand, f"demand for {agent_id!r}")
        for agent_id, demand in agent_demands.items()
    }
    total_demand = sum(agent_demands.values())

    # Check for overflow (issue #26)
    if not math.isfinite(total_demand):
        raise ValueError(
            f"total_demand overflowed to {total_demand!r}; "
            f"individual demands are finite but sum is not representable"
        )

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
