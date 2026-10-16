"""Closed-loop twin runner for allocator comparisons.

The runner is deliberately separate from the existing open-loop simulation
helpers.  It observes a closed-loop twin, builds the same agent bids for the
selected allocator, and commits the resulting delivery back into the twin.
"""

import math
from typing import Callable

from agents.battery_agent import BatteryAgent, BatteryLimits
from simulation.battery_mode import choose_battery_mode
from simulation.run_simulation import (
    _build_agents,
    _coerce_roster_records,
    get_twin_appliances_df,
    get_twin_ev_max_charge_kw,
)
from .baseline_allocator import allocate


Allocator = Callable[[float, list[dict], dict[str, int]], dict[str, float]]
_EPSILON = 1e-9


def proportional_adapter(
    available_kw: float,
    bids: list[dict],
    priorities: dict[str, int],
) -> dict[str, float]:
    """Adapt the baseline proportional allocator to the hybrid interface."""
    del priorities
    return {
        agent_id: float(value)
        for agent_id, value in allocate(
            available_kw,
            {bid["agent_id"]: bid["demand_kw"] for bid in bids},
        ).items()
    }


def _require_finite_nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise RuntimeError(f"{name} must be finite and non-negative")
    return result


def _build_priorities(twin) -> dict[str, int]:
    """Build allocator priorities from the twin's configured appliance roster."""
    records = _coerce_roster_records(twin._appliances.config)
    priorities = {"battery_1": 5}
    ev_priority = 3
    for entry in records or []:
        appliance_id = entry["appliance_id"]
        priority = entry.get("appliance_priority")
        if appliance_id == "ev_charger":
            if priority is not None:
                ev_priority = int(priority)
            continue
        priorities[f"appliance_{appliance_id}"] = int(priority)
    priorities["ev_1"] = ev_priority
    return priorities


def _build_closed_loop_agents(twin):
    """Construct agents using the twin's roster and EV charging limit."""
    appliances_df = get_twin_appliances_df(twin._appliances.config)
    agents = _build_agents(
        appliances_df,
        ev_max_charge_kw=get_twin_ev_max_charge_kw(twin._ev.config),
    )
    battery = next(agent for agent in agents if agent.agent_id == "battery_1")
    battery.limits = BatteryLimits.from_twin_config(twin._battery.config)
    return agents


def _bids_for_observation(agents, observation: dict) -> list[dict]:
    """Update agents from an observation and return their bids."""
    for agent in agents:
        agent.update_state(observation)
    return [agent.get_bid() for agent in agents]


def _validate_allocations(allocations: dict, bids: list[dict]) -> None:
    """Reject allocator output that violates demand caps or is malformed."""
    demand_by_id = {bid["agent_id"]: float(bid["demand_kw"]) for bid in bids}
    if set(allocations) != set(demand_by_id):
        raise RuntimeError("allocator must return exactly one allocation per bid")
    for agent_id, demand in demand_by_id.items():
        value = allocations[agent_id]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"allocation for {agent_id!r} must be numeric"
            ) from exc
        if not math.isfinite(numeric) or numeric < -_EPSILON:
            raise RuntimeError(f"invalid allocation for {agent_id!r}: {value!r}")
        if numeric > demand + _EPSILON:
            raise RuntimeError(
                f"allocation for {agent_id!r} exceeds demand: {numeric} > {demand}"
            )


def _allocation_record(bids: list[dict], allocations: dict) -> dict[str, dict[str, float]]:
    """Return a stable per-agent demand/allocation/urgency record."""
    return {
        bid["agent_id"]: {
            "demand_kw": float(bid["demand_kw"]),
            "allocated_kw": float(allocations[bid["agent_id"]]),
            "urgency": float(bid["urgency"]),
        }
        for bid in sorted(bids, key=lambda item: item["agent_id"])
    }


def _check_power_balance(
    solar_kw: float,
    battery_kw: float,
    delivered_to_loads_kw: float,
    curtailed_kw: float,
) -> None:
    """Check the closed-loop power conservation invariant."""
    charge_kw = max(0.0, battery_kw)
    discharge_kw = max(0.0, -battery_kw)
    if charge_kw > _EPSILON and discharge_kw > _EPSILON:
        raise RuntimeError("battery cannot charge and discharge in one step")
    if abs(
        solar_kw + discharge_kw
        - delivered_to_loads_kw
        - charge_kw
        - curtailed_kw
    ) > _EPSILON:
        raise RuntimeError("closed-loop power balance violated")


def _step_record(
    observation: dict,
    dt_min: float,
    mode: str,
    available_kw: float,
    bids: list[dict],
    allocations: dict,
    battery_kw: float,
    curtailed_kw: float,
    committed: dict,
) -> dict:
    """Build the public record for one committed interval."""
    agents = _allocation_record(bids, allocations)
    desired = dict(observation["appliance_desired_kw"])
    served = dict(committed.get("appliance_served_kw", {}))
    appliances = {
        appliance_id: {
            "desired_kw": float(desired_kw),
            "served_kw": float(served.get(appliance_id, 0.0)),
        }
        for appliance_id, desired_kw in sorted(desired.items())
    }
    return {
        "timestamp": observation["timestamp"],
        "dt_min": float(dt_min),
        "solar_kw": float(observation["solar_gen_kw"]),
        "mode": mode,
        "available_kw": float(available_kw),
        "agents": agents,
        "demand_kw": {key: value["demand_kw"] for key, value in agents.items()},
        "allocated_kw": {
            key: value["allocated_kw"] for key, value in agents.items()
        },
        "urgency": {key: value["urgency"] for key, value in agents.items()},
        "battery_kw": float(battery_kw),
        "curtailed_kw": float(curtailed_kw),
        "appliances": appliances,
        "appliance_desired_kw": {
            key: value["desired_kw"] for key, value in appliances.items()
        },
        "appliance_served_kw": {
            key: value["served_kw"] for key, value in appliances.items()
        },
        "battery_soc_pct": float(committed["battery_soc_pct"]),
        "ev_soc_pct": float(committed["ev_soc_pct"]),
        "ev_plugged_in": bool(committed["ev_plugged_in"]),
    }


def run_closed_loop(
    twin,
    allocator: Allocator,
    steps: int = 288,
) -> list[dict]:
    """Run an allocator against a fresh closed-loop digital twin.

    The twin is observed, bids are built from that observation, battery mode
    is selected, the allocator is called, and the resulting service is
    committed exactly once.  ``allocator`` can be ``hybrid_allocate`` or
    ``proportional_adapter`` (or another function with the same interface).
    """
    if not getattr(twin, "_closed_loop", False):
        raise ValueError("run_closed_loop requires a twin configured with closed_loop=True")
    if twin.step_count != 0:
        raise ValueError("run_closed_loop requires a fresh twin with step_count == 0")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")

    agents = _build_closed_loop_agents(twin)
    priorities = _build_priorities(twin)
    battery = next(agent for agent in agents if isinstance(agent, BatteryAgent))
    dt_min = float(twin._time_delta_min)
    records = []

    for _ in range(steps):
        observation = twin.observe()
        bids = _bids_for_observation(agents, observation)

        battery_demand_kw = _require_finite_nonnegative(
            battery.get_demand_kw(), "battery_demand_kw"
        )
        battery_supply_kw = _require_finite_nonnegative(
            battery.get_supply_kw(dt_min), "battery_supply_kw"
        )
        charge_headroom_kw = _require_finite_nonnegative(
            battery.get_charge_headroom_kw(dt_min), "battery_headroom_kw"
        )
        other_demand_kw = sum(
            float(bid["demand_kw"])
            for bid in bids
            if bid["agent_id"] != "battery_1"
        )
        mode = choose_battery_mode(
            observation["solar_gen_kw"],
            other_demand_kw,
            battery_demand_kw,
            battery_supply_kw,
            charge_headroom_kw,
        )

        battery_id = "battery_1"
        non_battery_bids = [bid for bid in bids if bid["agent_id"] != battery_id]
        if mode["mode"] == "charge":
            allocator_bids = non_battery_bids + [{
                "agent_id": battery_id,
                "demand_kw": mode["battery_demand_kw"],
                "urgency": float(next(
                    bid["urgency"] for bid in bids if bid["agent_id"] == battery_id
                )),
            }]
            available_kw = float(observation["solar_gen_kw"])
        elif mode["mode"] == "discharge":
            allocator_bids = non_battery_bids
            available_kw = float(observation["solar_gen_kw"]) + mode["battery_supply_kw"]
        else:
            allocator_bids = non_battery_bids
            available_kw = float(observation["solar_gen_kw"])

        allocations = allocator(available_kw, allocator_bids, priorities)
        # Allocators omit the battery in discharge/idle mode by design. Keep
        # the record complete by adding an explicit zero battery allocation.
        allocations = {agent_id: float(value) for agent_id, value in allocations.items()}
        if mode["mode"] != "charge":
            allocations[battery_id] = 0.0
        _validate_allocations(allocations, bids)

        load_allocations = {
            agent_id: value
            for agent_id, value in allocations.items()
            if agent_id not in {"solar_1", battery_id}
        }
        delivered_to_loads_kw = sum(load_allocations.values())
        if mode["mode"] == "charge":
            battery_kw = allocations[battery_id]
        elif mode["mode"] == "discharge":
            battery_kw = -(delivered_to_loads_kw - float(observation["solar_gen_kw"]))
        else:
            battery_kw = 0.0

        curtailed_kw = max(
            0.0,
            float(observation["solar_gen_kw"])
            + max(0.0, -battery_kw)
            - delivered_to_loads_kw
            - max(0.0, battery_kw),
        )
        if mode["mode"] == "discharge" and curtailed_kw > _EPSILON:
            raise RuntimeError("discharge mode cannot curtail power")
        _check_power_balance(
            float(observation["solar_gen_kw"]),
            battery_kw,
            delivered_to_loads_kw,
            curtailed_kw,
        )

        delivered = {"battery_1": battery_kw}
        for agent_id, value in load_allocations.items():
            if agent_id == "ev_1":
                delivered[agent_id] = value
            elif agent_id.startswith("appliance_"):
                delivered[agent_id] = value
        committed = twin.commit(delivered)

        records.append(_step_record(
            observation,
            dt_min,
            mode["mode"],
            available_kw,
            bids,
            allocations,
            float(committed["battery_applied_kw"]),
            curtailed_kw,
            committed,
        ))

    return records
