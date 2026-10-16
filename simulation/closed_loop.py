"""Closed-loop twin runner for allocator comparisons.

The runner is deliberately separate from the existing open-loop simulation
helpers.  It observes a closed-loop twin, builds the same agent bids for the
selected allocator, and commits the resulting delivery back into the twin.
"""

import math
from typing import Callable

from agents.appliance_agent import ApplianceAgent
from agents.battery_agent import BatteryAgent, BatteryLimits
from digital_twin.faults import DeviceFaultConfig, DeviceFaultInjector
from digital_twin.faults import FaultConfig, FaultInjector
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
_ON_BAD_READING_POLICIES = ("raise", "hold_last", "exclude")
_DEFAULT_MAX_HOLD_STEPS = 3
_MISSING = object()


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
    priorities = {"battery_1": 5, "household_base_load": 1}
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


def _coerce_sensor_injector(source):
    """Normalize a sensor fault config/injector to a FaultInjector or None."""
    if source is None:
        return None
    if isinstance(source, FaultInjector):
        return source
    return FaultInjector(source)


def _coerce_device_injector(source):
    """Normalize a device fault config/injector to a DeviceFaultInjector or None."""
    if source is None:
        return None
    if isinstance(source, DeviceFaultInjector):
        return source
    return DeviceFaultInjector(source)


def _observation_watch_keys(agents) -> dict[str, list[str]]:
    """Map each observation key to the agents that read it.

    Used to attribute a bad reading to the affected agent(s) so the runner
    can sanitize the observation before ``update_state()`` without changing
    the agents themselves.
    """
    appliance_agents = [
        agent.agent_id for agent in agents if isinstance(agent, ApplianceAgent)
    ]
    return {
        "solar_gen_kw": ["solar_1"],
        "battery_soc_pct": ["battery_1"],
        "ev_soc_pct": ["ev_1"],
        "ev_departure_min": ["ev_1"],
        "ev_plugged_in": ["ev_1"],
        "timestamp": ["ev_1"] + appliance_agents,
        "appliance_desired_kw": list(appliance_agents),
    }


def _is_bad_reading(value) -> bool:
    """A reading is bad when missing, None, or non-finite numeric."""
    if value is _MISSING or value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(value)
    if isinstance(value, (str, dict)):
        return False
    return True


def _sanitize_observation(observation, watch, policy, max_hold_steps, state):
    """Sanitize bad readings before they reach the agents.

    Returns ``(sanitized, substitutions, excluded)``. ``state`` persists
    the last good value per key and the consecutive bad-step count per
    ``(agent_id, key)`` across the run. With the ``raise`` policy nothing
    is changed, so agents keep their fail-fast behavior.
    """
    sanitized = dict(observation)
    substitutions: list[dict] = []
    excluded: set[str] = set()
    if policy == "raise":
        return sanitized, substitutions, excluded

    last_good = state["last_good"]
    bad_counts = state["bad_counts"]
    for key, agent_ids in watch.items():
        raw = observation.get(key, _MISSING)
        if not _is_bad_reading(raw):
            last_good[key] = raw
            for agent_id in agent_ids:
                bad_counts[(agent_id, key)] = 0
            continue
        for agent_id in agent_ids:
            count = bad_counts.get((agent_id, key), 0) + 1
            bad_counts[(agent_id, key)] = count
            previous = last_good.get(key, _MISSING)
            held = (
                policy == "hold_last"
                and previous is not _MISSING
                and count <= max_hold_steps
            )
            if held:
                sanitized[key] = previous
                substitutions.append({
                    "agent_id": agent_id,
                    "key": key,
                    "policy": "hold_last",
                    "observed": raw,
                    "used": previous,
                })
            else:
                # Exclude the agent from this step's bidding. Keep the last
                # good value (when one exists) so the runner's own physics
                # and record stay finite.
                excluded.add(agent_id)
                if previous is not _MISSING:
                    sanitized[key] = previous
                substitutions.append({
                    "agent_id": agent_id,
                    "key": key,
                    "policy": "exclude",
                    "observed": None if raw is _MISSING else raw,
                    "used": None if previous is _MISSING else previous,
                })
    return sanitized, substitutions, excluded


def _bids_for_observation(agents, observation: dict, excluded=None) -> list[dict]:
    """Update included agents from an observation and return their bids.

    Excluded agents do not update state and bid zero so their demand goes
    unserved for the step.
    """
    excluded = excluded or set()
    bids = []
    for agent in agents:
        if agent.agent_id in excluded:
            bids.append({"agent_id": agent.agent_id, "demand_kw": 0.0, "urgency": 0.0})
            continue
        agent.update_state(observation)
        bids.append(agent.get_bid())
    return bids


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
        "ev_applied_kw": float(committed["ev_applied_kw"]),
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
    on_bad_reading: str = "raise",
    max_hold_steps: int = _DEFAULT_MAX_HOLD_STEPS,
    sensor_faults=None,
    device_faults=None,
) -> list[dict]:
    """Run an allocator against a fresh closed-loop digital twin.

    The twin is observed, bids are built from that observation, battery mode
    is selected, the allocator is called, and the resulting service is
    committed exactly once.  ``allocator`` can be ``hybrid_allocate`` or
    ``proportional_adapter`` (or another function with the same interface).

    Once sensor faults are injected the runner keeps the agents fail-fast
    but sanitizes the observation for them according to ``on_bad_reading``:

    - ``"raise"`` (default): nothing is changed; a bad reading still raises,
      reproducing the pre-fault behaviour exactly.
    - ``"hold_last"``: a bad value is replaced with the agent's last good
      observation for at most ``max_hold_steps`` consecutive bad steps, then
      the agent falls back to ``"exclude"``.
    - ``"exclude"``: the affected agent bids zero for that step, so its
      demand goes unserved and is recorded.

    ``sensor_faults`` and ``device_faults`` accept a config, an injector, or
    ``None`` (feature off). Device faults (``inverter_trip``,
    ``charger_fault``, ``battery_derate``) change delivered physics through
    the commit path. Every substitution and active fault is added to the
    step record only when the corresponding fault feature is enabled, so a
    run without faults is byte-identical to one before this feature.
    """
    if not getattr(twin, "_closed_loop", False):
        raise ValueError("run_closed_loop requires a twin configured with closed_loop=True")
    if twin.step_count != 0:
        raise ValueError("run_closed_loop requires a fresh twin with step_count == 0")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if on_bad_reading not in _ON_BAD_READING_POLICIES:
        raise ValueError(
            f"on_bad_reading must be one of {_ON_BAD_READING_POLICIES}, "
            f"got {on_bad_reading!r}"
        )
    if (
        isinstance(max_hold_steps, bool)
        or not isinstance(max_hold_steps, int)
        or max_hold_steps < 0
    ):
        raise ValueError("max_hold_steps must be a non-negative integer")

    agents = _build_closed_loop_agents(twin)
    priorities = _build_priorities(twin)
    battery = next(agent for agent in agents if isinstance(agent, BatteryAgent))
    watch = _observation_watch_keys(agents)
    sensor_injector = _coerce_sensor_injector(sensor_faults)
    device_injector = _coerce_device_injector(device_faults)
    sensor_enabled = sensor_injector is not None and sensor_injector.config.enabled
    device_enabled = device_injector is not None and device_injector.config.enabled
    fault_state = {"last_good": {}, "bad_counts": {}}
    dt_min = float(twin._time_delta_min)
    records = []

    for step in range(steps):
        observation = twin.observe()
        # Display-rounded SOC must not create energy or cross reserve limits.
        observation["battery_soc_pct"] = twin._battery.soc_pct
        observation["ev_soc_pct"] = twin._ev.soc_pct
        observation["solar_gen_kw"] = twin._pending_observation["solar_gen_kw"]

        device_effects = {
            "solar_scale": 1.0,
            "ev_limit_kw": None,
            "battery_factor": 1.0,
        }
        device_labels = []
        if device_injector is not None:
            device_effects, device_labels = device_injector.apply(step)
        observation["solar_gen_kw"] = (
            float(observation["solar_gen_kw"]) * device_effects["solar_scale"]
        )

        agent_observation = dict(observation)
        sensor_labels = []
        if sensor_injector is not None:
            agent_observation, sensor_labels = sensor_injector.apply(
                agent_observation, step
            )
        sanitized, substitutions, excluded = _sanitize_observation(
            agent_observation, watch, on_bad_reading, max_hold_steps, fault_state
        )

        bids = _bids_for_observation(agents, sanitized, excluded)
        bids.append({
            "agent_id": "household_base_load",
            "demand_kw": _require_finite_nonnegative(
                twin._appliances.config.base_load_kw, "base_load_kw"
            ),
            "urgency": 1.0,
        })
        ev_config = twin._ev.config
        ev_headroom_kw = (
            (100.0 - twin._ev.soc_pct) / 100 * ev_config.battery_capacity_kwh
            / ev_config.charge_efficiency / (dt_min / 60)
        )
        ev_cap_kw = max(0.0, ev_headroom_kw)
        if device_effects["ev_limit_kw"] is not None:
            ev_cap_kw = min(ev_cap_kw, device_effects["ev_limit_kw"])
        for bid in bids:
            if bid["agent_id"] == "ev_1":
                bid["demand_kw"] = min(bid["demand_kw"], ev_cap_kw)

        battery_factor = device_effects["battery_factor"]
        if "battery_1" in excluded:
            battery_demand_kw = 0.0
            battery_supply_kw = 0.0
            charge_headroom_kw = 0.0
        else:
            battery_demand_kw = _require_finite_nonnegative(
                battery.get_demand_kw(), "battery_demand_kw"
            )
            battery_supply_kw = _require_finite_nonnegative(
                battery.get_supply_kw(dt_min), "battery_supply_kw"
            )
            charge_headroom_kw = _require_finite_nonnegative(
                battery.get_charge_headroom_kw(dt_min), "battery_headroom_kw"
            )
        if battery_factor < 1.0:
            battery_demand_kw = min(
                battery_demand_kw, battery_factor * battery.limits.max_charge_kw
            )
            battery_supply_kw = min(
                battery_supply_kw, battery_factor * battery.limits.max_discharge_kw
            )
            charge_headroom_kw = min(
                charge_headroom_kw, battery_factor * battery.limits.max_charge_kw
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
        _validate_allocations(allocations, allocator_bids)
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
        actual_load_kw = (
            allocations["household_base_load"]
            + sum(committed["appliance_served_kw"].values())
            + committed["ev_applied_kw"]
        )
        if abs(actual_load_kw - delivered_to_loads_kw) > _EPSILON:
            raise RuntimeError("committed load power differs from allocation")
        _check_power_balance(
            observation["solar_gen_kw"], committed["battery_applied_kw"],
            actual_load_kw, curtailed_kw,
        )
        dt_hours = dt_min / 60
        battery_cfg = twin._battery.config
        battery_delta_kwh = (
            (twin._battery.soc_pct - observation["battery_soc_pct"])
            / 100 * battery_cfg.capacity_kwh
        )
        expected_battery_delta = (
            battery_kw * dt_hours * battery_cfg.charge_efficiency if battery_kw >= 0
            else battery_kw * dt_hours / battery_cfg.discharge_efficiency
        )
        ev_delta_kwh = (
            (twin._ev.soc_pct - observation["ev_soc_pct"])
            / 100 * ev_config.battery_capacity_kwh
        )
        if (abs(battery_delta_kwh - expected_battery_delta) > _EPSILON
                or abs(ev_delta_kwh - committed["ev_applied_kw"] * dt_hours
                       * ev_config.charge_efficiency) > _EPSILON):
            raise RuntimeError("committed SOC change violates energy accounting")
        if (observation["battery_soc_pct"] >= battery.limits.reserve_pct
                and twin._battery.soc_pct < battery.limits.reserve_pct - _EPSILON):
            raise RuntimeError("battery discharge crossed reserve")
        committed["battery_soc_pct"] = twin._battery.soc_pct
        committed["ev_soc_pct"] = twin._ev.soc_pct

        record = _step_record(
            observation,
            dt_min,
            mode["mode"],
            available_kw,
            bids,
            allocations,
            float(committed["battery_applied_kw"]),
            curtailed_kw,
            committed,
        )
        if sensor_enabled:
            record["sensor_faults"] = sensor_labels
            record["substitutions"] = substitutions
            record["excluded_agents"] = sorted(excluded)
        if device_enabled:
            record["device_faults"] = device_labels
        records.append(record)

    return records
