"""Deterministic, sensor-only fault injection for digital-twin observations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Integral
from typing import Any, Iterable, Mapping

import pandas as pd


FAULT_TYPES = {"dropout", "nan", "stuck", "spike", "drift"}

# Device faults change delivered physics rather than only the observation.
# ``battery_derate`` accepts a ``factor`` in (0, 1] scaling both the charge
# and discharge power limits.
DEVICE_FAULT_TYPES = {"inverter_trip", "charger_fault", "battery_derate"}
DEVICE_FAULT_DEVICE = {
    "inverter_trip": "solar",
    "charger_fault": "ev",
    "battery_derate": "battery",
}


@dataclass
class FaultConfig:
    """Configuration for :class:`FaultInjector`.

    A fault uses either ``start_step``/``end_step`` (end is exclusive), or
    ``probability`` and ``duration_range`` (inclusive minimum and maximum).
    ``spike`` accepts ``value`` or ``offset``; ``drift`` accepts ``rate``.
    """

    enabled: bool = False
    seed: int = 0
    faults: list[Mapping[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        try:
            self.seed = int(self.seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("seed must be an integer") from exc
        if self.faults is None:
            self.faults = []
        self.faults = [_normalize_spec(spec) for spec in self.faults]


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError("each fault must be a mapping")
    spec = dict(spec)
    fault_type = spec.get("type", spec.get("fault_type"))
    key = spec.get("key")
    if not isinstance(fault_type, str) or fault_type not in FAULT_TYPES:
        raise ValueError(f"unsupported fault type: {fault_type!r}")
    if not isinstance(key, str) or not key:
        raise ValueError("fault key must be a non-empty string")
    spec["type"] = fault_type
    has_window = "start_step" in spec or "end_step" in spec
    has_probability = "probability" in spec
    if has_window and has_probability:
        raise ValueError("a fault must use a fixed window or probability, not both")
    if not has_window and not has_probability:
        raise ValueError("fault needs start_step/end_step or probability")
    if has_window:
        start = _integer(spec.get("start_step", 0), "start_step")
        end = _integer(spec.get("end_step", start), "end_step")
        if start < 0 or end <= start:
            raise ValueError("fault window must satisfy 0 <= start_step < end_step")
        spec.update(start_step=start, end_step=end)
    else:
        probability = _finite(spec["probability"], "probability")
        if not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError("probability must be finite and in [0, 1]")
        duration = spec.get("duration_range", spec.get("duration_steps", (1, 1)))
        if isinstance(duration, Integral):
            duration = (duration, duration)
        if not isinstance(duration, (tuple, list)) or len(duration) != 2:
            raise ValueError("duration_range must be (positive minimum, maximum)")
        minimum, maximum = (_integer(item, "duration") for item in duration)
        if minimum < 1 or maximum < minimum:
            raise ValueError("duration_range must be (positive minimum, maximum)")
        spec.update(probability=probability, duration_range=(minimum, maximum))
    if fault_type == "drift":
        spec["rate"] = _finite(spec.get("rate", spec.get("drift_per_step", 1.0)), "drift rate")
    if fault_type == "spike":
        if "value" in spec:
            spec["value"] = _finite(spec["value"], "spike value")
        else:
            spec["offset"] = _finite(spec.get("offset", spec.get("magnitude", 1.0)), "spike offset")
    return spec


class FaultInjector:
    """Apply configured faults without modifying clean input rows."""

    def __init__(self, config: FaultConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, FaultConfig) else FaultConfig(**config)
        self._rng = random.Random(self.config.seed)
        self._events: list[tuple[dict[str, Any], int, int]] = []
        self._probabilistic_specs = []
        self._spec_order = {}
        self._generated_step = -1
        self._last_good: dict[str, Any] = {}
        if self.config.enabled:
            self._build_events()

    def _build_events(self) -> None:
        for index, raw in enumerate(self.config.faults):
            spec = _normalize_spec(raw)
            self._spec_order[id(spec)] = index
            if "probability" not in spec:
                start = int(spec.get("start_step", 0))
                end = int(spec["end_step"])
                self._events.append((spec, start, end))
                continue
            self._probabilistic_specs.append(spec)

    def _advance_events(self, step: int) -> None:
        if step < self._generated_step:
            raise ValueError("fault steps must be supplied in chronological order")
        for current in range(self._generated_step + 1, step + 1):
            self._events = [event for event in self._events if event[2] > current]
            for spec in self._probabilistic_specs:
                if self._rng.random() < spec["probability"]:
                    duration = self._rng.randint(*spec["duration_range"])
                    self._events.append((spec, current, current + duration))
        self._generated_step = step

    def _active(self, step: int, key: str) -> list[tuple]:
        active = [event for event in self._events
                  if event[1] <= step < event[2] and event[0]["key"] == key]
        return sorted(active, key=lambda event: (self._spec_order[id(event[0])], event[1]))

    def apply(self, row: Mapping[str, Any], step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return copied observations and labels; steps must be chronological."""
        if not isinstance(row, Mapping):
            raise TypeError("row must be a mapping")
        observed = dict(row)
        if not self.config.enabled:
            return observed, []
        step = _integer(step, "step")
        if step < 0:
            raise ValueError("step must be non-negative")
        self._advance_events(step)
        labels = []
        for key, true_value in row.items():
            active = self._active(int(step), key)
            if not active:
                self._last_good[key] = true_value
                continue
            spec, start, _end = active[0]
            kind = spec["type"]
            if kind == "dropout":
                observed.pop(key, None)
                value = None
            elif kind == "nan":
                value = float("nan")
                observed[key] = value
            elif kind == "stuck":
                value = self._last_good.setdefault(key, true_value)
                observed[key] = value
            elif kind == "spike":
                if "value" in spec:
                    value = spec["value"]
                else:
                    value = true_value + float(spec.get("offset", spec.get("magnitude", 1.0)))
                observed[key] = value
            else:  # drift
                rate = float(spec.get("rate", spec.get("drift_per_step", 1.0)))
                value = true_value + rate * (step - start + 1)
                observed[key] = value
            if kind in {"spike", "drift"}:
                _finite(value, f"{kind} observed value")
            labels.append({"step": int(step), "key": key, "fault_type": kind,
                           "true_value": true_value, "observed_value": value})
        return observed, labels


def generate_fault_dataset(rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
                           config: FaultConfig | Mapping[str, Any],
                           output_path: str | Path | None = None) -> pd.DataFrame:
    """Create clean/observed/fault-labelled columns, one row per input row."""
    records = rows.to_dict(orient="records") if isinstance(rows, pd.DataFrame) else list(rows)
    injector = FaultInjector(config)
    output = []
    for step, row in enumerate(records):
        observed, labels = injector.apply(row, step)
        label_by_key = {label["key"]: label for label in labels}
        result = {"step": step}
        for key, clean in row.items():
            result[f"{key}_clean"] = clean
            result[f"{key}_observed"] = observed.get(key)
            result[f"{key}_fault"] = label_by_key.get(key, {}).get("fault_type", "none")
        output.append(result)
    frame = pd.DataFrame(output)
    if output_path is not None:
        frame.to_csv(output_path, index=False)
    return frame


def _normalize_device_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one device-fault specification."""
    if not isinstance(spec, Mapping):
        raise ValueError("each device fault must be a mapping")
    spec = dict(spec)
    fault_type = spec.get("type", spec.get("fault_type"))
    if not isinstance(fault_type, str) or fault_type not in DEVICE_FAULT_TYPES:
        raise ValueError(f"unsupported device fault type: {fault_type!r}")
    spec["type"] = fault_type
    has_window = "start_step" in spec or "end_step" in spec
    has_probability = "probability" in spec
    if has_window and has_probability:
        raise ValueError("a device fault must use a fixed window or probability, not both")
    if not has_window and not has_probability:
        raise ValueError("device fault needs start_step/end_step or probability")
    if has_window:
        start = _integer(spec.get("start_step", 0), "start_step")
        end = _integer(spec.get("end_step", start), "end_step")
        if start < 0 or end <= start:
            raise ValueError("fault window must satisfy 0 <= start_step < end_step")
        spec.update(start_step=start, end_step=end)
    else:
        probability = _finite(spec["probability"], "probability")
        if not 0 <= probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        duration = spec.get("duration_range", spec.get("duration_steps", (1, 1)))
        if isinstance(duration, Integral):
            duration = (duration, duration)
        if not isinstance(duration, (tuple, list)) or len(duration) != 2:
            raise ValueError("duration_range must be (positive minimum, maximum)")
        minimum, maximum = (_integer(item, "duration") for item in duration)
        if minimum < 1 or maximum < minimum:
            raise ValueError("duration_range must be (positive minimum, maximum)")
        spec.update(probability=probability, duration_range=(minimum, maximum))
    if fault_type == "battery_derate":
        factor = _finite(spec.get("factor", 0.5), "factor")
        if not 0 < factor <= 1:
            raise ValueError("battery_derate factor must be in (0, 1]")
        spec["factor"] = factor
    return spec


@dataclass
class DeviceFaultConfig:
    """Configuration for :class:`DeviceFaultInjector`.

    Device faults alter delivered physics instead of only observations.
    Supported types are ``inverter_trip`` (solar delivered is forced to
    zero), ``charger_fault`` (the EV accepts zero power while plugged in),
    and ``battery_derate`` (charge/discharge limits scaled by ``factor``).

    A fault uses either ``start_step``/``end_step`` (end exclusive) or
    ``probability`` plus ``duration_range``, matching :class:`FaultConfig`.
    """

    enabled: bool = False
    seed: int = 0
    faults: list[Mapping[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        try:
            self.seed = int(self.seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("seed must be an integer") from exc
        if self.faults is None:
            self.faults = []
        self.faults = [_normalize_device_spec(spec) for spec in self.faults]


def _neutral_device_effects() -> dict[str, Any]:
    """Return the no-op effects for a step without an active device fault."""
    return {"solar_scale": 1.0, "ev_limit_kw": None, "battery_factor": 1.0}


class DeviceFaultInjector:
    """Apply deterministic device faults that change delivered physics.

    ``apply(step)`` returns ``(effects, labels)``. ``effects`` carries a
    ``solar_scale`` (0.0 during an inverter trip), an optional
    ``ev_limit_kw`` (0.0 during a charger fault), and a ``battery_factor``
    scaling the battery power limits during a derate. Labels describe the
    active faults for the step so closed-loop records can carry them.
    """

    def __init__(self, config: DeviceFaultConfig | Mapping[str, Any]):
        self.config = (
            config if isinstance(config, DeviceFaultConfig) else DeviceFaultConfig(**config)
        )
        self._rng = random.Random(self.config.seed)
        self._events: list[tuple[dict[str, Any], int, int]] = []
        self._probabilistic_specs: list[dict[str, Any]] = []
        self._spec_order: dict[int, int] = {}
        self._generated_step = -1
        if self.config.enabled:
            self._build_events()

    def _build_events(self) -> None:
        for index, raw in enumerate(self.config.faults):
            spec = _normalize_device_spec(raw)
            self._spec_order[id(spec)] = index
            if "probability" not in spec:
                self._events.append(
                    (spec, int(spec.get("start_step", 0)), int(spec["end_step"]))
                )
                continue
            self._probabilistic_specs.append(spec)

    def _advance_events(self, step: int) -> None:
        if step < self._generated_step:
            raise ValueError("fault steps must be supplied in chronological order")
        for current in range(self._generated_step + 1, step + 1):
            self._events = [event for event in self._events if event[2] > current]
            for spec in self._probabilistic_specs:
                if self._rng.random() < spec["probability"]:
                    duration = self._rng.randint(*spec["duration_range"])
                    self._events.append((spec, current, current + duration))
        self._generated_step = step

    def _active(self, step: int) -> list[tuple[dict[str, Any], int, int]]:
        active = [event for event in self._events if event[1] <= step < event[2]]
        return sorted(active, key=lambda event: (self._spec_order[id(event[0])], event[1]))

    def apply(self, step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return the device effects and labels for one chronological step."""
        effects = _neutral_device_effects()
        if not self.config.enabled:
            return effects, []
        step = _integer(step, "step")
        if step < 0:
            raise ValueError("step must be non-negative")
        self._advance_events(step)
        labels = []
        for spec, _start, _end in self._active(step):
            kind = spec["type"]
            label = {"step": int(step), "device": DEVICE_FAULT_DEVICE[kind],
                     "fault_type": kind}
            if kind == "inverter_trip":
                effects["solar_scale"] = 0.0
            elif kind == "charger_fault":
                effects["ev_limit_kw"] = 0.0
            else:  # battery_derate
                effects["battery_factor"] = min(
                    effects["battery_factor"], float(spec["factor"])
                )
                label["factor"] = float(spec["factor"])
            labels.append(label)
        return effects, labels
