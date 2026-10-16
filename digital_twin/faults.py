"""Deterministic, sensor-only fault injection for digital-twin observations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


FAULT_TYPES = {"dropout", "nan", "stuck", "spike", "drift"}


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
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
        if self.faults is None:
            self.faults = []
        self.faults = list(self.faults)
        for spec in self.faults:
            _normalize_spec(spec)


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError("each fault must be a mapping")
    fault_type = spec.get("type", spec.get("fault_type"))
    key = spec.get("key")
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"unsupported fault type: {fault_type!r}")
    if not isinstance(key, str) or not key:
        raise ValueError("fault key must be a non-empty string")
    has_window = "start_step" in spec or "end_step" in spec
    has_probability = "probability" in spec
    if has_window and has_probability:
        raise ValueError("a fault must use a fixed window or probability, not both")
    if not has_window and not has_probability:
        raise ValueError("fault needs start_step/end_step or probability")
    if has_window:
        start = int(spec.get("start_step", 0))
        end = int(spec.get("end_step", start))
        if start < 0 or end <= start:
            raise ValueError("fault window must satisfy 0 <= start_step < end_step")
    else:
        probability = float(spec["probability"])
        if not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError("probability must be finite and in [0, 1]")
        duration = spec.get("duration_range", spec.get("duration_steps", (1, 1)))
        if isinstance(duration, int):
            duration = (duration, duration)
        if len(duration) != 2 or int(duration[0]) < 1 or int(duration[1]) < int(duration[0]):
            raise ValueError("duration_range must be (positive minimum, maximum)")
    if fault_type == "drift" and not math.isfinite(float(spec.get("rate", spec.get("drift_per_step", 1.0)))):
        raise ValueError("drift rate must be finite")
    if fault_type == "spike" and "value" in spec and not math.isfinite(float(spec["value"])):
        raise ValueError("spike value must be finite")
    return dict(spec)


class FaultInjector:
    """Apply configured faults without modifying clean input rows."""

    def __init__(self, config: FaultConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, FaultConfig) else FaultConfig(**config)
        self._rng = random.Random(self.config.seed)
        self._events: list[tuple[dict[str, Any], int, int]] = []
        self._last_good: dict[str, Any] = {}
        if self.config.enabled:
            self._build_events()

    def _build_events(self) -> None:
        for raw in self.config.faults:
            spec = _normalize_spec(raw)
            if "probability" not in spec:
                start = int(spec.get("start_step", 0))
                end = int(spec["end_step"])
                self._events.append((spec, start, end))
                continue
            minimum, maximum = spec.get("duration_range", spec.get("duration_steps", (1, 1)))
            minimum, maximum = int(minimum), int(maximum)
            # The finite horizon is intentionally generous; apply() remains
            # deterministic for normal day-sized (288 step) datasets.
            for step in range(10000):
                if self._rng.random() < float(spec["probability"]):
                    duration = self._rng.randint(minimum, maximum)
                    self._events.append((spec, step, step + duration))

    def _active(self, step: int, key: str) -> list[dict[str, Any]]:
        return [spec for spec, start, end in self._events if start <= step < end and spec["key"] == key]

    def apply(self, row: Mapping[str, Any], step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return ``(observed_row, labels)`` for one timestep."""
        if not isinstance(row, Mapping):
            raise TypeError("row must be a mapping")
        observed = dict(row)
        if not self.config.enabled:
            return observed, []
        labels = []
        for key, true_value in row.items():
            active = self._active(int(step), key)
            if not active:
                self._last_good[key] = true_value
                continue
            spec = active[0]
            kind = spec["type"]
            if kind == "dropout":
                observed.pop(key, None)
                value = None
            elif kind == "nan":
                value = float("nan")
                observed[key] = value
            elif kind == "stuck":
                value = self._last_good.get(key, true_value)
                observed[key] = value
            elif kind == "spike":
                if "value" in spec:
                    value = spec["value"]
                else:
                    value = true_value + float(spec.get("offset", spec.get("magnitude", 1.0)))
                observed[key] = value
            else:  # drift
                rate = float(spec.get("rate", spec.get("drift_per_step", 1.0)))
                start = next(start for candidate, start, end in self._events if candidate is spec and start <= step < end)
                value = true_value + rate * (step - start + 1)
                observed[key] = value
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
