from agents.base_agent import Agent

import math
from dataclasses import dataclass, fields

BATTERY_MAX_CHARGE_KW = 5.0


@dataclass
class BatteryLimits:
    """Physical power/energy limits plus the minimum discharge SOC."""

    capacity_kwh: float = 12.0
    max_charge_kw: float = 6.0
    max_discharge_kw: float = 12.0
    charge_efficiency: float = 0.949
    discharge_efficiency: float = 0.949
    reserve_pct: float = 20.0

    def __post_init__(self):
        for field in fields(self):
            try:
                value = float(getattr(self, field.name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field.name} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            setattr(self, field.name, value)
        for name in ("capacity_kwh", "max_charge_kw", "max_discharge_kw"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("charge_efficiency", "discharge_efficiency"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.reserve_pct < 100:
            raise ValueError("reserve_pct must be in [0, 100)")

    @classmethod
    def from_twin_config(cls, cfg):
        """Convert BatteryModelConfig C-rates to kW; retain the 20% reserve."""
        return cls(
            capacity_kwh=cfg.capacity_kwh,
            max_charge_kw=cfg.max_charge_rate_c * cfg.capacity_kwh,
            max_discharge_kw=cfg.max_discharge_rate_c * cfg.capacity_kwh,
            charge_efficiency=cfg.charge_efficiency,
            discharge_efficiency=cfg.discharge_efficiency,
        )


def _require_positive_timestep(time_delta_min):
    try:
        minutes = float(time_delta_min)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_delta_min must be finite and positive") from exc
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("time_delta_min must be finite and positive")
    return minutes / 60.0


def _require_finite_soc(value):
    """Coerce SOC to float, rejecting NaN/Inf loudly (issue #26)."""
    soc = float(value)
    if not math.isfinite(soc):
        raise ValueError(f"battery_soc_pct must be finite, got {value!r}")
    return soc


class BatteryAgent(Agent):
    def __init__(self, agent_id, limits=None):
        super().__init__(agent_id)
        self.limits = BatteryLimits() if limits is None else limits
        self.battery_soc_pct = None
        self.battery_health_pct = None

    def get_supply_kw(self, time_delta_min=5.0):
        """Deliverable electrical kW, bounded by discharge rate and reserve."""
        dt_hours = _require_positive_timestep(time_delta_min)
        if self.battery_soc_pct is None:
            return 0.0
        soc = _require_finite_soc(self.battery_soc_pct)
        if soc <= self.limits.reserve_pct:
            return 0.0
        energy_kwh = (soc - self.limits.reserve_pct) / 100.0 * self.limits.capacity_kwh
        return min(self.limits.max_discharge_kw,
                   energy_kwh * self.limits.discharge_efficiency / dt_hours)

    def get_charge_headroom_kw(self, time_delta_min=5.0):
        """Absorbable electrical kW, bounded by charge rate and free capacity."""
        dt_hours = _require_positive_timestep(time_delta_min)
        if self.battery_soc_pct is None:
            return 0.0
        soc = _require_finite_soc(self.battery_soc_pct)
        if soc >= 100.0:
            return 0.0
        energy_kwh = (100.0 - soc) / 100.0 * self.limits.capacity_kwh
        return min(self.limits.max_charge_kw,
                   energy_kwh / self.limits.charge_efficiency / dt_hours)

    def update_state(self, row):
        # Defaults (50%/100%) keep legacy fake rows with missing keys
        # producing sane non-zero demand so old tests still exercise
        # the proportional allocator.
        self.battery_soc_pct = row.get("battery_soc_pct", 50.0)
        self.battery_health_pct = row.get("battery_health_pct", 100.0)

    def get_demand_kw(self):
        # Linear top-up demand: full 5 kW at 0% SOC, 0 kW at 100%.
        # Independent assumption (option (a)): generate_dataset.py has
        # NO max charge/discharge rate concept -- battery SOC there is
        # driven purely by surplus_kw = solar_kw - load_kw against
        # capacity_kwh (rng.uniform(10, 14), internal only, not exposed
        # in the returned tables), with no inverter/rate cap. So this
        # 5 kW is our own inverter-limit choice from a different
        # physical model, not derived from the generator. Health is
        # tracked but does not scale demand yet (out of scope).
        if self.battery_soc_pct is None:
            return 0.0
        gap = max(0.0, 100.0 - _require_finite_soc(self.battery_soc_pct))
        return BATTERY_MAX_CHARGE_KW * (gap / 100.0)

    def compute_urgency(self):
        if self.battery_soc_pct is None:
            return 0.0
        # Non-linear curve
        urgency = ((100.0 - _require_finite_soc(self.battery_soc_pct)) / 100.0) ** 2
        return max(0.0, min(1.0, urgency))
