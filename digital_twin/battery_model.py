"""Stateful battery model with charge/discharge efficiency and thermal dynamics.

State:
- current SOC (%)
- capacity (kWh)
- temperature (°C)
- battery health (%)

Design assumptions (documented):
- Capacity: 12 kWh (midpoint of generator's 10-14 kWh range)
- Round-trip efficiency: 90% (charge_eff = discharge_eff = sqrt(0.90) ≈ 0.949)
  This is a conservative assumption for Li-ion. Real systems: 90-95% round-trip.
  We apply efficiency symmetrically: charging stores less, discharging
  requires more stored energy than delivered.
- Temperature model: Simple first-order thermal model.
  - Temperature rises under high charge/discharge power (I²R losses)
  - Temperature decays exponentially toward ambient
  - Not the Phase 5 PINN — this is a deliberately simplified approximation.
- Battery health: Constant for this version (no degradation modeling per day).

Energy conservation: The model never creates energy from nowhere.
SOC is clamped to [0, 100].
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatteryModelConfig:
    """Configuration for the battery model.

    Attributes:
        capacity_kwh: Usable battery capacity in kWh.
                      Default 12.0 (midpoint of generator's 10-14 range).
        charge_efficiency: Efficiency when charging (0-1).
                           Default sqrt(0.90) ≈ 0.949 for 90% round-trip.
        discharge_efficiency: Efficiency when discharging (0-1).
                              Default sqrt(0.90) ≈ 0.949.
        initial_soc_pct: Initial state of charge (%).
                         Default 50.0 (midpoint of generator's 35-65 range).
        initial_health_pct: Initial battery health (%).
                            Default 95.0 (midpoint of generator's 88-99 range).
        thermal_resistance: Thermal resistance (°C/kW) - how much temp rises
                            per kW of loss. Default 0.5 °C/kW.
        thermal_capacity: Thermal capacity (kWh/°C) - energy to raise temp 1°C.
                          Default 0.5 kWh/°C.
        ambient_temp_c: Ambient temperature (°C). Default 25.0.
        max_charge_rate_c: Maximum charge C-rate. Default 0.5C (6 kW for 12 kWh).
        max_discharge_rate_c: Maximum discharge C-rate. Default 1.0C (12 kW).
    """
    capacity_kwh: float = 12.0
    charge_efficiency: float = 0.949  # sqrt(0.90)
    discharge_efficiency: float = 0.949
    initial_soc_pct: float = 50.0
    initial_health_pct: float = 95.0
    thermal_resistance: float = 0.5  # °C/kW
    thermal_capacity: float = 0.5    # kWh/°C
    ambient_temp_c: float = 25.0
    max_charge_rate_c: float = 0.5
    max_discharge_rate_c: float = 1.0

    def __post_init__(self):
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if not 0 <= self.initial_soc_pct <= 100:
            raise ValueError("initial_soc_pct must be in [0, 100]")
        if not 0 <= self.initial_health_pct <= 100:
            raise ValueError("initial_health_pct must be in [0, 100]")


class BatteryModel:
    """Stateful battery model with efficiency and thermal dynamics."""

    def __init__(self, config: Optional[BatteryModelConfig] = None):
        """
        Args:
            config: BatteryModelConfig instance. If None, uses defaults.
        """
        self.config = config or BatteryModelConfig()
        self._soc_pct = self.config.initial_soc_pct
        self._health_pct = self.config.initial_health_pct
        self._temperature_c = self.config.ambient_temp_c
        self._ambient_temp_c = self.config.ambient_temp_c

    @property
    def soc_pct(self) -> float:
        """Current state of charge (%) clamped to [0, 100]."""
        return self._soc_pct

    @property
    def health_pct(self) -> float:
        """Current battery health (%).

        Note: In this version, health is constant (no daily degradation).
        """
        return self._health_pct

    @property
    def temperature_c(self) -> float:
        """Current battery temperature (°C)."""
        return self._temperature_c

    @property
    def capacity_kwh(self) -> float:
        """Usable battery capacity (kWh)."""
        return self.config.capacity_kwh

    @property
    def available_energy_kwh(self) -> float:
        """Energy currently stored in battery (kWh)."""
        return self._soc_pct / 100.0 * self.config.capacity_kwh

    @property
    def max_charge_power_kw(self) -> float:
        """Maximum charge power (kW) based on C-rate limit."""
        return self.config.capacity_kwh * self.config.max_charge_rate_c

    @property
    def max_discharge_power_kw(self) -> float:
        """Maximum discharge power (kW) based on C-rate limit."""
        return self.config.capacity_kwh * self.config.max_discharge_rate_c

    def update(
        self,
        charge_power_kw: float,
        time_delta_min: float,
        ambient_temp_c: Optional[float] = None,
    ) -> dict:
        """Update battery state for one timestep.

        Args:
            charge_power_kw: Positive = charging, negative = discharging (kW).
                             Will be clamped to C-rate limits.
            time_delta_min: Timestep duration in minutes.
            ambient_temp_c: Optional ambient temperature override (°C).
                            If None, uses config default.

        Returns:
            Dict with updated state:
                - "soc_pct": State of charge (%)
                - "health_pct": Battery health (%)
                - "temperature_c": Battery temperature (°C)
                - "stored_energy_kwh": Energy stored (kWh)
                - "charge_power_kw": Actual charge power applied (kW)
                - "energy_charged_kwh": Energy added to battery (kWh)
                - "energy_discharged_kwh": Energy removed from battery (kWh)
        """
        if ambient_temp_c is not None:
            self._ambient_temp_c = ambient_temp_c

        time_delta_h = time_delta_min / 60.0

        # Clamp power to C-rate limits
        max_charge = self.max_charge_power_kw
        max_discharge = self.max_discharge_power_kw
        charge_power_kw = max(-max_discharge, min(max_charge, charge_power_kw))

        energy_charged_kwh = 0.0
        energy_discharged_kwh = 0.0
        soc_before = self._soc_pct

        if charge_power_kw > 0:
            # Charging: electrical energy in -> stored energy (with loss)
            electrical_energy_in_kwh = charge_power_kw * time_delta_h
            stored_energy_gained_kwh = electrical_energy_in_kwh * self.config.charge_efficiency

            new_soc = self._soc_pct + (stored_energy_gained_kwh / self.config.capacity_kwh) * 100.0
            self._soc_pct = min(100.0, new_soc)
            # Report ACTUAL stored energy (accounts for the 100% clamp).
            energy_charged_kwh = (self._soc_pct - soc_before) / 100.0 * self.config.capacity_kwh

        elif charge_power_kw < 0:
            # Discharging: stored energy -> electrical energy out (with loss)
            discharge_power_kw = -charge_power_kw
            electrical_energy_out_kwh = discharge_power_kw * time_delta_h
            stored_energy_lost_kwh = electrical_energy_out_kwh / self.config.discharge_efficiency

            new_soc = self._soc_pct - (stored_energy_lost_kwh / self.config.capacity_kwh) * 100.0
            self._soc_pct = max(0.0, new_soc)
            # Report ACTUAL stored energy removed (accounts for the 0% clamp).
            energy_discharged_kwh = (soc_before - self._soc_pct) / 100.0 * self.config.capacity_kwh

        # Thermal model: first-order lumped model (deliberately simplified,
        # NOT the Phase 5 PINN). Temperature rises due to I2R-style losses and
        # decays exponentially toward ambient with time constant
        # tau = thermal_capacity * thermal_resistance (hours).
        # Exact discrete update for constant loss over the step:
        #   T_new = T_amb + (T - T_amb) * exp(-dt/tau)
        #                 + (loss * R_thermal) * (1 - exp(-dt/tau))
        # Power loss = |charge_power| * (1 - efficiency) during charge/discharge.
        if charge_power_kw > 0:
            loss_power_kw = charge_power_kw * (1.0 - self.config.charge_efficiency)
        elif charge_power_kw < 0:
            loss_power_kw = (-charge_power_kw) * (1.0 - self.config.discharge_efficiency)
        else:
            loss_power_kw = 0.0

        steady_rise_c = loss_power_kw * self.config.thermal_resistance

        tau_h = self.config.thermal_capacity * self.config.thermal_resistance
        if tau_h > 0:
            import math as _math
            exp_factor = _math.exp(-time_delta_h / tau_h)
        else:
            exp_factor = 0.0

        temp_excess = self._temperature_c - self._ambient_temp_c
        self._temperature_c = (
            self._ambient_temp_c
            + temp_excess * exp_factor
            + steady_rise_c * (1.0 - exp_factor)
        )

        # Clamp temperature to reasonable bounds
        self._temperature_c = max(-20.0, min(60.0, self._temperature_c))

        return {
            "soc_pct": self._soc_pct,
            "health_pct": self._health_pct,
            "temperature_c": self._temperature_c,
            "stored_energy_kwh": self.available_energy_kwh,
            "charge_power_kw": charge_power_kw,
            "energy_charged_kwh": energy_charged_kwh,
            "energy_discharged_kwh": energy_discharged_kwh,
        }

    def reset(self, soc_pct: Optional[float] = None,
              health_pct: Optional[float] = None,
              temperature_c: Optional[float] = None) -> None:
        """Reset battery to initial or specified state."""
        self._soc_pct = soc_pct if soc_pct is not None else self.config.initial_soc_pct
        self._health_pct = health_pct if health_pct is not None else self.config.initial_health_pct
        self._temperature_c = temperature_c if temperature_c is not None else self.config.ambient_temp_c
        self._ambient_temp_c = self.config.ambient_temp_c