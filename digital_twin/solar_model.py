"""Physically grounded solar generation model.

Base formula:
    power_kw = irradiance_w_m2 * panel_area_m2 * panel_efficiency / 1000

Temperature derating:
    -0.4% / °C above 25°C (standard temperature coefficient for crystalline silicon)

This is a simplified model. Real systems have additional losses:
- Inverter efficiency (~95-98%)
- Wiring losses
- Soiling/dirt on panels
- Spectral mismatch
- Angle of incidence effects (beyond cosine)
- Module mismatch

These are NOT modeled here. The model is intentionally simplified for the
digital twin's Phase 1 scope.

Defaults (documented assumptions):
- Panel area: 20 m² (typical residential ~3-4 kWp system at ~18-20% efficiency)
- Panel efficiency: 0.20 (20% - mid-range for modern monocrystalline)
- Temperature coefficient: -0.004 / °C (-0.4%/°C, standard for c-Si)
- Reference temperature: 25°C (STC)

Source for temperature coefficient: IEC 61724 / NREL PVWatts standard
assumption for crystalline silicon modules.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SolarModelConfig:
    """Configuration for the solar model.

    Attributes:
        panel_area_m2: Total panel area in square meters.
                       Default 20 m² ≈ 4 kWp at 20% efficiency.
        panel_efficiency: Panel conversion efficiency (0-1).
                          Default 0.20 (20%).
        temp_coeff_per_c: Temperature coefficient per °C.
                          Default -0.004 (-0.4%/°C).
        reference_temp_c: Reference temperature for STC rating.
                          Default 25°C.
    """
    panel_area_m2: float = 20.0
    panel_efficiency: float = 0.20
    temp_coeff_per_c: float = -0.004
    reference_temp_c: float = 25.0

    def __post_init__(self):
        if self.panel_area_m2 <= 0:
            raise ValueError("panel_area_m2 must be positive")
        if not 0 < self.panel_efficiency <= 1:
            raise ValueError("panel_efficiency must be in (0, 1]")
        if self.temp_coeff_per_c >= 0:
            raise ValueError("temp_coeff_per_c should be negative")


class SolarModel:
    """Solar PV generation model with temperature derating."""

    def __init__(self, config: Optional[SolarModelConfig] = None):
        """
        Args:
            config: SolarModelConfig instance. If None, uses defaults.
        """
        self.config = config or SolarModelConfig()

    def calculate_power_kw(
        self,
        irradiance_w_m2: float,
        temperature_c: float,
    ) -> float:
        """Calculate DC power output in kW.

        Args:
            irradiance_w_m2: Solar irradiance in W/m² (global horizontal
                             or plane-of-array; we assume plane-of-array
                             for simplicity).
            temperature_c: Cell temperature in °C. If ambient temperature
                           is known instead, a rough approximation is
                           cell_temp ≈ ambient + (irradiance/800)*25
                           (NOCT approximation), but this model expects
                           cell temperature directly.

        Returns:
            Power output in kW (never negative).
        """
        if irradiance_w_m2 <= 0:
            return 0.0

        # Base power at STC conditions
        power_kw = (
            irradiance_w_m2
            * self.config.panel_area_m2
            * self.config.panel_efficiency
            / 1000.0
        )

        # Temperature derating
        temp_delta = temperature_c - self.config.reference_temp_c
        if temp_delta > 0:
            derating_factor = 1.0 + self.config.temp_coeff_per_c * temp_delta
            # Clamp derating factor to reasonable bounds (max 30% loss)
            derating_factor = max(0.7, min(1.0, derating_factor))
            power_kw *= derating_factor

        return max(0.0, power_kw)

    def calculate_power_kw_from_ambient(
        self,
        irradiance_w_m2: float,
        ambient_temp_c: float,
    ) -> float:
        """Calculate power using ambient temperature with NOCT approximation.

        Uses the Nominal Operating Cell Temperature (NOCT) approximation:
            T_cell = T_ambient + (irradiance / 800) * (NOCT - 20)

        Where NOCT is typically 45°C for rooftop mounting.

        Args:
            irradiance_w_m2: Solar irradiance in W/m².
            ambient_temp_c: Ambient air temperature in °C.

        Returns:
            Power output in kW.
        """
        if irradiance_w_m2 <= 0:
            return 0.0

        # NOCT approximation: cell temp rises above ambient under irradiance
        # NOCT = 45°C at 800 W/m², 20°C ambient, 1 m/s wind
        noct_c = 45.0
        cell_temp_c = ambient_temp_c + (irradiance_w_m2 / 800.0) * (noct_c - 20.0)

        return self.calculate_power_kw(irradiance_w_m2, cell_temp_c)