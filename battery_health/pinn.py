"""Optional daily PINN using #69's dataset, targets and operating exposures."""

import numpy as np
import torch
from torch import nn

from battery_health.dataset import summarize_days
from digital_twin.battery_model import BatteryModelConfig

INPUTS = ("start_health_pct", "start_temperature_c", "mean_ambient_c", "max_c_rate",
          "cycle_exposure", "calendar_exposure")
SCALES = np.array([100, 50, 50, 1, 100, 1], dtype=np.float32)
LOSS_SCALE = .01


def daily_frame(steps):
    """Use existing daily targets; physics targets use known operation only.

    Both models are conditional fits on the day's operating exposures, not
    forecasts of unknown usage. Thermal physics replays the existing first-
    order recurrence from initial temperature, power and ambient readings.
    """
    daily = summarize_days(steps)
    cfg = BatteryModelConfig()
    peaks = {}
    for (scenario, seed), rows in steps.groupby(["scenario", "seed"], sort=True):
        temperature = float(rows.iloc[0].initial_temperature_c)
        for day, group in rows.groupby("day", sort=True):
            peak = -float("inf")
            for row in group.itertuples():
                efficiency = cfg.charge_efficiency if row.power_kw >= 0 else cfg.discharge_efficiency
                decay = np.exp(-row.dt_min / 60 / (cfg.thermal_capacity * cfg.thermal_resistance))
                temperature = np.clip(row.ambient_temperature_c
                    + (temperature - row.ambient_temperature_c) * decay
                    + abs(row.power_kw) * (1 - efficiency) * cfg.thermal_resistance * (1 - decay), -20, 60)
                peak = max(peak, temperature)
            peaks[scenario, seed, day] = peak
    daily["physics_peak_c"] = [peaks[r.scenario, r.seed, r.day] for r in daily.itertuples()]
    daily["physics_loss_pct"] = (daily.cycle_exposure * cfg.cycle_fade_pct_per_kwh
                                  + daily.calendar_exposure * cfg.calendar_fade_pct_per_day)
    return daily


class BatteryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(INPUTS), 24), nn.Tanh(),
                                     nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 2))

    def forward(self, inputs):
        return self.network(inputs)

    def predict(self, frame):
        with torch.no_grad():
            output = self(_inputs(frame)).numpy()
        return np.column_stack((frame.start_health_pct - output[:, 0] * LOSS_SCALE,
                                output[:, 1] * 50))


def _inputs(frame):
    return torch.tensor(frame[list(INPUTS)].to_numpy(dtype=np.float32) / SCALES)


def train(frame, *, seed=42, epochs=400, physics_weight=1.0):
    """Train on daily rows; zero physics weight gives the same-network ablation."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = BatteryPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    inputs = _inputs(frame)
    target = torch.tensor(np.column_stack(((frame.start_health_pct - frame.end_health_pct) / LOSS_SCALE,
                                          frame.peak_temperature_c / 50)), dtype=torch.float32)
    physics = torch.tensor(np.column_stack((frame.physics_loss_pct / LOSS_SCALE,
                                           frame.physics_peak_c / 50)), dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model(inputs)
        loss = (predicted - target).square().mean() + physics_weight * (
            (predicted - physics).square().mean() + torch.relu(-predicted[:, 0]).square().mean())
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def thermal_risk(temperature_c, previous_temperature_c, dt_min,
                 *, threshold_c=45.0, rise_threshold_c_per_min=1.0):
    """Demonstration thresholds, not battery manufacturer safety limits."""
    values = (temperature_c, previous_temperature_c, dt_min, threshold_c, rise_threshold_c_per_min)
    if not all(np.isfinite(v) for v in values) or dt_min <= 0:
        raise ValueError("thermal-risk inputs must be finite with positive dt_min")
    flags = []
    if temperature_c >= threshold_c:
        flags.append("high_temperature")
    if (temperature_c - previous_temperature_c) / dt_min >= rise_threshold_c_per_min:
        flags.append("rapid_temperature_rise")
    return flags
