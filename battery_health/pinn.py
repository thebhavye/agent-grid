"""Small CPU network with empirical aging and first-order thermal residuals.

These are simplified twin relations, not validated electrochemical physics.
Torch is imported only when this optional module is explicitly requested.
"""

import numpy as np
import torch
from torch import nn

INPUTS = ("soc_pct", "c_rate", "throughput_kwh", "temperature_c", "ambient_temp_c",
          "health_pct", "power_kw", "dt_min")
SCALES = np.array([100, 1, 100, 50, 50, 100, 12, 5], dtype=np.float32)
LOSS_SCALE = 0.0001


class BatteryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(INPUTS), 24), nn.Tanh(),
                                     nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 2))

    def forward(self, inputs):
        return self.network(inputs)

    def predict(self, frame):
        with torch.no_grad():
            delta = self(_inputs(frame)).numpy()
        return np.column_stack((frame.health_pct - delta[:, 0] * LOSS_SCALE,
                                frame.temperature_c + delta[:, 1]))


def _inputs(frame):
    return torch.tensor(frame[list(INPUTS)].to_numpy(dtype=np.float32) / SCALES)


def train(frame, *, seed=42, epochs=400, physics_weight=1.0):
    """Fixed-seed full-batch training; zero physics weight gives the ablation."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = BatteryPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    inputs = _inputs(frame)
    target = torch.tensor(np.column_stack(((frame.health_pct - frame.next_health_pct) / LOSS_SCALE,
                                          frame.next_temperature_c - frame.temperature_c)), dtype=torch.float32)
    physics = torch.tensor(np.column_stack((frame.physics_loss_pct / LOSS_SCALE,
                                           frame.physics_temp_delta_c)), dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model(inputs)
        data_loss = (predicted - target).square().mean()
        residual = (predicted - physics).square().mean()
        monotonic = torch.relu(-predicted[:, 0]).square().mean()
        loss = data_loss + physics_weight * (residual + monotonic)
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
