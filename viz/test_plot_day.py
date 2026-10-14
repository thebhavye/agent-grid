"""Headless smoke tests for the day visualization."""

import matplotlib
from matplotlib.figure import Figure
import numpy as np
import pytest

from data.generate_dataset import STEP_MIN, generate_dataset
import viz.plot_day as plot_module
from viz.plot_day import plot_day


@pytest.mark.parametrize("seed", [42, 7])
def test_plot_day(seed, tmp_path):
    output = tmp_path / f"day_{seed}.png"
    try:
        plot_day(seed=seed, save_path=output)
        assert matplotlib.get_backend().lower() == "agg"
        assert output.is_file()
        assert output.stat().st_size > 0
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        output.unlink(missing_ok=True)


@pytest.mark.parametrize("plugged_steps", [[], [0], [10], [10, 11, 12], [0, 1, 10, 11, 287]])
def test_plug_in_shading_covers_full_intervals(monkeypatch, plugged_steps):
    dataset = generate_dataset(seed=42)
    plugged = np.zeros(len(dataset["timeseries"]), dtype=bool)
    plugged[plugged_steps] = True
    dataset["timeseries"]["ev_plugged_in"] = plugged
    monkeypatch.setattr(plot_module, "generate_dataset", lambda seed: dataset)

    def check_shading(fig, *args, **kwargs):
        ax = fig.axes[2]
        paths = [patch.get_path() for patch in ax.patches]
        paths += [path for collection in ax.collections for path in collection.get_paths()]
        # Check both ends of every interval, including just before unplugging.
        for offset in (0.1, STEP_MIN - 0.1):
            points = np.column_stack((np.arange(len(plugged)) * STEP_MIN + offset,
                                      np.full(len(plugged), 50)))
            shaded = np.zeros(len(plugged), dtype=bool)
            for path in paths:
                shaded |= path.contains_points(points)
            np.testing.assert_array_equal(shaded, plugged)

    monkeypatch.setattr(Figure, "savefig", check_shading)
    plot_day()
