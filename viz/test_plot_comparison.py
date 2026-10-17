"""Verify the figure against independent closed-loop runs, headlessly."""

from copy import deepcopy

import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import requests

from scenarios import build_twin
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.hybrid_arbiter import hybrid_allocate
import viz.plot_comparison as module


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("plot attempted network access")
    monkeypatch.setattr(requests.sessions.Session, "request", fail)


@pytest.mark.parametrize("scenario", ["clear_day", "overnight_ev_cloudy"])
def test_png(scenario, tmp_path):
    path = tmp_path / f"{scenario}.png"
    module.plot_comparison(scenario, path)
    assert matplotlib.get_backend().lower() == "agg"
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 10000
    assert not plt.get_fignums()


@pytest.mark.parametrize("scenario", ["clear_day", "overnight_ev_cloudy"])
def test_series_and_full_interval_shading(monkeypatch, scenario):
    runs = [run_closed_loop(build_twin(scenario, closed_loop=True), allocator)
            for allocator in (hybrid_allocate, proportional_adapter)]
    before = deepcopy(runs)

    def inspect(fig, *args, **kwargs):
        axes = fig.axes
        edges = np.r_[0, np.cumsum([r["dt_min"] for r in runs[0]])]
        np.testing.assert_array_equal(axes[0].patches[0].get_data().values,
                                      [r["solar_kw"] for r in runs[0]])
        for i, records in enumerate(runs):
            delivered = [sum(r["appliance_served_kw"].values()) + r["ev_applied_kw"]
                         + r["agents"]["household_base_load"]["allocated_kw"] for r in records]
            np.testing.assert_array_equal(axes[0].patches[i+1].get_data().values, delivered)
            np.testing.assert_array_equal(axes[1].lines[i].get_xdata(), edges[1:])
            np.testing.assert_array_equal(axes[1].lines[i].get_ydata(),
                                          [r["battery_soc_pct"] for r in records])
            expected_ev = [r["ev_soc_pct"] if r["ev_plugged_in"] else np.nan for r in records]
            np.testing.assert_allclose(axes[2].lines[i].get_ydata(), expected_ev, equal_nan=True)
            for j, critical in enumerate((True, False)):
                expected = [sum(max(0, a["demand_kw"]-a["allocated_kw"])
                                for name, a in r["agents"].items()
                                if name not in {"battery_1", "solar_1"}
                                and (r["priorities"][name] <= 2) == critical)
                            * r["dt_min"]/60 for r in records]
                np.testing.assert_allclose(axes[3].patches[2*i+j].get_data().values, expected)
        plugged = np.array([r["ev_plugged_in"] for r in runs[0]])
        path = axes[2].patches[0].get_path()
        for offset in (0.1, 4.9):
            points = np.column_stack((edges[:-1]+offset, np.full(len(plugged), 50)))
            np.testing.assert_array_equal(path.contains_points(points), plugged)
        departures = [edges[i+1] for i, r in enumerate(runs[0]) if r["ev_departures"]]
        assert [line.get_xdata()[0] for line in axes[2].lines[2:]] == departures

    monkeypatch.setattr(Figure, "savefig", inspect)
    module.plot_comparison(scenario)
    assert runs == before
    assert not plt.get_fignums()


def test_factory_and_save_failure_cleanup(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        module.plot_comparison(lambda: build_twin("clear_day", closed_loop=True))
    assert not plt.get_fignums()


def test_reject_reused_factory_twin():
    twin = build_twin("clear_day", closed_loop=True)
    with pytest.raises(ValueError, match="separate"):
        module.plot_comparison(lambda: twin)
