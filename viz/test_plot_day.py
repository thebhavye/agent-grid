"""Headless smoke tests for the day visualization."""

import matplotlib
import pytest

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
