"""Issue #11: overnight EV regression checks must never silently skip.

Uses the test-only forcing helper from data.test_generate_dataset so the
overnight assertions run unconditionally, regardless of which EV scenario
seed=42 happens to produce.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from data.generate_dataset import N_STEPS
from data.test_generate_dataset import _generate_overnight_tables

REPO_ROOT = Path(__file__).resolve().parent


def _transition_indices(ots):
    parsed = ots["timestamp"].apply(lambda s: int(s[:2]) * 60 + int(s[3:5]))
    dep = int(ots["ev_departure_min"].iloc[0])
    return ots, int(np.searchsorted(parsed.to_numpy(), dep, side="left"))


def test_forced_overnight_transition_before_and_after():
    ots = _generate_overnight_tables(seed=1)["timeseries"]
    assert bool(ots["ev_plugged_in"].iloc[0]), "forced tables must be overnight"

    ots, after_dep_idx = _transition_indices(ots)
    assert after_dep_idx == 0 or bool(ots["ev_plugged_in"].iloc[after_dep_idx - 1]), (
        "EV must be plugged in on the step just before departure_min"
    )
    assert after_dep_idx >= N_STEPS or not bool(ots["ev_plugged_in"].iloc[after_dep_idx]), (
        "EV must be unplugged on the step at/after departure_min"
    )


def test_forced_overnight_is_seed_independent():
    # Forcing must win for every seed; if the forcing setup were removed,
    # raw seeds vary by luck and this fails loudly instead of skipping.
    for seed in (0, 1, 2, 42, 123):
        ots = _generate_overnight_tables(seed=seed)["timeseries"]
        assert bool(ots["ev_plugged_in"].iloc[0]), f"seed {seed} not forced overnight"


def test_script_always_reports_overnight_checks():
    # Tripwire against reverting to `if plugged_in[0]` seed-42 luck: the
    # script must print both regression checks as executed (PASS).
    proc = subprocess.run(
        [sys.executable, "-m", "data.test_generate_dataset"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in (
        "overnight scenario: EV is plugged in on the step just before departure_min",
        "overnight scenario: EV is unplugged on the step at/after departure_min",
    ):
        assert f"[PASS] {name}" in proc.stdout, (
            f"overnight check did not execute: {name!r}\n{proc.stdout}"
        )
