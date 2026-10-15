from simulation.baseline_allocator import allocate
from simulation.run_simulation import run_simulation


def test_proportional_allocation():
    demands = {"battery_1": 2.0, "ev_1": 4.0, "appliance_1": 1.0}
    result = allocate(5.0, demands)

    assert round(result["battery_1"], 2) == 1.43
    assert round(result["ev_1"], 2) == 2.86
    assert round(result["appliance_1"], 2) == 0.71
    assert round(sum(result.values()), 2) == 5.00


def test_zero_demand():
    demands = {"battery_1": 0.0, "ev_1": 0.0}
    assert allocate(5.0, demands) == {"battery_1": 0, "ev_1": 0}


def test_surplus_capped_at_demand():
    # Solar exceeds total demand: each agent gets exactly what it asked
    # for, surplus curtailed, never fabricated as consumption (issue #5).
    demands = {"fridge": 1.0}
    result = allocate(5.0, demands)

    assert result == {"fridge": 1.0}
    assert sum(result.values()) == 1.0

    demands = {"a": 0.2, "b": 0.3}
    result = allocate(5.0, demands)

    assert result == {"a": 0.2, "b": 0.3}
    assert sum(result.values()) == 0.5


def test_simulation_runs_for_each_timestep():
    dataset = [{"solar_gen_kw": 5.0}, {"solar_gen_kw": 3.0}]
    results = run_simulation(dataset, allocate)

    assert len(results) == 2
    assert round(sum(results[0].values()), 2) == 5.00
    assert round(sum(results[1].values()), 2) == 3.00


def test_rejects_nonfinite_and_negative_inputs():
    """Issue #26: NaN/Inf/negative power must raise, not poison allocations."""
    import math

    import pytest

    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="available_power_kw"):
            allocate(bad, {"a": 1.0})
    for bad in (float("nan"), float("inf"), -2.5):
        with pytest.raises(ValueError, match="demand"):
            allocate(5.0, {"a": 1.0, "b": bad})


def test_total_demand_overflow_raises():
    """Issue #26: finite demands whose sum overflows to inf must raise."""
    import pytest

    with pytest.raises(ValueError, match="overflow"):
        allocate(5.0, {"a": 1e308, "b": 1e308})
