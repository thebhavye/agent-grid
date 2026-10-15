"""Deterministic checks for the shared battery mode selector."""

from itertools import product

import pytest

from simulation.battery_mode import choose_battery_mode


@pytest.mark.parametrize("inputs,mode,demand,supply", [
    ((10, 4, 5, 12, 6), "charge", 5, 0),
    ((10, 4, 5, 12, 2), "charge", 2, 0),
    ((4, 4, 5, 12, 2), "charge", 2, 0),
    ((10, 4, 0, 12, 6), "idle", 0, 0),
    ((10, 4, 5, 12, 0), "idle", 0, 0),
    ((1, 4, 5, 12, 6), "discharge", 0, 3),
    ((1, 4, 5, 2, 6), "discharge", 0, 2),
    ((1, 4, 5, 0, 6), "idle", 0, 0),
    ((0, 0, 0, 0, 0), "idle", 0, 0),
])
def test_mode_selection(inputs, mode, demand, supply):
    result = choose_battery_mode(*inputs)
    assert result == {"mode": mode, "battery_demand_kw": demand,
                      "battery_supply_kw": supply}
    assert isinstance(result["battery_demand_kw"], float)
    assert isinstance(result["battery_supply_kw"], float)


@pytest.mark.parametrize("name", ["solar_kw", "other_demand_kw", "battery_demand_kw",
                                  "supply_kw", "headroom_kw"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1])
@pytest.mark.parametrize("solar", [0, 10])
def test_mode_rejects_every_invalid_input_even_when_unused(name, value, solar):
    inputs = dict(solar_kw=solar, other_demand_kw=5, battery_demand_kw=2,
                  supply_kw=3, headroom_kw=4)
    inputs[name] = value
    with pytest.raises(ValueError, match=name):
        choose_battery_mode(**inputs)


def test_grid_one_mode_caps_and_purity():
    for inputs in product((0.0, 0.1, 2.0, 10.0), repeat=5):
        solar, other, battery, supply, headroom = inputs
        result = choose_battery_mode(*inputs)
        assert set(result) == {"mode", "battery_demand_kw", "battery_supply_kw"}
        demand = result["battery_demand_kw"]
        delivered = result["battery_supply_kw"]
        assert isinstance(demand, float) and isinstance(delivered, float)
        assert demand == 0 or delivered == 0
        if solar >= other:
            assert demand == min(battery, headroom)
            assert delivered == 0
            assert result["mode"] == ("charge" if demand else "idle")
        else:
            assert demand == 0
            assert delivered == min(supply, other - solar)
            assert result["mode"] == ("discharge" if delivered else "idle")
        assert choose_battery_mode(*inputs) == result
