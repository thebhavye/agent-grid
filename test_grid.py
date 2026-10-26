"""Synthetic tariffs, pure settlement and actual closed-loop accounting."""

from copy import deepcopy
from dataclasses import replace

import pytest

from orchestration.graph import run_graph_closed_loop
from scenarios import build_twin
from simulation.closed_loop import proportional_adapter, run_closed_loop
from simulation.compare import _make_twin
from simulation.grid import GridConfig, settle_grid
from simulation.hybrid_arbiter import hybrid_allocate
from simulation.metrics import METRICS, score_run
from test_metrics import record


def band(start=0, end=1440, buy=2.0, sell=0.5):
    return dict(start_min=start, end_min=end,
                import_price_per_kwh=buy, export_price_per_kwh=sell)


def config(**kwargs):
    return GridConfig(enabled=True, import_limit_kw=3, export_limit_kw=2,
                      tariff=[band()], **kwargs)


def test_caps_costs_floats_and_non_five_minute():
    cfg = config()
    assert settle_grid(4, 0, 0, 15, cfg) == {
        "import_kw": 3.0, "export_kw": 0.0, "import_cost": 1.5, "export_credit": 0.0}
    assert settle_grid(0, 5, 1439, 30, cfg) == {
        "import_kw": 0.0, "export_kw": 2.0, "import_cost": 0.0, "export_credit": 0.5}
    assert all(isinstance(x, float) for x in settle_grid(0, 0, 0, 5, cfg).values())
    assert settle_grid(1, 0, 0, 5, GridConfig()) == dict.fromkeys(
        ["import_kw", "export_kw", "import_cost", "export_credit"], 0.0)


def test_midnight_wrap_boundary_pricing_and_no_input_mutation():
    tariff = [band(1200, 360, 1), band(360, 1200, 4)]
    before = deepcopy(tariff)
    cfg = GridConfig(enabled=True, import_limit_kw=1, tariff=tariff)
    assert [settle_grid(1, 0, m, 60, cfg)["import_cost"]
            for m in (0, 359, 360, 1199, 1200, 1439)] == [1, 1, 4, 4, 1, 1]
    assert tariff == before
    tariff[0]["import_price_per_kwh"] = 100
    assert settle_grid(1, 0, 0, 60, cfg)["import_cost"] == 1


@pytest.mark.parametrize("tariff", [[], [band(0, 100)], [band(0, 800), band(700, 1440)],
                                    [band(0, 700), band(800, 1440)], [band(5, 5)],
                                    [band(-1, 1440)], [band(0, 1441)], [{}], None])
def test_bad_tariffs(tariff):
    with pytest.raises(ValueError):
        GridConfig(enabled=True, tariff=tariff)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), None])
@pytest.mark.parametrize("field", ["import_limit_kw", "export_limit_kw"])
def test_invalid_limits(field, value):
    with pytest.raises(ValueError):
        GridConfig(**{field: value})


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), None])
@pytest.mark.parametrize("field", ["import_price_per_kwh", "export_price_per_kwh"])
def test_invalid_prices(field, value):
    tariff = band()
    tariff[field] = value
    with pytest.raises(ValueError):
        GridConfig(enabled=True, tariff=[tariff])


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), None])
def test_invalid_settlement_input(index, value):
    args = [1, 0, 0, 5]
    args[index] = value
    with pytest.raises(ValueError):
        settle_grid(*args, config())


def test_invalid_timestep_minute_and_simultaneous_flows():
    for args in [(1, 0, 1440, 5), (1, 0, 0, 0), (1, 1, 0, 5)]:
        with pytest.raises(ValueError):
            settle_grid(*args, config())


@pytest.mark.parametrize("scenario", ["overnight", "midday", "evening_topup"])
@pytest.mark.parametrize("allocator", [hybrid_allocate, proportional_adapter])
def test_disabled_grid_keeps_exact_records_and_metrics(scenario, allocator):
    original = run_closed_loop(_make_twin(42, scenario), allocator)
    disabled = run_closed_loop(_make_twin(42, scenario), allocator, grid_config=GridConfig())
    assert original == disabled
    assert score_run(original) == score_run(disabled)
    assert set(score_run(disabled)) == set(METRICS)
    assert all("grid_import_kw" not in r for r in disabled)


@pytest.mark.parametrize("scenario", ["demo_sunny_home", "overnight_ev_cloudy"])
@pytest.mark.parametrize("allocator", [hybrid_allocate, proportional_adapter])
def test_full_day_grid_energy_and_caps(scenario, allocator):
    records = run_closed_loop(build_twin(scenario, closed_loop=True), allocator,
                              grid_config=config())
    score = score_run(records)
    assert score["energy_balance_max_error_kw"] < 1e-9
    assert score["battery_reserve_violation_steps"] == 0
    assert score["grid_import_kwh"] > 0
    for r in records:
        assert 0 <= r["grid_import_kw"] <= 3
        assert 0 <= r["grid_export_kw"] <= 2
        assert not (r["grid_import_kw"] > 1e-9 and r["battery_kw"] > 1e-9)
        assert not (r["grid_export_kw"] > 1e-9 and r["battery_kw"] < -1e-9)
        assert all(0 <= a["allocated_kw"] <= a["demand_kw"] + 1e-9
                   for a in r["agents"].values())
    if scenario == "demo_sunny_home":
        assert score["grid_export_kwh"] > 0


def test_limited_import_serves_priority_one_before_ev():
    twin = build_twin("overnight_ev_cloudy", closed_loop=True)
    twin._battery._soc_pct = 20
    records = run_closed_loop(twin, hybrid_allocate, steps=1,
                              grid_config=replace(config(), import_limit_kw=0.6))
    row = records[0]
    assert row["grid_import_kw"] == pytest.approx(0.6)
    assert row["agents"]["household_base_load"]["allocated_kw"] > 0
    assert row["agents"]["ev_1"]["allocated_kw"] == 0
    assert row["battery_kw"] == 0
    assert row["agents"]["battery_1"]["allocated_kw"] == 0


def test_graph_grid_parity():
    def factory():
        return build_twin("clear_day", closed_loop=True)
    assert run_graph_closed_loop(factory(), hybrid_allocate, grid_config=config()) == (
        run_closed_loop(factory(), hybrid_allocate, grid_config=config()))


def test_known_metrics_non_five_minute_and_zero_solar():
    row = record(solar_kw=4, battery_kw=1, curtailed_kw=1,
                 grid_import_kw=1, grid_export_kw=0.5, import_cost=0.5, export_credit=0.1)
    before = deepcopy(row)
    score = score_run([row])
    assert score["grid_import_kwh"] == 0.25
    assert score["grid_export_kwh"] == 0.125
    assert score["net_cost"] == pytest.approx(0.4)
    assert score["self_consumption_pct"] == 62.5
    assert score["energy_balance_max_error_kw"] == 0
    assert row == before
    assert score_run([record(solar_kw=0, battery_kw=-3, curtailed_kw=0.5,
                             grid_import_kw=0.0, grid_export_kw=0.0,
                             import_cost=0.0, export_credit=0.0)])["self_consumption_pct"] == 0
