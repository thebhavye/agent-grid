"""Graph topology and exact parity against the measured closed-loop runner."""

import socket
import subprocess
import sys

import pytest
import requests

from digital_twin.appliance_model import ApplianceModelConfig, ApplianceSpec
from digital_twin.ev_model import EVModelConfig, EVScenario
from digital_twin.faults import DeviceFaultConfig, FaultConfig
from digital_twin.test_digital_twin import make_test_twin
from orchestration.graph import build_step_graph, run_graph_closed_loop
from simulation.closed_loop import (
    _build_closed_loop_agents,
    _build_priorities,
    proportional_adapter,
    run_closed_loop,
)
from simulation.hybrid_arbiter import hybrid_allocate


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("graph attempted network access")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("scenario", list(EVScenario))
@pytest.mark.parametrize("allocator", [hybrid_allocate, proportional_adapter])
def test_records_identical(seed, scenario, allocator):
    def twin():
        return make_test_twin(seed=seed, closed_loop=True,
                              ev_config=EVModelConfig(seed=seed, scenario=scenario))
    assert run_graph_closed_loop(twin(), allocator) == run_closed_loop(twin(), allocator)


def graph_for(twin):
    return build_step_graph(_build_closed_loop_agents(twin), _build_priorities(twin),
                            hybrid_allocate).get_graph()


def test_graph_structure_independent_bidding_and_join():
    twin = make_test_twin(closed_loop=True)
    graph = graph_for(twin)
    bids = {f"bid_{agent.agent_id}" for agent in _build_closed_loop_agents(twin)}
    assert set(graph.nodes) == bids | {"__start__", "__end__", "battery_mode", "arbiter", "commit"}
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert edges == ({("__start__", bid) for bid in bids}
                     | {(bid, "battery_mode") for bid in bids}
                     | {("battery_mode", "arbiter"), ("arbiter", "commit"), ("commit", "__end__")})


def test_roster_add_remove_changes_one_bid_node():
    roster = ApplianceModelConfig().appliances
    original = set(graph_for(make_test_twin(closed_loop=True)).nodes)
    extended = set(graph_for(make_test_twin(closed_loop=True, appliance_config=
                   ApplianceModelConfig(appliances=roster + [ApplianceSpec("fan", 3, False, 0.1)]))).nodes)
    reduced = set(graph_for(make_test_twin(closed_loop=True, appliance_config=
                  ApplianceModelConfig(appliances=roster[1:]))).nodes)
    assert extended - original == {"bid_appliance_fan"}
    assert original - reduced == {"bid_appliance_fridge"}
    assert not original - extended and not reduced - original


def test_fault_parity_and_no_remote_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    options = dict(on_bad_reading="hold_last", sensor_faults=FaultConfig(enabled=True, faults=[
        {"type": "dropout", "key": "battery_soc_pct", "start_step": 13, "end_step": 18},
    ]), device_faults=DeviceFaultConfig(enabled=True, faults=[
        {"type": "battery_derate", "start_step": 10, "end_step": 15, "factor": 0.1},
    ]))
    assert run_graph_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate, steps=20,
                                 **options) == run_closed_loop(
        make_test_twin(closed_loop=True), hybrid_allocate, steps=20, **options)


def test_deterministic_reset_and_no_stale_bid_accumulation():
    twin = make_test_twin(closed_loop=True)
    first = run_graph_closed_loop(twin, hybrid_allocate, steps=5)
    twin.reset()
    assert first == run_graph_closed_loop(twin, hybrid_allocate, steps=5)
    assert all(len(row["agents"]) == len(first[0]["agents"]) for row in first)


def test_core_imports_do_not_import_langgraph():
    result = subprocess.run([sys.executable, "-c", "import simulation.closed_loop, sys; "
                             "assert not any(k.startswith('langgraph') for k in sys.modules)"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_invalid_steps_rejected(steps):
    with pytest.raises(ValueError, match="steps"):
        run_graph_closed_loop(make_test_twin(closed_loop=True), hybrid_allocate, steps=steps)
