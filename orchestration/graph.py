"""Real LangGraph step execution over the shared closed-loop phases."""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from simulation.closed_loop import (
    _allocate_step,
    _bids_for_observation,
    _commit_step,
    _prepare_step,
    _run_context,
    _select_battery_mode,
)


class StepState(TypedDict):
    """Public step results plus run-local context (no checkpoint or network)."""

    obs: dict
    bids: Annotated[dict[str, dict], operator.or_]
    battery_mode: dict
    available_kw: float
    allocation: dict
    delivered: dict
    record: dict
    context: dict
    data: dict


def build_step_graph(agents, priorities, allocator):
    """Compile independent bid nodes, a join, battery mode, arbiter and commit.

    Invoke with obs (the sanitized observation), data from _prepare_step, and
    context from _run_context. run_graph_closed_loop supplies these normally.
    Each bid node owns one agent; merging is by ID and summation uses the
    original agent order, independent of worker completion order.
    """
    agents = list(agents)
    ids = [agent.agent_id for agent in agents]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("graph requires non-empty agents with unique IDs")
    builder = StateGraph(StepState)

    def bidder(agent):
        def bid(state):
            result = _bids_for_observation(
                [agent], state["obs"], state["data"]["excluded"]
            )[0]
            return {"bids": {agent.agent_id: result}}
        return bid

    for agent in agents:
        name = f"bid_{agent.agent_id}"
        builder.add_node(name, bidder(agent))
        builder.add_edge(START, name)

    def battery_mode(state):
        data = dict(state["data"])
        bids = [state["bids"][agent_id] for agent_id in ids]
        _select_battery_mode(state["context"], data, bids)
        return {"data": data, "battery_mode": data["mode"],
                "available_kw": data["available_kw"]}

    def arbiter(state):
        data = dict(state["data"])
        context = {**state["context"], "allocator": allocator, "priorities": priorities}
        _allocate_step(context, data)
        return {"data": data, "allocation": data["allocations"],
                "delivered": data["delivered"]}

    def commit(state):
        return {"record": _commit_step(state["context"], state["data"])}

    builder.add_node("battery_mode", battery_mode)
    builder.add_node("arbiter", arbiter)
    builder.add_node("commit", commit)
    builder.add_edge([f"bid_{agent_id}" for agent_id in ids], "battery_mode")
    builder.add_edge("battery_mode", "arbiter")
    builder.add_edge("arbiter", "commit")
    builder.add_edge("commit", END)
    return builder.compile()


def run_graph_closed_loop(twin, allocator, steps=288, **fault_options):
    """Run actual graph nodes per step; return identical closed-loop records.

    Optional fault keywords are forwarded to the same policy implementation.
    No LLM clients, remote checkpointers or tracing callbacks are configured.
    """
    context = _run_context(twin, allocator, steps, **fault_options)
    graph = build_step_graph(context["agents"], context["priorities"], allocator)
    records = []
    for step in range(steps):
        data = _prepare_step(context, step)
        with tracing_context(enabled=False):
            result = graph.invoke({"obs": data["sanitized"], "bids": {},
                                   "context": context, "data": data})
        records.append(result["record"])
    return records
