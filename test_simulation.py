"""End-to-end checks for the real-agent simulation pipeline."""

from data.generate_dataset import generate_dataset
from simulation.baseline_allocator import allocate
from simulation.run_simulation import _build_agents, run_simulation

# Fixed system agents _build_agents() always creates (not roster-driven).
SYSTEM_AGENT_IDS = {"solar_1", "battery_1", "ev_1"}


def expected_agent_ids(appliances_df):
    """Derive allocation keys from the generated roster, mirroring
    _build_agents(): system agents plus appliance_<id> for every roster
    row except ev_charger (which becomes ev_1, not appliance_ev_charger)."""
    ids = set(SYSTEM_AGENT_IDS)
    roster = appliances_df.to_dict(orient="records")
    for entry in roster:
        aid = entry.get("appliance_id", "unknown")
        if aid == "ev_charger":
            continue
        ids.add(f"appliance_{aid}")
    return ids


def _ev_signature(timeseries):
    """Observable EV trajectory signature: plugged-in trajectory plus the
    reported departure minute. Distinct scenarios produce distinct
    signatures without the test assuming which seed maps to which
    scenario name."""
    return (
        timeseries["ev_plugged_in"].to_numpy().tobytes(),
        int(timeseries["ev_departure_min"].iloc[0]),
    )


def _select_scenario_datasets(count=2, candidates=range(32)):
    """Pick deterministic datasets with distinct EV trajectories.

    Scans candidate seeds in order, groups by actual EV signature, and
    returns the first dataset of each of the first `count` distinct
    groups as (seed, tables) pairs. Fails loudly if the generator stops
    exposing enough distinct EV trajectories.
    """
    groups = {}
    for seed in candidates:
        tables = generate_dataset(seed=seed)
        sig = _ev_signature(tables["timeseries"])
        if sig not in groups:
            groups[sig] = (seed, tables)
        if len(groups) >= count:
            break
    assert len(groups) >= count, (
        f"generator exposed only {len(groups)} distinct EV trajectory(s) "
        f"across {len(list(candidates))} candidate seeds; need {count}"
    )
    return list(groups.values())


# Deterministic multi-scenario coverage (Issue #12): resolved once at
# import so every behavior test below runs on the same two datasets.
SCENARIO_DATASETS = _select_scenario_datasets()


def test_scenario_selection_covers_distinct_ev_trajectories():
    assert len(SCENARIO_DATASETS) >= 2
    seeds = [seed for seed, _ in SCENARIO_DATASETS]
    assert len(set(seeds)) >= 2
    sigs = [_ev_signature(tables["timeseries"]) for _, tables in SCENARIO_DATASETS]
    assert len(set(sigs)) >= 2, "selected datasets must have distinct EV trajectories"


def test_full_pipeline_runs_all_288_timesteps():
    for _seed, tables in SCENARIO_DATASETS:
        results = run_simulation(tables, allocate)

        assert len(results) == 288
        assert len(tables["timeseries"]) == 288
        expected = expected_agent_ids(tables["appliances"])
        for allocation in results:
            assert set(allocation.keys()) == expected


def test_no_negative_allocations():
    for _seed, tables in SCENARIO_DATASETS:
        results = run_simulation(tables, allocate)

        for allocation in results:
            for agent_id, value in allocation.items():
                assert value >= -1e-9, f"negative allocation for {agent_id}: {value}"


def test_allocations_do_not_exceed_available_solar():
    for _seed, tables in SCENARIO_DATASETS:
        ts = tables["timeseries"]
        results = run_simulation(tables, allocate)

        agents = _build_agents(tables["appliances"])
        rows = ts.to_dict(orient="records")

        for i, allocation in enumerate(results):
            available = float(ts["solar_gen_kw"].iloc[i])
            total = sum(allocation.values())
            for agent in agents:
                agent.update_state(rows[i])
            demand_total = sum(float(agent.get_demand_kw()) for agent in agents)
            # Capped allocator: total == min(available, demand); surplus solar
            # is curtailed, never fabricated as consumption (issue #5).
            assert total <= available + 1e-6, f"step {i}: {total} > solar {available}"
            assert abs(total - min(available, demand_total)) < 1e-6, (
                f"step {i}: {total} != min(solar {available}, demand {demand_total})"
            )


def test_no_allocation_exceeds_demand():
    for _seed, tables in SCENARIO_DATASETS:
        ts = tables["timeseries"]
        results = run_simulation(tables, allocate)

        agents = _build_agents(tables["appliances"])
        rows = ts.to_dict(orient="records")

        for i, allocation in enumerate(results):
            for agent in agents:
                agent.update_state(rows[i])
            for agent in agents:
                demand = float(agent.get_demand_kw())
                assert allocation[agent.agent_id] <= demand + 1e-9, (
                    f"step {i} {agent.agent_id}: allocated "
                    f"{allocation[agent.agent_id]} > demand {demand}"
                )


def main():
    _seed, tables = SCENARIO_DATASETS[0]

    results = run_simulation(tables, allocate)

    print("Simulation ran successfully.\n")
    print(f"Timesteps: {len(results)}")

    for timestep, allocation in enumerate(results[:2], start=1):
        total_allocated = sum(allocation.values())

        print(f"Timestep {timestep}")
        print(f"  Allocation: {allocation}")
        print(f"  Total allocated: {total_allocated:.2f} kW")
        print()

    print(f"... ({len(results) - 2} more timesteps)")


if __name__ == "__main__":
    main()
