"""End-to-end checks for the real-agent simulation pipeline."""

from data.generate_dataset import generate_dataset
from simulation.baseline_allocator import allocate
from simulation.run_simulation import _build_agents, run_simulation

EXPECTED_AGENTS = {
    "solar_1",
    "battery_1",
    "ev_1",
    "appliance_fridge",
    "appliance_lighting",
    "appliance_security_system",
    "appliance_dishwasher",
    "appliance_washing_machine",
    "appliance_water_heater",
    "appliance_pool_pump",
}


def test_full_pipeline_runs_all_288_timesteps():
    tables = generate_dataset(seed=42)
    results = run_simulation(tables, allocate)

    assert len(results) == 288
    assert len(tables["timeseries"]) == 288
    for allocation in results:
        assert set(allocation.keys()) == EXPECTED_AGENTS


def test_no_negative_allocations():
    tables = generate_dataset(seed=42)
    results = run_simulation(tables, allocate)

    for allocation in results:
        for agent_id, value in allocation.items():
            assert value >= -1e-9, f"negative allocation for {agent_id}: {value}"


def test_allocations_do_not_exceed_available_solar():
    tables = generate_dataset(seed=42)
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
    tables = generate_dataset(seed=42)
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
    tables = generate_dataset(seed=42)

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
