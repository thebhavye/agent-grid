"""End-to-end checks for the real-agent simulation pipeline."""

from data.generate_dataset import generate_dataset
from simulation.baseline_allocator import allocate
from simulation.run_simulation import run_simulation

EXPECTED_AGENTS = {
    "solar_1",
    "battery_1",
    "ev_1",
    "appliance_fridge",
    "appliance_lighting",
    "appliance_security_system",
    "appliance_ev_charger",
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

    for i, allocation in enumerate(results):
        available = float(ts["solar_gen_kw"].iloc[i])
        total = sum(allocation.values())
        # Baseline allocator splits exactly `available` proportionally;
        # essentials guarantee non-zero demand, so total == available.
        # The <= covers the degenerate all-zero-demand case (sum 0).
        assert total <= available + 1e-6, f"step {i}: {total} > solar {available}"
        if total > 1e-9:
            assert abs(total - available) < 1e-6, f"step {i}: {total} != solar {available}"


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
