"""Basic end-to-end checks for the placeholder simulation."""

from simulation.baseline_allocator import allocate
from simulation.run_simulation import run_simulation


def main():
    fake_dataset = [
        {"solar_gen_kw": 5.0},
        {"solar_gen_kw": 3.0},
    ]

    results = run_simulation(fake_dataset, allocate)

    print("Simulation ran successfully.\n")

    for timestep, allocation in enumerate(results, start=1):
        total_allocated = sum(allocation.values())

        print(f"Timestep {timestep}")
        print(f"  Allocation: {allocation}")
        print(f"  Total allocated: {total_allocated:.2f} kW")
        print()


if __name__ == "__main__":
    main()
