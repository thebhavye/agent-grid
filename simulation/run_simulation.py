"""Timestep-based simulation runner."""

from .baseline_allocator import allocate


# Temporary placeholder demands.
# These will be replaced by values from the real dataset generator later.
DEFAULT_DEMANDS = {
    "battery_1": 2.0,
    "ev_1": 4.0,
    "appliance_1": 1.0,
}


def run_simulation(dataset, allocator):
    """Run an allocator once for every row/timestep in the dataset.

    The allocator is passed as an argument so the baseline can later be
    replaced by a smarter allocation strategy without changing this loop.
    """
    results = []

    for row in dataset:
        fake_demands = DEFAULT_DEMANDS.copy()
        available_power = row.get("solar_gen_kw", 5.0)

        allocation = allocator(available_power, fake_demands)

        results.append(allocation)

    return results


if __name__ == "__main__":
    # Small local smoke test for manual execution.
    fake_dataset = [
        {"solar_gen_kw": 5.0},
        {"solar_gen_kw": 3.0},
    ]

    results = run_simulation(fake_dataset, allocate)

    for timestep, allocation in enumerate(results, start=1):
        print(f"Timestep {timestep}: {allocation}")
