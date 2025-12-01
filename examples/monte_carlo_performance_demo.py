"""
Monte Carlo Performance Demo

This script demonstrates how to run a large Monte Carlo simulation
efficiently using PyCircuitKit's advanced features.

It leverages:
1.  `ParallelExecutor` to distribute simulation runs across multiple CPU cores.
2.  `HDF5ResultStorage` to efficiently save massive datasets to a single,
    compressed HDF5 file.
3.  A `tqdm` progress bar to provide real-time feedback during the long run.
4.  Detailed result objects (`JobResult`, `BatchResult`) for robust error
    tracking and analysis.

**Scenario:**
We will run a Monte Carlo simulation on a simple RC low-pass filter.
We'll vary the resistance and capacitance values randomly around a nominal
value and run a transient analysis for each variation. The goal is to
see the impact of component tolerance on the filter's step response.
"""

import os
import random
import time
from pathlib import Path

import numpy as np

# Core spicelab components
from spicelab import Circuit, Vpulse, Resistor, Capacitor, Net, GND
from spicelab.shortcuts import quick_tran
from spicelab.parallel import ParallelExecutor
from spicelab.storage import HDF5ResultStorage


# --- 1. Define the Simulation Function ---
# This function will be executed in parallel for each Monte Carlo run.
# It takes the parameters for a single run, builds the circuit,
# simulates it, and returns the results.
def run_single_simulation(params: dict) -> dict:
    """
    Builds and simulates a single RC filter circuit.

    Args:
        params: A dictionary containing 'run_id', 'resistance', and 'capacitance'.

    Returns:
        A dictionary containing the 'vout' and 'time' arrays from the simulation.
    """
    # Build the circuit with the specified R and C values
    c = Circuit(name=f"mc_run_{params['run_id']}")
    vin_pulse = Vpulse(
        "in", v1=0, v2=1, td="0s", tr="1ns", tf="1ns", pw="100us", per="200us"
    )
    r1 = Resistor("1", resistance=params["resistance"])
    c1 = Capacitor("1", capacitance=params["capacitance"])
    c.add(vin_pulse, r1, c1)

    vin_net, vout_net = Net("vin"), Net("vout")
    c.connect(vin_pulse.ports[0], vin_net)
    c.connect(r1.ports[0], vin_net)
    c.connect(r1.ports[1], vout_net)
    c.connect(c1.ports[0], vout_net)
    c.connect(vin_pulse.ports[1], GND)
    c.connect(c1.ports[1], GND)

    # Run a transient analysis using the high-level shortcut
    # The `probes` argument ensures we measure the voltage at 'vout'
    # In a real scenario, you could save specific measurements instead of
    # the full waveform to save space for very large runs.
    result_handle = quick_tran(c, "10u", probes=["V(vout)"])

    # Extract the dataset and return the relevant arrays
    ds = result_handle.dataset()
    return {"vout": ds["V(vout)"].values, "time": ds.coords["time"].values}


def main():
    """Main function to set up and run the Monte Carlo simulation."""

    print("=" * 60)
    print("  High-Performance Monte Carlo Simulation Demo")
    print("=" * 60)

    # --- 2. Setup Simulation Parameters ---
    NUM_RUNS = 100  # Number of Monte Carlo runs
    NOMINAL_R = 1e3  # 1 kΩ
    NOMINAL_C = 1e-9  # 1 nF
    TOLERANCE = 0.1  # 10% tolerance

    # Generate a list of parameters for each simulation run
    sim_params = [
        {
            "run_id": i,
            "resistance": NOMINAL_R * (1 + random.uniform(-TOLERANCE, TOLERANCE)),
            "capacitance": NOMINAL_C * (1 + random.uniform(-TOLERANCE, TOLERANCE)),
        }
        for i in range(NUM_RUNS)
    ]

    # --- 3. Setup Parallel Executor ---
    # Use all available CPU cores for maximum speed.
    # Enable the progress bar for user feedback.
    # The executor will automatically use `tqdm` if it's installed.
    max_workers = os.cpu_count()
    print(f"\nRunning {NUM_RUNS} simulations on {max_workers} CPU cores...")
    executor = ParallelExecutor(max_workers=max_workers, progress=True)

    # --- 4. Run the Simulations in Parallel ---
    start_time = time.time()
    batch_result = executor.map(run_single_simulation, sim_params)
    end_time = time.time()
    print(f"\nParallel simulation finished in {end_time - start_time:.2f} seconds.")

    # --- 5. Process and Save the Results ---
    if batch_result.failed_jobs > 0:
        print(f"\nWARNING: {batch_result.failed_jobs} simulations failed.")
        # You can inspect the failures:
        for job in batch_result.get_failed_jobs():
            print(f"  Job {job.job_id} failed: {job.error}")

    # Prepare results for storage
    successful_results = batch_result.get_successful_results()
    if not successful_results:
        print("No successful simulations to save. Exiting.")
        return

    # Create a dictionary to hold all results for batch saving
    # The key is the run ID, and the value is the simulation output
    results_to_save = {}
    for i, job in enumerate(batch_result.results):
        if job.success:
            run_id = job.input_params["run_id"]
            results_to_save[f"run_{run_id}"] = job.value


    # Save to HDF5
    output_path = Path("examples_output/monte_carlo_results.h5")
    output_path.parent.mkdir(exist_ok=True)
    if output_path.exists():
        output_path.unlink() # Remove old file

    print(f"\nSaving {len(successful_results)} results to '{output_path}'...")
    storage = HDF5ResultStorage(output_path, compression="gzip")
    storage.save_batch(results_to_save, group_name="monte_carlo_runs")
    storage.close()

    print(f"Successfully saved results. File size: {output_path.stat().st_size / 1e6:.3f} MB")

    # You can now load the results later for post-processing
    # Example of loading one result:
    # with HDF5ResultStorage(output_path, mode='r') as store:
    #     run_5_data = store.load_result('monte_carlo_runs/run_5')
    #     plt.plot(run_5_data['time'], run_5_data['vout'])

    print("\nDemo finished.")


if __name__ == "__main__":
    main()
