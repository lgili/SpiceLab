"""PT1000 Circuit Analysis Example

This example demonstrates how to:
1. Load an LTspice .asc circuit file
2. Run Monte Carlo analysis with component tolerances
3. Run Worst Case analysis
4. Analyze and visualize the results

Circuit: PT1000 temperature sensor signal conditioning
File: PT1000_circuit_5.asc

Requirements:
- LTspice installed and accessible
- spicelib package (pip install spicelib)
- numpy, pandas, matplotlib

Run: python old/sim_files/pt1000_analysis_example.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spicelib import AscEditor, SimRunner
from spicelib.sim.tookit.montecarlo import Montecarlo
from spicelib.sim.tookit.worst_case import WorstCaseAnalysis
from spicelib.simulators.ltspice_simulator import LTspice

# Suppress verbose logging
logging.getLogger("spicelib").setLevel(logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

# Path to the circuit file
CIRCUIT_FILE = Path(__file__).parent / "PT1000_circuit_5.asc"
OUTPUT_FOLDER = Path(__file__).parent / "simulation_output"

# Component tolerances (as fractions, e.g., 0.01 = 1%)
TOLERANCES = {
    "R1": 0.001,   # 0.1% precision resistor (reference)
    "R2": 0.01,    # 1% resistor
    "R3": 0.01,    # 1% resistor
    "R4": 0.01,    # 1% resistor
    "R5": 0.01,    # 1% resistor
    "R7": 0.05,    # 5% potentiometer/trimmer
}

# Variables to measure from simulation
MEASUREMENT_VARS = {
    "Vout": ("MAX", "V(Vout)"),
    "Vrtd": ("MAX", "V(Vrtd)"),
}

# Monte Carlo settings
MC_ITERATIONS = 100

# =============================================================================
# Helper Functions
# =============================================================================


def load_circuit(file_path: Path) -> AscEditor:
    """Load an LTspice .asc circuit file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Circuit file not found: {file_path}")

    print(f"Loading circuit: {file_path.name}")
    circuit = AscEditor(str(file_path))

    # Display components
    components = circuit.get_components()
    print(f"Found {len(components)} components:")
    for comp in components:
        try:
            params = circuit.get_component_parameters(comp)
            value = params.get("Value", "N/A")
            print(f"  - {comp}: {value}")
        except Exception:
            pass

    return circuit


def run_monte_carlo(circuit_file: Path, output_folder: Path,
                    tolerances: dict, n_iterations: int) -> pd.DataFrame:
    """
    Run Monte Carlo analysis on the circuit.

    Monte Carlo analysis varies component values randomly within their
    tolerance ranges to determine statistical distribution of outputs.

    Args:
        circuit_file: Path to .asc file
        output_folder: Folder for simulation outputs
        tolerances: Dict of component tolerances {name: tolerance_fraction}
        n_iterations: Number of Monte Carlo runs

    Returns:
        DataFrame with simulation results
    """
    print(f"\n{'='*60}")
    print("Running Monte Carlo Analysis")
    print(f"{'='*60}")
    print(f"Iterations: {n_iterations}")
    print(f"Components with tolerances:")
    for comp, tol in tolerances.items():
        print(f"  - {comp}: ±{tol*100:.1f}%")

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Setup simulation
    circuit = AscEditor(str(circuit_file))
    runner = SimRunner(simulator=LTspice, output_folder=str(output_folder))

    # Create Monte Carlo analysis
    mc = Montecarlo(circuit, runner)

    # Set tolerances for each component
    for component, tolerance in tolerances.items():
        try:
            mc.set_tolerance(component, tolerance)
        except Exception as e:
            print(f"  Warning: Could not set tolerance for {component}: {e}")

    # Prepare and run simulation
    mc.prepare_testbench(num_runs=n_iterations)

    netlist_file = output_folder / f"{circuit_file.stem}_mc.asc"
    mc.save_netlist(str(netlist_file))

    print(f"\nRunning {n_iterations} simulations...")
    mc.run_testbench(runs_per_sim=min(10, n_iterations),
                     wait_resource=True, timeout=600)

    # Read results
    logs = mc.read_logfiles()

    # Extract measurement data
    data = {}
    for var_name in MEASUREMENT_VARS.keys():
        var_key = var_name.lower()
        if var_key in logs.dataset:
            data[var_name] = logs.dataset[var_key]

    # Extract component values used in each run
    for comp in tolerances.keys():
        comp_key = comp.lower()
        if comp_key in logs.dataset:
            data[comp] = [abs(v) for v in logs.dataset[comp_key]]

    # Cleanup
    mc.cleanup_files()

    df = pd.DataFrame(data)
    print(f"\nCollected {len(df)} data points")

    return df


def run_worst_case(circuit_file: Path, output_folder: Path,
                   tolerances: dict) -> pd.DataFrame:
    """
    Run Worst Case Analysis on the circuit.

    Worst Case Analysis tests all combinations of component values at
    their tolerance extremes to find the worst-case output conditions.

    Args:
        circuit_file: Path to .asc file
        output_folder: Folder for simulation outputs
        tolerances: Dict of component tolerances {name: tolerance_fraction}

    Returns:
        DataFrame with worst case results
    """
    print(f"\n{'='*60}")
    print("Running Worst Case Analysis")
    print(f"{'='*60}")
    print(f"Components with tolerances:")
    for comp, tol in tolerances.items():
        print(f"  - {comp}: ±{tol*100:.1f}%")

    n_combinations = 2 ** len(tolerances)
    print(f"Total combinations to test: {n_combinations}")

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Setup simulation
    circuit = AscEditor(str(circuit_file))
    runner = SimRunner(simulator=LTspice, output_folder=str(output_folder))

    # Create Worst Case analysis
    wca = WorstCaseAnalysis(circuit, runner)

    # Set tolerances for each component
    for component, tolerance in tolerances.items():
        try:
            wca.set_tolerance(component, tolerance)
        except Exception as e:
            print(f"  Warning: Could not set tolerance for {component}: {e}")

    # Save and run
    netlist_file = output_folder / f"{circuit_file.stem}_wc.asc"
    wca.save_netlist(str(netlist_file))

    print(f"\nRunning {n_combinations} worst-case combinations...")
    wca.run_testbench(wait_resource=True, timeout=600)

    # Read results
    logs = wca.read_logfiles()

    # Extract data
    data = {}
    for var_name in MEASUREMENT_VARS.keys():
        var_key = var_name.lower()
        if var_key in logs.dataset:
            data[var_name] = logs.dataset[var_key]

    for comp in tolerances.keys():
        comp_key = comp.lower()
        if comp_key in logs.dataset:
            data[comp] = [abs(v) for v in logs.dataset[comp_key]]

    # Cleanup
    wca.cleanup_files()

    df = pd.DataFrame(data)
    print(f"\nCollected {len(df)} worst-case combinations")

    return df


def analyze_monte_carlo_results(df: pd.DataFrame, variable: str = "Vout"):
    """
    Analyze and visualize Monte Carlo results.

    Args:
        df: DataFrame with Monte Carlo results
        variable: Variable to analyze
    """
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in results")
        return

    values = df[variable].dropna()

    print(f"\n{'='*60}")
    print(f"Monte Carlo Statistics for {variable}")
    print(f"{'='*60}")
    print(f"  Samples:     {len(values)}")
    print(f"  Mean:        {values.mean():.6f}")
    print(f"  Std Dev:     {values.std():.6f}")
    print(f"  Min:         {values.min():.6f}")
    print(f"  Max:         {values.max():.6f}")
    print(f"  Range:       {values.max() - values.min():.6f}")
    print(f"  3σ Range:    {values.mean() - 3*values.std():.6f} to {values.mean() + 3*values.std():.6f}")

    # Calculate yield (assuming ±5% tolerance on output)
    target = values.mean()
    tolerance = 0.05 * abs(target)
    in_spec = ((values >= target - tolerance) & (values <= target + tolerance)).sum()
    yield_pct = 100 * in_spec / len(values)
    print(f"  Yield (±5%): {yield_pct:.1f}%")

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(values, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(values.mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {values.mean():.4f}')
    ax1.axvline(values.mean() + values.std(), color='g', linestyle=':', linewidth=2)
    ax1.axvline(values.mean() - values.std(), color='g', linestyle=':', linewidth=2,
                label=f'±σ: {values.std():.4f}')
    ax1.set_xlabel(variable)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Monte Carlo Distribution of {variable}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot with component correlation
    ax2 = axes[1]
    comp_cols = [c for c in df.columns if c.startswith('R')]
    if comp_cols and variable in df.columns:
        # Normalize component values and calculate correlation
        correlations = []
        for comp in comp_cols:
            if comp in df.columns:
                corr = df[variable].corr(df[comp])
                correlations.append((comp, corr))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        comps = [c[0] for c in correlations]
        corrs = [c[1] for c in correlations]

        colors = ['green' if c > 0 else 'red' for c in corrs]
        ax2.barh(comps, corrs, color=colors, alpha=0.7)
        ax2.set_xlabel('Correlation with ' + variable)
        ax2.set_ylabel('Component')
        ax2.set_title('Sensitivity Analysis')
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f'monte_carlo_{variable}.png', dpi=150)
    plt.show()

    return fig


def analyze_worst_case_results(df: pd.DataFrame, variable: str = "Vout"):
    """
    Analyze Worst Case results.

    Args:
        df: DataFrame with worst case results
        variable: Variable to analyze
    """
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in results")
        return

    values = df[variable].dropna()

    print(f"\n{'='*60}")
    print(f"Worst Case Analysis for {variable}")
    print(f"{'='*60}")
    print(f"  Combinations: {len(values)}")
    print(f"  Nominal:      {values.iloc[0]:.6f} (first run, nominal)")
    print(f"  Minimum:      {values.min():.6f}")
    print(f"  Maximum:      {values.max():.6f}")
    print(f"  Total Range:  {values.max() - values.min():.6f}")

    # Find which combination gives min/max
    idx_min = values.idxmin()
    idx_max = values.idxmax()

    print(f"\n  Worst case MIN ({variable}={values.min():.6f}):")
    for comp in TOLERANCES.keys():
        if comp in df.columns:
            print(f"    {comp}: {df.loc[idx_min, comp]:.2f}")

    print(f"\n  Worst case MAX ({variable}={values.max():.6f}):")
    for comp in TOLERANCES.keys():
        if comp in df.columns:
            print(f"    {comp}: {df.loc[idx_max, comp]:.2f}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run the complete analysis example."""
    print("=" * 70)
    print("PT1000 Circuit Monte Carlo and Worst Case Analysis Example")
    print("=" * 70)

    # Check if circuit file exists
    if not CIRCUIT_FILE.exists():
        print(f"\nError: Circuit file not found: {CIRCUIT_FILE}")
        print("Please ensure PT1000_circuit_5.asc is in the same directory.")
        return

    # Load and display circuit info
    try:
        circuit = load_circuit(CIRCUIT_FILE)
    except Exception as e:
        print(f"Error loading circuit: {e}")
        return

    # Run Monte Carlo analysis
    try:
        mc_results = run_monte_carlo(
            CIRCUIT_FILE,
            OUTPUT_FOLDER,
            TOLERANCES,
            MC_ITERATIONS
        )

        # Save results
        mc_results.to_csv(OUTPUT_FOLDER / "monte_carlo_results.csv", index=False)
        print(f"\nMonte Carlo results saved to: {OUTPUT_FOLDER / 'monte_carlo_results.csv'}")

        # Analyze results
        analyze_monte_carlo_results(mc_results, "Vout")

    except Exception as e:
        print(f"\nMonte Carlo analysis failed: {e}")
        print("Make sure LTspice is installed and accessible.")

    # Run Worst Case analysis
    try:
        wc_results = run_worst_case(
            CIRCUIT_FILE,
            OUTPUT_FOLDER,
            TOLERANCES
        )

        # Save results
        wc_results.to_csv(OUTPUT_FOLDER / "worst_case_results.csv", index=False)
        print(f"\nWorst case results saved to: {OUTPUT_FOLDER / 'worst_case_results.csv'}")

        # Analyze results
        analyze_worst_case_results(wc_results, "Vout")

    except Exception as e:
        print(f"\nWorst case analysis failed: {e}")
        print("Make sure LTspice is installed and accessible.")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
