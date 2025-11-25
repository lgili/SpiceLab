"""Batch Simulation Runner

Demonstrates how to automate running multiple simulations
with different parameters and collect results.

Run: python examples/automation/batch_simulation.py
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    name: str
    vin: float
    r1: float
    r2: float
    capacitor: float | None = None


@dataclass
class SimulationResult:
    """Result of a single simulation."""

    config: SimulationConfig
    vout_calculated: float
    circuit_valid: bool
    error_message: str | None = None


def build_circuit(config: SimulationConfig) -> Circuit:
    """Build circuit from configuration."""
    circuit = Circuit(config.name)

    v_in = Vdc("in", config.vin)
    r1 = Resistor("1", resistance=config.r1)
    r2 = Resistor("2", resistance=config.r2)

    circuit.add(v_in, r1, r2)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(r2.ports[0], vout)
    circuit.connect(r2.ports[1], GND)

    if config.capacitor:
        c1 = Capacitor("1", capacitance=config.capacitor)
        circuit.add(c1)
        circuit.connect(c1.ports[0], vout)
        circuit.connect(c1.ports[1], GND)

    return circuit


def calculate_vout(config: SimulationConfig) -> float:
    """Calculate expected output voltage."""
    return config.vin * config.r2 / (config.r1 + config.r2)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run a single simulation."""
    try:
        circuit = build_circuit(config)
        result = circuit.validate()
        vout = calculate_vout(config)

        return SimulationResult(
            config=config,
            vout_calculated=vout,
            circuit_valid=result.is_valid,
            error_message=None if result.is_valid else "Validation failed",
        )
    except Exception as e:
        return SimulationResult(
            config=config,
            vout_calculated=0.0,
            circuit_valid=False,
            error_message=str(e),
        )


class BatchSimulator:
    """Batch simulation runner."""

    def __init__(self):
        self.configs: list[SimulationConfig] = []
        self.results: list[SimulationResult] = []

    def add_config(self, config: SimulationConfig):
        """Add a simulation configuration."""
        self.configs.append(config)

    def add_configs_from_sweep(
        self,
        base_name: str,
        vin: float,
        r1_values: list[float],
        r2_values: list[float],
    ):
        """Generate configurations from parameter sweep."""
        for i, r1 in enumerate(r1_values):
            for j, r2 in enumerate(r2_values):
                config = SimulationConfig(
                    name=f"{base_name}_r1{i}_r2{j}",
                    vin=vin,
                    r1=r1,
                    r2=r2,
                )
                self.configs.append(config)

    def run_all(self, progress_callback=None) -> list[SimulationResult]:
        """Run all configured simulations."""
        self.results = []
        total = len(self.configs)

        for i, config in enumerate(self.configs):
            result = run_simulation(config)
            self.results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return self.results

    def get_summary(self) -> dict:
        """Get summary statistics of results."""
        if not self.results:
            return {}

        valid_count = sum(1 for r in self.results if r.circuit_valid)
        vout_values = [r.vout_calculated for r in self.results if r.circuit_valid]

        return {
            "total_simulations": len(self.results),
            "valid_circuits": valid_count,
            "failed_circuits": len(self.results) - valid_count,
            "vout_min": min(vout_values) if vout_values else None,
            "vout_max": max(vout_values) if vout_values else None,
            "vout_mean": sum(vout_values) / len(vout_values) if vout_values else None,
        }

    def export_results(self, filepath: Path | str):
        """Export results to JSON file."""
        filepath = Path(filepath)
        data = {
            "summary": self.get_summary(),
            "results": [
                {
                    "config": asdict(r.config),
                    "vout_calculated": r.vout_calculated,
                    "circuit_valid": r.circuit_valid,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
        }
        filepath.write_text(json.dumps(data, indent=2))


def main():
    """Demonstrate batch simulation automation."""
    print("=" * 60)
    print("Automation: Batch Simulation Runner")
    print("=" * 60)

    # Create batch simulator
    batch = BatchSimulator()

    # Add individual configurations
    batch.add_config(SimulationConfig("test_5v", 5.0, 10_000, 10_000))
    batch.add_config(SimulationConfig("test_12v", 12.0, 10_000, 5_000))
    batch.add_config(SimulationConfig("test_3v3", 3.3, 4_700, 10_000))

    # Add sweep configurations
    r1_values = [1_000, 4_700, 10_000, 47_000]
    r2_values = [1_000, 4_700, 10_000, 47_000]
    batch.add_configs_from_sweep("sweep", 5.0, r1_values, r2_values)

    print(f"""
   Batch Simulation Configuration
   ─────────────────────────────────
   Individual tests: 3
   Sweep configs: {len(r1_values)} × {len(r2_values)} = {len(r1_values) * len(r2_values)}
   Total simulations: {len(batch.configs)}
""")

    # Define progress callback
    def progress(current, total, result):
        status = "✓" if result.circuit_valid else "✗"
        print(
            f"   [{current:3d}/{total}] {result.config.name:20s} "
            f"Vout={result.vout_calculated:.3f}V {status}"
        )

    # Run all simulations
    print("   Running simulations:")
    print("   " + "-" * 50)
    batch.run_all(progress_callback=progress)

    # Get summary
    summary = batch.get_summary()
    print(f"""
   Summary:
   ─────────────────────────────────
   Total: {summary['total_simulations']}
   Valid: {summary['valid_circuits']}
   Failed: {summary['failed_circuits']}
   Vout range: {summary['vout_min']:.3f}V to {summary['vout_max']:.3f}V
   Vout mean: {summary['vout_mean']:.3f}V

   Use Cases:
   ┌────────────────────────────────────────────────────────┐
   │ 1. Component selection: Test multiple R values         │
   │ 2. Tolerance analysis: Sweep within tolerance range    │
   │ 3. Design exploration: Try different topologies        │
   │ 4. Regression testing: Verify designs after changes    │
   │ 5. Documentation: Generate results for reports         │
   └────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
