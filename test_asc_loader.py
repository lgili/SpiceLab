"""
Test script for loading an LTspice .asc file and running a simulation.
"""

import re
import tempfile
from pathlib import Path
from spicelab.io.ltspice_asc import parse_asc, schematic_to_circuit
from spicelab.core.types import AnalysisSpec
from spicelab.engines.orchestrator import run_simulation


def parse_rtd_params(asc_content: str) -> dict[str, float]:
    params = {}
    
    # Extract individual parameters
    param_patterns = {
        "A": r"!\.param A=([0-9e.-]+)([munp]?)",
        "B": r"!\.param B=([0-9e.-]+)([munp]?)",
        "C": r"!\.param C=([0-9e.-]+)([munp]?)",
        "R0": r"!\.param R0=([0-9e.-]+)",
        "T1": r"!\.param T1=([0-9e.-]+)",
    }

    unit_multipliers = {
        "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "": 1.0
    }

    for key, pattern in param_patterns.items():
        match = re.search(pattern, asc_content)
        if match:
            value = float(match.group(1))
            unit = match.group(2) if len(match.groups()) > 1 else ""
            params[key] = value * unit_multipliers.get(unit, 1.0)
        else:
            # Provide default values if parameters are not found, or raise an error
            if key == "T1":
                params[key] = 0.0 # Default T1 to 0 if not specified
            else:
                params[key] = 0.0 # Default other missing params to 0.0


    # Calculate Rrtd
    T1 = params["T1"]
    A = params["A"]
    B = params["B"]
    C = params["C"]
    R0 = params["R0"]

    UNIT = 1.0 if T1 < 0 else 0.0
    rrtd_val = R0 * (1 + A*T1 + B*(T1**2) + C*(T1-100)*(T1**3)*UNIT)
    params["Rrtd"] = rrtd_val
    
    return params


def main():
    print("Loading circuit from .asc file...")
    asc_path = Path("old/sim_files/PT1000_circuit_1.asc")
    
    # Read the original content
    original_asc_content = asc_path.read_text(encoding="utf-8")
    
    # Parse parameters
    params = parse_rtd_params(original_asc_content)
    
    # Replace Rrtd in the content
    modified_asc_content = original_asc_content
    if "Rrtd" in params:
        modified_asc_content = re.sub(r"\{Rrtd\}", str(params["Rrtd"]), modified_asc_content)

    # Remove .param and .meas directives
    modified_asc_content = re.sub(r"TEXT.*?!\.(param|meas)\s+.*", "", modified_asc_content, flags=re.IGNORECASE)
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".asc", delete=False, encoding="utf-8") as temp_asc_file:
        temp_asc_file.write(modified_asc_content)
        temp_file_path = Path(temp_asc_file.name)

    try:
        # Load the modified schematic
        schematic_content = temp_file_path.read_text(encoding="utf-8")
        schematic = parse_asc(schematic_content)
        circuit = schematic_to_circuit(schematic)
        print("Circuit loaded successfully from modified .asc content.")
    except Exception as e:
        print(f"Error loading circuit from modified .asc content: {e}")
        return
    finally:
        # Clean up the temporary file
        temp_file_path.unlink()

    print("\n--- Netlist ---")
    print(circuit.build_netlist())

    print("\n## Running Transient Analysis")
    tran = AnalysisSpec("tran", {"tstep": "10us", "tstop": "5ms"})
    
    try:
        handle = run_simulation(circuit, [tran], engine="ltspice")
        ds = handle.dataset()
        print("\n--- Simulation Results ---")
        print(ds)
    except Exception as e:
        print(f"\nError running simulation: {e}")


if __name__ == "__main__":
    main()