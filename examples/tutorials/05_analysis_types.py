"""Tutorial 05: Analysis Types

This tutorial teaches you about SPICE analysis types:
1. DC Operating Point (.OP)
2. DC Sweep (.DC)
3. AC Analysis (.AC)
4. Transient Analysis (.TRAN)

Run: python examples/tutorials/05_analysis_types.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac, Vdc, Vpulse
from spicelab.core.net import GND, Net


def demo_dc_operating_point():
    """Demonstrate DC operating point analysis."""
    print("\n1. DC Operating Point Analysis (.OP)")
    print("-" * 40)

    circuit = Circuit("dc_op_demo")

    # Simple voltage divider
    v1 = Vdc("1", 10.0)
    r1 = Resistor("1", resistance=10_000)
    r2 = Resistor("2", resistance=10_000)

    circuit.add(v1, r1, r2)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(r2.ports[0], vout)
    circuit.connect(r2.ports[1], GND)

    print("""
   .OP (Operating Point Analysis)

   Purpose: Find DC voltages and currents at all nodes

   SPICE command: .OP

   Use cases:
   - Verify bias points in amplifiers
   - Check voltage divider outputs
   - Find transistor operating regions
   - Initial conditions for transient analysis

   Results: Node voltages, branch currents, power dissipation
""")
    print("   Circuit netlist:")
    print(circuit.build_netlist())
    print("   Add '.OP' command for DC analysis")
    print(f"   Expected Vout = {10.0 * 10000 / 20000}V")


def demo_dc_sweep():
    """Demonstrate DC sweep analysis."""
    print("\n2. DC Sweep Analysis (.DC)")
    print("-" * 40)

    circuit = Circuit("dc_sweep_demo")

    # Voltage divider for sweep
    v1 = Vdc("1", 0.0)  # Will be swept
    r1 = Resistor("1", resistance=1000)
    r2 = Resistor("2", resistance=1000)

    circuit.add(v1, r1, r2)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(r2.ports[0], vout)
    circuit.connect(r2.ports[1], GND)

    print("""
   .DC (DC Sweep Analysis)

   Purpose: Sweep a source and plot DC response

   SPICE commands:
   .DC V1 0 10 0.1        ; Sweep V1 from 0V to 10V in 0.1V steps
   .DC V1 0 10 1 V2 0 5 1 ; Nested sweep (two sources)

   Syntax: .DC <source> <start> <stop> <step>

   Use cases:
   - Transistor characteristic curves (I-V curves)
   - Transfer functions
   - Finding threshold voltages
   - Diode forward voltage analysis

   Output: Plot of node voltages/currents vs swept parameter
""")
    print("   Circuit netlist:")
    print(circuit.build_netlist())
    print("   Add '.DC V1 0 10 0.1' to sweep input voltage")


def demo_ac_analysis():
    """Demonstrate AC analysis."""
    print("\n3. AC Analysis (.AC)")
    print("-" * 40)

    circuit = Circuit("ac_analysis_demo")

    # RC lowpass filter
    v1 = Vac("1", amplitude=1.0)
    r1 = Resistor("1", resistance=10_000)
    c1 = Capacitor("1", capacitance=15.9e-9)  # fc ≈ 1kHz

    circuit.add(v1, r1, c1)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    print("""
   .AC (AC Frequency Analysis)

   Purpose: Compute frequency response (gain and phase)

   SPICE commands:
   .AC DEC 10 1 1MEG      ; Decade sweep, 10 pts/decade, 1Hz to 1MHz
   .AC LIN 100 1K 10K     ; Linear sweep, 100 points, 1kHz to 10kHz
   .AC OCT 8 10 10K       ; Octave sweep, 8 pts/octave, 10Hz to 10kHz

   Syntax: .AC <type> <points> <start_freq> <stop_freq>

   Types:
   - DEC: Logarithmic (decade) - best for wide ranges
   - LIN: Linear - for narrow frequency bands
   - OCT: Logarithmic (octave)

   Use cases:
   - Bode plots (magnitude and phase)
   - Filter frequency response
   - Amplifier bandwidth
   - Stability analysis (gain/phase margins)

   Requires: AC source (Vac or VAC 1 0)
""")
    print("   Circuit netlist (RC lowpass, fc ≈ 1kHz):")
    print(circuit.build_netlist())
    print("   Add '.AC DEC 20 10 100K' for frequency sweep")


def demo_transient_analysis():
    """Demonstrate transient analysis."""
    print("\n4. Transient Analysis (.TRAN)")
    print("-" * 40)

    circuit = Circuit("transient_demo")

    # RC circuit with pulse input
    v1 = Vpulse(
        "1",
        v1=0,
        v2=5,
        td=0,
        tr=1e-9,
        tf=1e-9,
        pw=1e-3,
        per=2e-3,  # 500Hz, 50% duty
    )
    r1 = Resistor("1", resistance=10_000)
    c1 = Capacitor("1", capacitance=100e-9)  # tau = 1ms

    circuit.add(v1, r1, c1)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    print("""
   .TRAN (Transient Analysis)

   Purpose: Time-domain simulation (voltage/current vs time)

   SPICE commands:
   .TRAN 1u 10m           ; 1us step, 10ms total
   .TRAN 1u 10m 0 1u      ; With print delay and max step
   .TRAN 1n 1m UIC        ; Use initial conditions

   Syntax: .TRAN <step> <stop> [<start>] [<max_step>] [UIC]

   Parameters:
   - step: Suggested output step
   - stop: End time
   - start: (optional) Start recording at this time
   - max_step: (optional) Maximum internal step
   - UIC: Use Initial Conditions from .IC statements

   Use cases:
   - Step response
   - Pulse response
   - Oscillator waveforms
   - Switching power supplies
   - Digital circuit timing
""")
    print("   Circuit netlist (RC with pulse, tau = 1ms):")
    print(circuit.build_netlist())
    print("   Add '.TRAN 10u 10m' for 10ms transient simulation")


def main():
    """Demonstrate SPICE analysis types."""
    print("=" * 60)
    print("Tutorial 05: Analysis Types")
    print("=" * 60)

    demo_dc_operating_point()
    demo_dc_sweep()
    demo_ac_analysis()
    demo_transient_analysis()

    print("\n" + "=" * 60)
    print("Summary: Analysis Commands")
    print("=" * 60)
    print("""
   | Analysis | Command               | Use Case               |
   |----------|-----------------------|------------------------|
   | .OP      | .OP                   | DC bias points         |
   | .DC      | .DC V1 0 10 0.1       | I-V curves, transfer   |
   | .AC      | .AC DEC 10 1 1MEG     | Frequency response     |
   | .TRAN    | .TRAN 1u 10m          | Time-domain response   |

   Output statements:
   .PRINT DC V(vout) I(R1)     ; Print to file
   .PLOT TRAN V(vin) V(vout)   ; ASCII plot
   .PROBE                       ; Save all for post-processing
""")


if __name__ == "__main__":
    main()
