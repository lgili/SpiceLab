"""Audio Compressor/Limiter

Audio dynamics processors that reduce the dynamic range of signals.
Compressors provide gradual gain reduction, limiters provide hard
clamping above a threshold.

Run: python examples/signal_processing/audio_compressor.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vsin
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_simple_limiter() -> Circuit:
    """Build a simple diode limiter (clipper).

    Uses back-to-back diodes to hard-limit signal amplitude.
    Simple but creates harmonic distortion.

    Returns:
        Circuit with diode limiter
    """
    circuit = Circuit("diode_limiter")

    # Input
    v_in = Vsin("in", "0 2.0 1000")  # 2V amplitude (will be limited)

    # Back-to-back diodes
    d1 = create_component("diode.1n4148", "1")  # Positive limit
    d2 = create_component("diode.1n4148", "2")  # Negative limit

    # Series resistor (current limiting)
    r1 = Resistor("1", resistance=1000)

    # Bias diodes for adjustable threshold
    d_bias1 = create_component("diode.1n4148", "b1")
    d_bias2 = create_component("diode.1n4148", "b2")

    circuit.add(v_in, d1, d2, d_bias1, d_bias2, r1)

    # Nets
    vin = Net("vin")
    vout = Net("vout")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Series resistor
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)

    # Positive clamp: D1 + D_bias1 in series to GND
    circuit.connect(d1.ports[0], vout)  # Anode at output
    circuit.connect(d1.ports[1], Net("pos_bias"))  # Cathode
    circuit.connect(d_bias1.ports[0], Net("pos_bias"))
    circuit.connect(d_bias1.ports[1], GND)

    # Negative clamp: D2 + D_bias2 in series from GND
    circuit.connect(d2.ports[0], Net("neg_bias"))  # Anode
    circuit.connect(d2.ports[1], vout)  # Cathode at output
    circuit.connect(d_bias2.ports[0], GND)
    circuit.connect(d_bias2.ports[1], Net("neg_bias"))

    return circuit


def build_soft_knee_compressor() -> Circuit:
    """Build a soft-knee compressor using op-amp and feedback.

    Provides gradual compression that sounds more natural than
    hard limiting. Uses variable gain element controlled by
    signal level.

    Returns:
        Circuit with soft-knee compressor
    """
    circuit = Circuit("soft_knee_compressor")

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # VCA section (simplified using JFET)
    j1 = create_component("jfet.2n5457", "1")

    # Sidechain detector
    u_det = create_component("opamp.ideal", "det")
    d_det = create_component("diode.1n4148", "det")

    # Output buffer
    u_out = create_component("opamp.ideal", "out")

    # Attack/Release timing
    r_attack = Resistor("att", resistance=10_000)  # Fast attack
    r_release = Resistor("rel", resistance=100_000)  # Slow release
    c_time = Capacitor("time", capacitance=10e-6)

    # Threshold setting
    r_thresh = Resistor("thresh", resistance=47_000)

    # Gain resistors
    r_in = Resistor("in", resistance=10_000)
    r_fb = Resistor("fb", resistance=10_000)

    # Input coupling
    c_in = Capacitor("in", capacitance=1e-6)

    circuit.add(v_in, j1, u_det, u_out, d_det)
    circuit.add(r_attack, r_release, c_time, r_thresh, r_in, r_fb, c_in)

    # Nets
    vin = Net("vin")
    v_coupled = Net("v_coupled")
    vout = Net("vout")
    v_ctrl = Net("v_ctrl")
    v_det_in = Net("v_det_in")
    v_det_out = Net("v_det_out")
    v_inv = Net("v_inv")

    # Input coupling
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(c_in.ports[0], vin)
    circuit.connect(c_in.ports[1], v_coupled)

    # JFET VCA
    circuit.connect(j1.ports[0], v_coupled)  # Drain
    circuit.connect(j1.ports[1], v_ctrl)  # Gate (control)
    circuit.connect(j1.ports[2], Net("v_atten"))  # Source

    # Load for JFET
    r_load = Resistor("load", resistance=10_000)
    circuit.add(r_load)
    circuit.connect(r_load.ports[0], Net("v_atten"))
    circuit.connect(r_load.ports[1], GND)

    # Output buffer
    circuit.connect(r_in.ports[0], Net("v_atten"))
    circuit.connect(r_in.ports[1], v_inv)

    circuit.connect(u_out.ports[0], v_inv)
    circuit.connect(u_out.ports[1], GND)
    circuit.connect(u_out.ports[2], vout)

    circuit.connect(r_fb.ports[0], v_inv)
    circuit.connect(r_fb.ports[1], vout)

    # Sidechain detector (from output)
    circuit.connect(r_thresh.ports[0], vout)
    circuit.connect(r_thresh.ports[1], v_det_in)

    circuit.connect(u_det.ports[0], v_det_in)
    circuit.connect(u_det.ports[1], GND)
    circuit.connect(u_det.ports[2], v_det_out)

    # Rectifier
    circuit.connect(d_det.ports[0], v_det_out)
    circuit.connect(d_det.ports[1], v_ctrl)

    # Attack path (fast charge)
    circuit.connect(r_attack.ports[0], v_det_out)
    circuit.connect(r_attack.ports[1], v_ctrl)

    # Release path (slow discharge)
    circuit.connect(r_release.ports[0], v_ctrl)
    circuit.connect(r_release.ports[1], GND)

    # Time constant capacitor
    circuit.connect(c_time.ports[0], v_ctrl)
    circuit.connect(c_time.ports[1], GND)

    return circuit


def build_optical_compressor() -> Circuit:
    """Build an opto-compressor concept (LED + LDR simulation).

    Optical compressors use an LED to illuminate a photoresistor.
    Known for smooth, musical compression characteristics.

    Returns:
        Circuit simulating opto-compressor behavior
    """
    circuit = Circuit("optical_compressor")

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Simulate optocoupler with JFET (LDR simulation)
    j_ldr = create_component("jfet.2n5457", "ldr")

    # LED driver (sidechain)
    u_led = create_component("opamp.ideal", "led")
    d_led = create_component("diode.led.red", "1")

    # Output stage
    u_out = create_component("opamp.ideal", "out")

    # Sidechain filter (slow time constants = smooth compression)
    r_side = Resistor("side", resistance=100_000)
    c_side = Capacitor("side", capacitance=47e-6)  # Large cap = slow

    # Gain stage
    r_in = Resistor("in", resistance=10_000)
    r_fb = Resistor("fb", resistance=100_000)

    # Input coupling
    c_in = Capacitor("in", capacitance=1e-6)

    circuit.add(v_in, j_ldr, u_led, u_out, d_led)
    circuit.add(r_side, c_side, r_in, r_fb, c_in)

    # Nets
    vin = Net("vin")
    v_coupled = Net("v_coupled")
    vout = Net("vout")
    v_ctrl = Net("v_ctrl")
    v_side = Net("v_side")
    v_inv = Net("v_inv")
    v_led = Net("v_led")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(c_in.ports[0], vin)
    circuit.connect(c_in.ports[1], v_coupled)

    # JFET as "LDR" (variable resistance)
    circuit.connect(j_ldr.ports[0], v_coupled)  # Drain
    circuit.connect(j_ldr.ports[1], v_ctrl)  # Gate (simulates light level)
    circuit.connect(j_ldr.ports[2], Net("v_atten"))  # Source

    r_load = Resistor("load", resistance=10_000)
    circuit.add(r_load)
    circuit.connect(r_load.ports[0], Net("v_atten"))
    circuit.connect(r_load.ports[1], GND)

    # Output buffer
    circuit.connect(r_in.ports[0], Net("v_atten"))
    circuit.connect(r_in.ports[1], v_inv)

    circuit.connect(u_out.ports[0], v_inv)
    circuit.connect(u_out.ports[1], GND)
    circuit.connect(u_out.ports[2], vout)

    circuit.connect(r_fb.ports[0], v_inv)
    circuit.connect(r_fb.ports[1], vout)

    # Sidechain: rectify output, drive "LED"
    circuit.connect(r_side.ports[0], vout)
    circuit.connect(r_side.ports[1], v_side)

    circuit.connect(u_led.ports[0], v_side)
    circuit.connect(u_led.ports[1], GND)
    circuit.connect(u_led.ports[2], v_led)

    # LED (represents the optical element)
    circuit.connect(d_led.ports[0], v_led)
    circuit.connect(d_led.ports[1], v_ctrl)

    # Smoothing capacitor (creates the "slow" opto response)
    circuit.connect(c_side.ports[0], v_ctrl)
    circuit.connect(c_side.ports[1], GND)

    return circuit


def main():
    """Demonstrate audio compressor circuits."""
    print("=" * 60)
    print("Audio Compressor/Limiter")
    print("=" * 60)

    circuit1 = build_simple_limiter()
    circuit2 = build_soft_knee_compressor()
    circuit3 = build_optical_compressor()

    print("""
   Audio Dynamics Processing

   1. Simple Diode Limiter:

   Vin ──[R]──┬── Vout
              │
           ┌──┴──┐
          [D1]  [D2]
           │     │
          [Db1] [Db2]
           │     │
          GND   GND

   - Hard clips signal at ~±1.2V (two diode drops)
   - Fast limiting, but creates harmonics
   - Threshold adjustable by adding bias diodes

   2. Soft-Knee Compressor:

   Vin ──[JFET VCA]──[Buffer]── Vout
              │                   │
              G                   │
              │                   │
              └──[Attack]──[Detector]
                    │
                 [Release]
                    │
                  [C_time]

   - Gradual gain reduction (soft knee)
   - Attack: How fast compression starts (~1-10ms)
   - Release: How fast compression ends (~100-500ms)
   - Ratio: Controlled by sidechain gain

   3. Optical Compressor:

   Vin ──[LDR sim]──[Amp]── Vout
             │               │
             │    ┌──────────┘
             │    │
            [LED driver]──[LED]
                          (light)
                            │
                         [LDR]

   - LED brightness controls LDR resistance
   - Very slow, smooth response (program dependent)
   - "Musical" compression - loved for vocals, bass

   Compressor Parameters:
   ┌─────────────┬──────────────────────────────────┐
   │ Threshold   │ Level where compression starts   │
   │ Ratio       │ Input change / Output change     │
   │ Attack      │ Time to reach full compression   │
   │ Release     │ Time to return to unity gain     │
   │ Knee        │ Soft (gradual) vs Hard (abrupt)  │
   │ Makeup Gain │ Compensate for gain reduction    │
   └─────────────┴──────────────────────────────────┘

   Applications:
   - Music production (vocals, drums, master bus)
   - Broadcasting (consistent levels)
   - Live sound reinforcement
   - Hearing aids
   - Telecommunications
""")

    print("   Diode Limiter Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Diode Limiter Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Soft-Knee Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Optical Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
