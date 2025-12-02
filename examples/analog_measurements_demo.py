"""Example: Analog Circuit Measurements (PSRR, CMRR, Noise)

This example demonstrates how to use SpiceLab's analog measurement functions
to characterize circuit performance:

1. PSRR (Power Supply Rejection Ratio) - How well supply noise is rejected
2. CMRR (Common Mode Rejection Ratio) - How well common-mode signals are rejected
3. Noise Analysis - Spot noise and integrated noise measurements

These measurements are fundamental for analog circuit design and are
commonly specified in op-amp datasheets.
"""

import numpy as np
import xarray as xr
from spicelab.measurements import (
    CMRRSweepMeasurement,
    IntegratedNoiseMeasurement,
    MeasurementRegistry,
    NoiseAnalysisMeasurement,
    PSRRSweepMeasurement,
    SpotNoiseMeasurement,
    measure_cmrr,
    measure_noise,
    measure_psrr,
)


def create_psrr_dataset():
    """Create mock AC dataset for PSRR measurement.

    In real usage, this data would come from an AC simulation with
    the supply voltage as the AC stimulus.

    Circuit setup for PSRR measurement:
    - Apply AC signal to VDD (e.g., 1V AC at 0° phase)
    - Ground or DC bias the input
    - Measure AC response at output

    PSRR = 20*log10(ΔVdd / ΔVout)
    """
    # Frequency sweep: 1 Hz to 1 MHz (100 points, log spacing)
    frequencies = np.logspace(0, 6, 100)

    # Supply voltage (AC stimulus) - constant 1V AC
    v_supply = np.ones(len(frequencies), dtype=complex)

    # Output voltage response
    # Model: PSRR of ~80dB at DC, decreasing at -20dB/decade above 10kHz
    psrr_db = 80 - 20 * np.log10(1 + (frequencies / 10000) ** 2) / 2
    psrr_linear = 10 ** (psrr_db / 20)
    v_out = v_supply / psrr_linear

    return xr.Dataset(
        {
            "vout": (["frequency"], v_out),
            "vdd": (["frequency"], v_supply),
        },
        coords={"frequency": frequencies},
    )


def create_cmrr_dataset():
    """Create mock AC dataset for CMRR measurement.

    In real usage, this requires two AC analyses:
    1. Differential mode: Input = Vdiff, Output = Vout_diff
    2. Common mode: Input = Vcm, Output = Vout_cm

    CMRR = 20*log10(Adiff / Acm)
    """
    frequencies = np.logspace(0, 6, 100)

    # Input voltage (1V AC)
    v_in = np.ones(len(frequencies), dtype=complex)

    # Differential gain: 1000 V/V (60 dB), flat response
    a_diff = 1000.0
    v_diff = a_diff * v_in

    # Common-mode gain: very small, increasing at high frequency
    # At DC: 0.001 V/V (-60 dB), giving CMRR of 120 dB
    a_cm_dc = 0.001
    a_cm = a_cm_dc * np.sqrt(1 + (frequencies / 50000) ** 2)
    v_cm = a_cm * v_in

    return xr.Dataset(
        {
            "vout_diff": (["frequency"], v_diff),
            "vout_cm": (["frequency"], v_cm),
            "vin": (["frequency"], v_in),
        },
        coords={"frequency": frequencies},
    )


def create_noise_dataset():
    """Create mock noise analysis dataset.

    In real usage, this data comes from a SPICE .noise analysis.

    Noise model: 1/f noise + white noise floor
    - 1/f noise dominates at low frequencies
    - White noise dominates at high frequencies
    """
    # Frequency sweep: 1 Hz to 1 MHz
    frequencies = np.logspace(0, 6, 500)

    # Noise model parameters (typical op-amp values)
    # 1/f noise: ~10 nV/sqrt(Hz) at 10 Hz
    # White noise floor: ~3 nV/sqrt(Hz)
    k_1f = 1e-15  # 1/f noise coefficient (V²/Hz)
    k_white = 9e-18  # White noise floor (V²/Hz) -> 3 nV/√Hz

    # Total noise spectral density
    noise_psd = np.sqrt(k_1f / frequencies + k_white)

    return xr.Dataset(
        {
            "onoise": (["frequency"], noise_psd),
        },
        coords={"frequency": frequencies},
    )


def example_psrr_measurement():
    """Demonstrate PSRR measurement."""
    print("=" * 60)
    print("Example 1: PSRR Measurement")
    print("=" * 60)

    dataset = create_psrr_dataset()

    # Method 1: Use measurement class directly
    print("\n1. Using PSRRSweepMeasurement class:")
    meas = PSRRSweepMeasurement(
        output_node="vout",
        supply_node="vdd",
        frequency=1000,  # Measure at 1 kHz
    )
    result = meas.measure(dataset)
    print(f"   PSRR at 1kHz: {result.value:.1f} dB")

    # Method 2: Use convenience function
    print("\n2. Using measure_psrr() convenience function:")
    psrr_result = measure_psrr(dataset, "vout", "vdd", frequency=1000)
    print(f"   PSRR: {psrr_result.value_db:.1f} dB at {psrr_result.frequency:.0f} Hz")
    print(f"   Vout magnitude: {psrr_result.v_out:.2e} V")
    print(f"   Vsupply magnitude: {psrr_result.v_supply:.2f} V")

    # Method 3: Get frequency sweep data
    print("\n3. PSRR vs Frequency sweep:")
    meas_sweep = PSRRSweepMeasurement(
        output_node="vout",
        supply_node="vdd",
        frequency=1000,
        return_sweep=True,
    )
    result_sweep = meas_sweep.measure(dataset)

    # Show PSRR at different frequencies
    freqs = result_sweep.metadata["frequencies"]
    psrr_vals = result_sweep.metadata["psrr_vs_freq"]
    freq_points = [10, 100, 1000, 10000, 100000]
    for f in freq_points:
        idx = np.argmin(np.abs(np.array(freqs) - f))
        actual_f = freqs[idx]
        p = psrr_vals[idx]
        print(f"   PSRR at {actual_f:.0f} Hz: {p:.1f} dB")


def example_cmrr_measurement():
    """Demonstrate CMRR measurement."""
    print("\n" + "=" * 60)
    print("Example 2: CMRR Measurement")
    print("=" * 60)

    dataset = create_cmrr_dataset()

    # Method 1: From dataset with differential and common-mode outputs
    print("\n1. Using CMRRSweepMeasurement class:")
    meas = CMRRSweepMeasurement(
        diff_output_node="vout_diff",
        cm_output_node="vout_cm",
        frequency=100,  # Near DC
    )
    result = meas.measure(dataset)
    print(f"   CMRR at 100Hz: {result.value:.1f} dB")
    print(f"   Differential gain: {result.metadata['diff_gain']:.1f} V/V")
    print(f"   Common-mode gain: {result.metadata['cm_gain']:.2e} V/V")

    # Method 2: Use convenience function
    print("\n2. Using measure_cmrr() convenience function:")
    cmrr_result = measure_cmrr(dataset, "vout_diff", "vout_cm", frequency=100)
    print(f"   CMRR: {cmrr_result.value_db:.1f} dB")

    # Method 3: From pre-computed gains (useful when you have datasheet values)
    print("\n3. From pre-computed gains:")
    meas_precomputed = CMRRSweepMeasurement(
        diff_gain=100000,  # 100 V/mV = 100,000 V/V
        cm_gain=10,  # 10 mV/V = 0.01 V/V (normalized)
    )
    result_precomputed = meas_precomputed.measure(dataset)  # Dataset ignored
    print(f"   CMRR from gains: {result_precomputed.value:.1f} dB")
    print("   (100dB for Adiff=100,000, Acm=10)")

    # Method 4: CMRR vs frequency
    print("\n4. CMRR vs Frequency:")
    meas_sweep = CMRRSweepMeasurement(
        diff_output_node="vout_diff",
        cm_output_node="vout_cm",
        frequency=100,
        return_sweep=True,
    )
    result_sweep = meas_sweep.measure(dataset)

    freqs = result_sweep.metadata["frequencies"]
    cmrr_vals = result_sweep.metadata["cmrr_vs_freq"]
    freq_points = [10, 100, 1000, 10000, 100000]
    for f in freq_points:
        idx = np.argmin(np.abs(np.array(freqs) - f))
        actual_f = freqs[idx]
        c = cmrr_vals[idx]
        print(f"   CMRR at {actual_f:.0f} Hz: {c:.1f} dB")


def example_noise_measurement():
    """Demonstrate noise measurement."""
    print("\n" + "=" * 60)
    print("Example 3: Noise Analysis")
    print("=" * 60)

    dataset = create_noise_dataset()

    # Method 1: Complete noise analysis
    print("\n1. Using NoiseAnalysisMeasurement class:")
    meas = NoiseAnalysisMeasurement(
        noise_node="onoise",
        frequency=1000,  # Spot noise frequency
        freq_min=10,  # Integration lower bound
        freq_max=100000,  # Integration upper bound
    )
    result = meas.measure(dataset)
    print(f"   Spot noise at 1kHz: {result.metadata['spot_noise']:.2e} V/√Hz")
    print(f"   Integrated noise (10Hz-100kHz): {result.value:.2e} V rms")

    # Method 2: Spot noise only
    print("\n2. Spot noise at different frequencies:")
    for f in [10, 100, 1000, 10000]:
        spot_meas = SpotNoiseMeasurement(noise_node="onoise", frequency=f)
        spot_result = spot_meas.measure(dataset)
        print(f"   Noise at {f} Hz: {spot_result.value:.2e} V/√Hz")

    # Method 3: Integrated noise with different bandwidths
    print("\n3. Integrated noise vs bandwidth:")
    bandwidths = [(10, 1000), (10, 10000), (10, 100000), (0.1, 100000)]
    for f_min, f_max in bandwidths:
        int_meas = IntegratedNoiseMeasurement(noise_node="onoise", freq_min=f_min, freq_max=f_max)
        int_result = int_meas.measure(dataset)
        bw_str = f"{f_min} - {f_max} Hz"
        print(f"   {bw_str:20s}: {int_result.value:.2e} V rms")

    # Method 4: Convenience function with gain for input-referred noise
    print("\n4. Using measure_noise() with gain:")
    gain = 100  # Amplifier gain
    noise_result = measure_noise(
        dataset,
        noise_node="onoise",
        frequency=1000,
        freq_min=10,
        freq_max=100000,
        gain=gain,
    )
    print(f"   Output noise: {noise_result.integrated_noise:.2e} V rms")
    if noise_result.equivalent_input_noise:
        ein = noise_result.equivalent_input_noise
        print(f"   Input-referred noise: {ein:.2e} V rms")
        print(f"   (Output noise / gain = {noise_result.integrated_noise/gain:.2e})")


def example_registry_usage():
    """Show how to use measurements via the registry."""
    print("\n" + "=" * 60)
    print("Example 4: Using MeasurementRegistry")
    print("=" * 60)

    print("\n1. List available analog measurements:")
    all_measurements = MeasurementRegistry.list_all()
    analog_measurements = [
        "psrr_sweep",
        "cmrr_sweep",
        "noise_analysis",
        "spot_noise",
        "integrated_noise",
    ]
    for name in analog_measurements:
        if name in all_measurements:
            cls = MeasurementRegistry.get(name)
            print(f"   - {name}: {cls.description}")

    print("\n2. Get measurement by name:")
    psrr_cls = MeasurementRegistry.get("psrr_sweep")
    print(f"   psrr_sweep class: {psrr_cls.__name__}")
    print(f"   Required analyses: {psrr_cls.required_analyses}")


def example_interpretation():
    """Provide guidance on interpreting measurement results."""
    print("\n" + "=" * 60)
    print("Interpreting Measurement Results")
    print("=" * 60)

    print(
        """
PSRR (Power Supply Rejection Ratio):
    - Typical good values: > 60 dB at 1 kHz
    - High-end op-amps: > 100 dB at DC
    - Decreases at high frequency (typically -20 dB/decade)
    - Important for: audio circuits, precision sensors

CMRR (Common-Mode Rejection Ratio):
    - Typical good values: > 80 dB at DC
    - High-end instrumentation amps: > 120 dB
    - Decreases at high frequency
    - Important for: differential sensors, instrumentation

Noise:
    - Spot noise: Shows frequency-dependent behavior
      - 1/f region: Increases at low frequency
      - White noise floor: Constant at high frequency
    - Integrated noise: Total RMS noise in bandwidth
      - Depends on bandwidth (wider = more noise)
      - Use for SNR calculations
    - Input-referred: Allows fair comparison between amps

Typical Datasheet Specs:
    - Voltage noise: 3-20 nV/√Hz (low noise op-amps)
    - 1/f corner: 1-100 Hz
    - Current noise: 0.1-10 pA/√Hz
    """
    )


if __name__ == "__main__":
    example_psrr_measurement()
    example_cmrr_measurement()
    example_noise_measurement()
    example_registry_usage()
    example_interpretation()

    print("\n" + "=" * 60)
    print("Analog Measurements Demo Complete!")
    print("=" * 60)
