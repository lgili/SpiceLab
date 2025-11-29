"""Tests for the measurement framework (M7).

Tests cover:
- Base classes (MeasurementResult, BaseMeasurement)
- Registry and decorator
- AC measurements
- Transient measurements
- Spectrum measurements
- Digital measurements
- Power measurements
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spicelab.measurements import (
    BaseMeasurement,
    MeasurementRegistry,
    MeasurementResult,
    measure,
    measurement,
)
from spicelab.measurements.ac import (
    BandwidthMeasurement,
    GainMeasurement,
    GainMarginMeasurement,
    PhaseMarginMeasurement,
)
from spicelab.measurements.digital import (
    DutyCycleMeasurement,
    EyeDiagram,
    EyeOpeningMeasurement,
    JitterMeasurement,
)
from spicelab.measurements.power import (
    EfficiencyMeasurement,
    LoadTransientMeasurement,
    PDNImpedanceMeasurement,
    RippleMeasurement,
)
from spicelab.measurements.spectrum import (
    ENOBMeasurement,
    SFDRMeasurement,
    SINADMeasurement,
    SNRMeasurement,
    THDMeasurement,
)
from spicelab.measurements.transient import (
    FallTimeMeasurement,
    OvershootMeasurement,
    PropagationDelayMeasurement,
    RiseTimeMeasurement,
    SettlingTimeMeasurement,
    SlewRateMeasurement,
)


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def ac_dataset() -> xr.Dataset:
    """Create a simple AC analysis dataset (lowpass filter response)."""
    freq = np.logspace(1, 6, 100)  # 10 Hz to 1 MHz

    # Simple first-order lowpass with fc = 10kHz, gain = 10 (20 dB)
    fc = 10000
    gain = 10
    h = gain / (1 + 1j * freq / fc)

    # Create dataset
    ds = xr.Dataset(
        {
            "V(out)": (["frequency"], h),
            "V(in)": (["frequency"], np.ones_like(freq, dtype=complex)),
        },
        coords={"frequency": freq},
    )
    return ds


@pytest.fixture
def tran_dataset() -> xr.Dataset:
    """Create a simple transient dataset (step response)."""
    dt = 1e-6  # 1 µs timestep
    t = np.arange(0, 1e-3, dt)  # 1 ms total

    # Step response with overshoot (second-order underdamped)
    tau = 50e-6  # 50 µs time constant
    zeta = 0.5  # damping ratio
    omega_n = 1 / tau
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    # Step from 0 to 5V at t=100µs
    step_time = 100e-6
    v_final = 5.0

    v = np.zeros_like(t)
    mask = t >= step_time
    t_step = t[mask] - step_time
    v[mask] = v_final * (1 - np.exp(-zeta * omega_n * t_step) *
                         (np.cos(omega_d * t_step) +
                          zeta / np.sqrt(1 - zeta**2) * np.sin(omega_d * t_step)))

    ds = xr.Dataset(
        {"V(out)": (["time"], v)},
        coords={"time": t},
    )
    return ds


@pytest.fixture
def sine_dataset() -> xr.Dataset:
    """Create a transient dataset with a sine wave (for THD, SNR tests)."""
    fs = 100e3  # 100 kHz sample rate
    duration = 10e-3  # 10 ms
    t = np.arange(0, duration, 1/fs)

    # 1 kHz sine with harmonics
    f0 = 1000  # 1 kHz fundamental
    amplitude = 1.0

    # Fundamental + harmonics (simulate distortion)
    v = amplitude * np.sin(2 * np.pi * f0 * t)
    v += 0.05 * amplitude * np.sin(2 * np.pi * 2 * f0 * t)  # 5% 2nd harmonic
    v += 0.03 * amplitude * np.sin(2 * np.pi * 3 * f0 * t)  # 3% 3rd harmonic
    v += 0.01 * np.random.randn(len(t))  # 1% noise

    ds = xr.Dataset(
        {"V(out)": (["time"], v)},
        coords={"time": t},
    )
    return ds


@pytest.fixture
def clock_dataset() -> xr.Dataset:
    """Create a clock signal for jitter and duty cycle tests."""
    fs = 1e9  # 1 GHz sample rate
    duration = 100e-6  # 100 µs
    t = np.arange(0, duration, 1/fs)

    # 10 MHz clock with slight jitter
    f_clk = 10e6
    period = 1 / f_clk

    # Generate square wave with jitter
    np.random.seed(42)
    jitter_rms = 100e-12  # 100 ps RMS jitter

    v = np.zeros_like(t)
    edge_time = 0.0
    high = True

    while edge_time < duration:
        # Add jitter to edge timing
        jittered_period = period / 2 + np.random.randn() * jitter_rms
        next_edge = edge_time + jittered_period

        mask = (t >= edge_time) & (t < next_edge)
        v[mask] = 3.3 if high else 0.0

        edge_time = next_edge
        high = not high

    ds = xr.Dataset(
        {"V(clk)": (["time"], v)},
        coords={"time": t},
    )
    return ds


@pytest.fixture
def power_tran_dataset() -> xr.Dataset:
    """Create a power supply transient dataset."""
    dt = 1e-6
    t = np.arange(0, 1e-3, dt)

    # 3.3V supply with 100kHz ripple
    v_dc = 3.3
    ripple_freq = 100e3
    ripple_amp = 0.05  # 50 mV ripple

    v = v_dc + ripple_amp * np.sin(2 * np.pi * ripple_freq * t)

    ds = xr.Dataset(
        {"V(vdd)": (["time"], v)},
        coords={"time": t},
    )
    return ds


# =============================================================================
# Test MeasurementResult
# =============================================================================


class TestMeasurementResult:
    """Tests for MeasurementResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic MeasurementResult creation."""
        result = MeasurementResult(value=42.0, unit="dB")
        assert result.value == 42.0
        assert result.unit == "dB"
        assert result.passed is None
        assert result.metadata == {}

    def test_with_pass_fail(self) -> None:
        """Test MeasurementResult with pass/fail status."""
        result = MeasurementResult(value=45.0, unit="degrees", passed=True)
        assert result.passed is True
        assert "PASS" in str(result)

    def test_with_metadata(self) -> None:
        """Test MeasurementResult with metadata."""
        result = MeasurementResult(
            value=10e6,
            unit="Hz",
            metadata={"reference_gain_db": 20.0, "n_db": 3.0},
        )
        assert result.metadata["reference_gain_db"] == 20.0

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = MeasurementResult(value=1.5, unit="V", passed=False)
        d = result.to_dict()
        assert d["value"] == 1.5
        assert d["unit"] == "V"
        assert d["passed"] is False

    def test_str_representation(self) -> None:
        """Test string representation."""
        result = MeasurementResult(value=60.5, unit="dB")
        assert "60.5 dB" in str(result)


# =============================================================================
# Test Registry
# =============================================================================


class TestMeasurementRegistry:
    """Tests for MeasurementRegistry."""

    def test_list_all(self) -> None:
        """Test listing all measurements."""
        all_measurements = MeasurementRegistry.list_all()
        assert len(all_measurements) >= 20  # Should have many measurements
        assert "gain" in all_measurements
        assert "bandwidth" in all_measurements
        assert "thd" in all_measurements

    def test_get_measurement(self) -> None:
        """Test getting a measurement by name."""
        gain_cls = MeasurementRegistry.get("gain")
        assert gain_cls is GainMeasurement

    def test_get_invalid(self) -> None:
        """Test getting non-existent measurement."""
        with pytest.raises(ValueError, match="not found"):
            MeasurementRegistry.get("nonexistent_measurement")

    def test_info(self) -> None:
        """Test getting measurement info."""
        info = MeasurementRegistry.info("bandwidth")
        assert info["name"] == "bandwidth"
        assert "description" in info
        assert "required_analyses" in info

    def test_list_by_analysis(self) -> None:
        """Test filtering measurements by analysis type."""
        ac_measurements = MeasurementRegistry.list_by_analysis("ac")
        assert "gain" in ac_measurements
        assert "bandwidth" in ac_measurements

        tran_measurements = MeasurementRegistry.list_by_analysis("tran")
        assert "thd" in tran_measurements
        assert "settling_time" in tran_measurements

    def test_custom_measurement_decorator(self) -> None:
        """Test registering a custom measurement."""
        # Clear registry first (use a unique name to avoid conflicts)
        test_name = "_test_custom_measurement_xyz123"

        @measurement(test_name)
        class TestCustomMeasurement(BaseMeasurement):
            name = test_name
            description = "Test measurement"
            required_analyses = ["tran"]

            def measure(self, dataset: xr.Dataset) -> MeasurementResult:
                return MeasurementResult(value=1.0, unit="test")

        assert test_name in MeasurementRegistry.list_all()

        # Clean up
        MeasurementRegistry._measurements.pop(test_name, None)


# =============================================================================
# Test AC Measurements
# =============================================================================


class TestACMeasurements:
    """Tests for AC analysis measurements."""

    def test_gain_measurement(self, ac_dataset: xr.Dataset) -> None:
        """Test gain measurement at DC."""
        gain = GainMeasurement(output_node="V(out)", input_node="V(in)", frequency=10)
        result = gain.measure(ac_dataset)

        # Should be close to 20 dB (gain = 10)
        assert result.unit == "dB"
        assert 19.0 < result.value < 21.0

    def test_gain_measurement_vv(self, ac_dataset: xr.Dataset) -> None:
        """Test gain measurement in V/V units."""
        gain = GainMeasurement(output_node="V(out)", frequency=10, unit="V/V")
        result = gain.measure(ac_dataset)

        assert result.unit == "V/V"
        assert 9.0 < result.value < 11.0

    def test_bandwidth_measurement(self, ac_dataset: xr.Dataset) -> None:
        """Test -3dB bandwidth measurement."""
        bw = BandwidthMeasurement(output_node="V(out)", n_db=3.0)
        result = bw.measure(ac_dataset)

        # Should be close to 10 kHz (fc)
        assert result.unit == "Hz"
        assert 8000 < result.value < 12000
        assert result.metadata["reached_target"] is True

    def test_phase_margin(self, ac_dataset: xr.Dataset) -> None:
        """Test phase margin measurement."""
        # For a simple lowpass, PM at unity gain crossover
        pm = PhaseMarginMeasurement(output_node="V(out)", input_node="V(in)")
        result = pm.measure(ac_dataset)

        assert result.unit == "degrees"
        # First-order system has ~90° phase margin
        assert 80 < result.value < 100

    def test_gain_margin(self, ac_dataset: xr.Dataset) -> None:
        """Test gain margin measurement."""
        gm = GainMarginMeasurement(output_node="V(out)", input_node="V(in)")
        result = gm.measure(ac_dataset)

        assert result.unit == "dB"
        # First-order system has infinite gain margin (never reaches -180°)
        # Our implementation returns inf or a large value


# =============================================================================
# Test Transient Measurements
# =============================================================================


class TestTransientMeasurements:
    """Tests for transient analysis measurements."""

    def test_slew_rate(self, tran_dataset: xr.Dataset) -> None:
        """Test slew rate measurement."""
        sr = SlewRateMeasurement(node="V(out)", edge="rising")
        result = sr.measure(tran_dataset)

        assert result.unit == "V/s"
        assert result.value > 0

    def test_settling_time(self, tran_dataset: xr.Dataset) -> None:
        """Test settling time measurement."""
        st = SettlingTimeMeasurement(node="V(out)", final_value=5.0, error_pct=1.0)
        result = st.measure(tran_dataset)

        assert result.unit == "s"
        assert result.value > 0
        assert result.metadata["final_value"] == 5.0

    def test_rise_time(self, tran_dataset: xr.Dataset) -> None:
        """Test rise time measurement."""
        rt = RiseTimeMeasurement(node="V(out)")
        result = rt.measure(tran_dataset)

        assert result.unit == "s"
        assert result.value > 0
        assert result.metadata["low_pct"] == 0.1
        assert result.metadata["high_pct"] == 0.9

    def test_fall_time(self) -> None:
        """Test fall time measurement with falling edge data."""
        # Create falling edge data
        t = np.linspace(0, 1e-6, 1000)
        v = 5.0 * np.exp(-t / 100e-9)  # Exponential decay

        ds = xr.Dataset({"V(out)": (["time"], v)}, coords={"time": t})

        ft = FallTimeMeasurement(node="V(out)")
        result = ft.measure(ds)

        assert result.unit == "s"
        assert result.value > 0

    def test_overshoot(self, tran_dataset: xr.Dataset) -> None:
        """Test overshoot measurement."""
        os = OvershootMeasurement(node="V(out)", target=5.0, reference=0.0)
        result = os.measure(tran_dataset)

        assert result.unit == "%"
        assert result.value >= 0  # Should have some overshoot
        assert result.metadata["target"] == 5.0

    def test_propagation_delay(self) -> None:
        """Test propagation delay measurement."""
        t = np.linspace(0, 1e-6, 1000)
        delay = 50e-9  # 50 ns delay

        v_in = np.heaviside(t - 100e-9, 0.5) * 3.3
        v_out = np.heaviside(t - 100e-9 - delay, 0.5) * 3.3

        ds = xr.Dataset(
            {"V(in)": (["time"], v_in), "V(out)": (["time"], v_out)},
            coords={"time": t},
        )

        pd = PropagationDelayMeasurement(input_node="V(in)", output_node="V(out)")
        result = pd.measure(ds)

        assert result.unit == "s"
        assert 40e-9 < result.value < 60e-9


# =============================================================================
# Test Spectrum Measurements
# =============================================================================


class TestSpectrumMeasurements:
    """Tests for spectrum analysis measurements."""

    def test_thd(self, sine_dataset: xr.Dataset) -> None:
        """Test THD measurement."""
        thd = THDMeasurement(node="V(out)", fundamental_freq=1000)
        result = thd.measure(sine_dataset)

        assert result.unit == "%"
        # We added 5% 2nd + 3% 3rd harmonic, so THD ≈ sqrt(5² + 3²) ≈ 5.8%
        assert 4.0 < result.value < 8.0
        assert result.metadata["fundamental_freq"] == pytest.approx(1000, rel=0.1)

    def test_snr(self, sine_dataset: xr.Dataset) -> None:
        """Test SNR measurement."""
        snr = SNRMeasurement(node="V(out)", signal_freq=1000)
        result = snr.measure(sine_dataset)

        assert result.unit == "dB"
        # Our test signal has harmonics + noise, so SNR won't be very high
        # The measurement should succeed and return a reasonable positive value
        assert result.value > 0
        assert "signal_power" in result.metadata
        assert "noise_power" in result.metadata

    def test_sinad(self, sine_dataset: xr.Dataset) -> None:
        """Test SINAD measurement."""
        sinad = SINADMeasurement(node="V(out)", signal_freq=1000)
        result = sinad.measure(sine_dataset)

        assert result.unit == "dB"
        assert result.value > 0

    def test_sfdr(self, sine_dataset: xr.Dataset) -> None:
        """Test SFDR measurement."""
        sfdr = SFDRMeasurement(node="V(out)", fundamental_freq=1000)
        result = sfdr.measure(sine_dataset)

        assert result.unit == "dBc"
        # The test signal has harmonics + noise which spread across bins
        # SFDR should be positive (fundamental > spur)
        assert result.value > 0
        assert "fundamental_freq" in result.metadata
        assert "max_spur_freq" in result.metadata

    def test_enob(self, sine_dataset: xr.Dataset) -> None:
        """Test ENOB measurement."""
        enob = ENOBMeasurement(node="V(out)", signal_freq=1000)
        result = enob.measure(sine_dataset)

        assert result.unit == "bits"
        assert result.value > 0
        assert "sinad_db" in result.metadata


# =============================================================================
# Test Digital Measurements
# =============================================================================


class TestDigitalMeasurements:
    """Tests for digital signal measurements."""

    def test_jitter(self, clock_dataset: xr.Dataset) -> None:
        """Test jitter measurement."""
        jitter = JitterMeasurement(node="V(clk)", edge="rising")
        result = jitter.measure(clock_dataset)

        assert result.unit == "s"
        assert result.value > 0
        # Should be close to our injected 100 ps jitter
        assert result.value < 1e-9  # Less than 1 ns
        assert "cycle_to_cycle_jitter" in result.metadata
        assert result.metadata["n_edges"] > 10

    def test_duty_cycle(self, clock_dataset: xr.Dataset) -> None:
        """Test duty cycle measurement."""
        dc = DutyCycleMeasurement(node="V(clk)")
        result = dc.measure(clock_dataset)

        assert result.unit == "%"
        # Should be close to 50%
        assert 45 < result.value < 55

    def test_eye_opening(self) -> None:
        """Test eye opening measurement."""
        # Create a simple NRZ signal
        bit_period = 1e-9  # 1 Gbps
        fs = 100e9  # 100 GHz sample rate
        n_bits = 100

        t = np.arange(0, n_bits * bit_period, 1/fs)
        np.random.seed(42)
        bits = np.random.randint(0, 2, n_bits)

        # Generate signal
        v = np.zeros_like(t)
        for i, bit in enumerate(bits):
            mask = (t >= i * bit_period) & (t < (i + 1) * bit_period)
            v[mask] = 3.3 if bit else 0.0

        ds = xr.Dataset({"V(data)": (["time"], v)}, coords={"time": t})

        eye = EyeOpeningMeasurement(node="V(data)", bit_period=bit_period)
        result = eye.measure(ds)

        assert result.unit == "V"
        # Eye should be open (height > 0)
        assert result.value > 0
        assert "eye_width_ui" in result.metadata

    def test_eye_diagram_generation(self) -> None:
        """Test eye diagram data generation."""
        bit_period = 1e-9
        fs = 10e9
        n_bits = 50

        t = np.arange(0, n_bits * bit_period, 1/fs)
        np.random.seed(42)
        bits = np.random.randint(0, 2, n_bits)

        v = np.zeros_like(t)
        for i, bit in enumerate(bits):
            mask = (t >= i * bit_period) & (t < (i + 1) * bit_period)
            v[mask] = 3.3 if bit else 0.0

        ds = xr.Dataset({"V(data)": (["time"], v)}, coords={"time": t})

        eye_gen = EyeDiagram(node="V(data)", bit_period=bit_period, n_ui=2)
        data = eye_gen.generate(ds)

        assert data.n_traces > 0
        assert len(data.time_ui) > 0
        assert data.bit_period == bit_period


# =============================================================================
# Test Power Measurements
# =============================================================================


class TestPowerMeasurements:
    """Tests for power integrity measurements."""

    def test_ripple_pp(self, power_tran_dataset: xr.Dataset) -> None:
        """Test peak-to-peak ripple measurement."""
        ripple = RippleMeasurement(node="V(vdd)")
        result = ripple.measure(power_tran_dataset)

        assert result.unit == "V"
        # Should be close to 100 mV pp (2 * 50 mV amplitude)
        assert 0.09 < result.value < 0.11
        assert result.metadata["measurement_type"] == "peak_to_peak"

    def test_ripple_fft(self, power_tran_dataset: xr.Dataset) -> None:
        """Test FFT-based ripple measurement."""
        ripple = RippleMeasurement(node="V(vdd)", ripple_freq=100e3)
        result = ripple.measure(power_tran_dataset)

        assert result.unit == "V"
        # Should be close to 50 mV amplitude
        assert 0.04 < result.value < 0.06
        assert result.metadata["measurement_type"] == "fft"

    def test_pdn_impedance(self) -> None:
        """Test PDN impedance measurement."""
        # Create AC dataset for PDN
        freq = np.logspace(1, 9, 100)  # 10 Hz to 1 GHz

        # Simple RLC model: Z = R + jwL + 1/jwC
        R = 0.01  # 10 mΩ ESR
        L = 1e-9  # 1 nH
        C = 100e-6  # 100 µF

        w = 2 * np.pi * freq
        Z = R + 1j * w * L + 1 / (1j * w * C)

        # For 1A current, voltage = Z
        V = Z * 1.0

        ds = xr.Dataset(
            {
                "V(vdd)": (["frequency"], V),
                "I(iac)": (["frequency"], np.ones_like(freq, dtype=complex)),
            },
            coords={"frequency": freq},
        )

        pdn = PDNImpedanceMeasurement(
            voltage_node="V(vdd)",
            current_node="I(iac)",
            target_impedance=0.1
        )
        result = pdn.measure(ds)

        assert result.unit == "Ohm"
        assert result.value > 0
        assert "frequency_at_max_z" in result.metadata

    def test_load_transient(self) -> None:
        """Test load transient measurement."""
        dt = 1e-6
        t = np.arange(0, 1e-3, dt)

        # 3.3V supply with droop at t=100µs
        step_time = 100e-6
        v_nominal = 3.3
        droop = 0.2  # 200 mV droop
        recovery_tau = 20e-6

        v = np.full_like(t, v_nominal)
        mask = t >= step_time
        v[mask] = v_nominal - droop * np.exp(-(t[mask] - step_time) / recovery_tau)

        ds = xr.Dataset({"V(vout)": (["time"], v)}, coords={"time": t})

        lt = LoadTransientMeasurement(
            voltage_node="V(vout)",
            step_time=step_time,
            nominal_voltage=v_nominal,
        )
        result = lt.measure(ds)

        assert result.unit == "V"
        # Should measure the droop
        assert 0.15 < result.value < 0.25
        assert result.metadata["deviation_type"] == "droop"
        assert "recovery_time" in result.metadata

    def test_efficiency(self) -> None:
        """Test efficiency measurement."""
        t = np.linspace(0, 1e-3, 1000)

        # Simple buck converter: Vin=12V, Vout=5V, Iout=1A, 90% efficient
        eff_target = 0.90
        v_in = np.full_like(t, 12.0)
        v_out = np.full_like(t, 5.0)
        i_out = np.full_like(t, 1.0)
        i_in = (v_out * i_out) / (v_in * eff_target)

        ds = xr.Dataset(
            {
                "V(vin)": (["time"], v_in),
                "V(vout)": (["time"], v_out),
                "I(vin)": (["time"], i_in),
                "I(rload)": (["time"], i_out),
            },
            coords={"time": t},
        )

        eff = EfficiencyMeasurement(
            output_voltage="V(vout)",
            output_current="I(rload)",
            input_voltage="V(vin)",
            input_current="I(vin)",
        )
        result = eff.measure(ds)

        assert result.unit == "%"
        # Should be close to 90%
        assert 88 < result.value < 92
        assert "p_out" in result.metadata
        assert "p_in" in result.metadata


# =============================================================================
# Test convenience function
# =============================================================================


class TestMeasureFunction:
    """Tests for the measure() convenience function."""

    def test_measure_by_name(self, ac_dataset: xr.Dataset) -> None:
        """Test using measure() with measurement name."""
        result = measure(ac_dataset, "gain", output_node="V(out)", frequency=10)

        assert isinstance(result, MeasurementResult)
        assert result.unit == "dB"

    def test_measure_with_spec(self, ac_dataset: xr.Dataset) -> None:
        """Test measure() with spec checking."""
        result = measure(
            ac_dataset,
            "gain",
            spec={"min": 15, "max": 25},
            output_node="V(out)",
            frequency=10,
        )

        assert result.passed is True  # 20 dB is within [15, 25]

    def test_measure_spec_fail(self, ac_dataset: xr.Dataset) -> None:
        """Test measure() with failing spec."""
        result = measure(
            ac_dataset,
            "gain",
            spec={"min": 30},  # Gain is only ~20 dB
            output_node="V(out)",
            frequency=10,
        )

        assert result.passed is False

    def test_measure_invalid_name(self, ac_dataset: xr.Dataset) -> None:
        """Test measure() with invalid measurement name."""
        with pytest.raises(ValueError, match="not found"):
            measure(ac_dataset, "nonexistent_measurement")


# =============================================================================
# Test BaseMeasurement
# =============================================================================


class TestBaseMeasurement:
    """Tests for BaseMeasurement base class."""

    def test_check_spec_min_max(self) -> None:
        """Test default spec checking with min/max."""
        m = BaseMeasurement()
        result = MeasurementResult(value=50.0, unit="dB")

        assert m.check_spec(result, {"min": 40, "max": 60}) is True
        assert m.check_spec(result, {"min": 60}) is False
        assert m.check_spec(result, {"max": 40}) is False

    def test_check_spec_target_tolerance(self) -> None:
        """Test spec checking with target and tolerance."""
        m = BaseMeasurement()
        result = MeasurementResult(value=50.0, unit="dB")

        assert m.check_spec(result, {"target": 50.0, "tolerance": 5.0}) is True
        assert m.check_spec(result, {"target": 60.0, "tolerance": 5.0}) is False

    def test_measure_not_implemented(self) -> None:
        """Test that base measure() raises NotImplementedError."""
        m = BaseMeasurement()
        ds = xr.Dataset()

        with pytest.raises(NotImplementedError):
            m.measure(ds)
