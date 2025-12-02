"""Tests for analog measurements (PSRR, CMRR, Noise)."""

import numpy as np
import pytest
import xarray as xr
from spicelab.measurements import (
    CMRRResult,
    CMRRSweepMeasurement,
    IntegratedNoiseMeasurement,
    MeasurementRegistry,
    NoiseAnalysisMeasurement,
    NoiseResult,
    PSRRResult,
    PSRRSweepMeasurement,
    SpotNoiseMeasurement,
    measure_cmrr,
    measure_noise,
    measure_psrr,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def ac_dataset_psrr():
    """Create a mock AC dataset for PSRR testing.

    Simulates an amplifier where supply noise is attenuated at the output.
    """
    # Frequency sweep from 1Hz to 1MHz
    frequencies = np.logspace(0, 6, 100)

    # Supply voltage (AC stimulus) - constant 1V AC
    v_supply = np.ones(len(frequencies), dtype=complex)

    # Output voltage - PSRR of ~60dB at low frequencies, decreasing at high frequencies
    # PSRR = 60dB means output is 1000x smaller than supply variation
    psrr_db = 60 - 10 * np.log10(1 + (frequencies / 10000) ** 2)  # -3dB at 10kHz
    psrr_linear = 10 ** (psrr_db / 20)
    v_out = v_supply / psrr_linear

    return xr.Dataset(
        {
            "vout": (["frequency"], v_out),
            "vdd": (["frequency"], v_supply),
        },
        coords={"frequency": frequencies},
    )


@pytest.fixture
def ac_dataset_cmrr():
    """Create a mock AC dataset for CMRR testing.

    Simulates a differential amplifier with differential and common-mode outputs.
    """
    frequencies = np.logspace(0, 6, 100)

    # Input voltage (1V AC)
    v_in = np.ones(len(frequencies), dtype=complex)

    # Differential gain: 100 V/V (40dB), flat
    a_diff = 100.0
    v_diff = a_diff * v_in

    # Common-mode gain: 0.001 V/V (-60dB) at DC, increasing at high frequency
    # CMRR = 100dB at DC (100 / 0.001 = 100,000)
    a_cm = 0.001 * np.sqrt(1 + (frequencies / 100000) ** 2)
    v_cm = a_cm * v_in

    return xr.Dataset(
        {
            "vout_diff": (["frequency"], v_diff),
            "vout_cm": (["frequency"], v_cm),
            "vin": (["frequency"], v_in),
        },
        coords={"frequency": frequencies},
    )


@pytest.fixture
def noise_dataset():
    """Create a mock noise analysis dataset.

    Simulates typical 1/f + white noise characteristic.
    """
    frequencies = np.logspace(0, 5, 200)  # 1Hz to 100kHz

    # Noise model: 1/f noise + white noise
    # onoise = sqrt(K_1f / f + K_white)
    k_1f = 1e-14  # 1/f noise coefficient (V^2/Hz)
    k_white = 1e-16  # White noise floor (V^2/Hz)

    noise_psd = np.sqrt(k_1f / frequencies + k_white)  # V/sqrt(Hz)

    return xr.Dataset(
        {
            "onoise": (["frequency"], noise_psd),
            "inoise": (["frequency"], noise_psd / 100),  # Input-referred (gain=100)
        },
        coords={"frequency": frequencies},
    )


# ============================================================================
# PSRR Measurement Tests
# ============================================================================


class TestPSRRMeasurement:
    """Tests for PSRR measurement."""

    def test_psrr_basic(self, ac_dataset_psrr):
        """Test basic PSRR measurement at 1kHz."""
        meas = PSRRSweepMeasurement(
            output_node="vout",
            supply_node="vdd",
            frequency=1000,
        )
        result = meas.measure(ac_dataset_psrr)

        # PSRR should be ~60dB at low frequencies
        assert result.value > 55  # Allow some tolerance
        assert result.value < 65
        assert result.unit == "dB"
        assert "frequency" in result.metadata

    def test_psrr_high_frequency(self, ac_dataset_psrr):
        """Test PSRR at high frequency (should be lower)."""
        meas_low = PSRRSweepMeasurement(output_node="vout", supply_node="vdd", frequency=100)
        meas_high = PSRRSweepMeasurement(output_node="vout", supply_node="vdd", frequency=100000)

        result_low = meas_low.measure(ac_dataset_psrr)
        result_high = meas_high.measure(ac_dataset_psrr)

        # PSRR should decrease at high frequency
        assert result_high.value < result_low.value

    def test_psrr_sweep_metadata(self, ac_dataset_psrr):
        """Test PSRR with frequency sweep data in metadata."""
        meas = PSRRSweepMeasurement(
            output_node="vout",
            supply_node="vdd",
            frequency=1000,
            return_sweep=True,
        )
        result = meas.measure(ac_dataset_psrr)

        assert "frequencies" in result.metadata
        assert "psrr_vs_freq" in result.metadata
        assert len(result.metadata["frequencies"]) == 100
        assert len(result.metadata["psrr_vs_freq"]) == 100

    def test_measure_psrr_convenience(self, ac_dataset_psrr):
        """Test convenience function measure_psrr."""
        result = measure_psrr(ac_dataset_psrr, "vout", "vdd", frequency=1000)

        assert isinstance(result, PSRRResult)
        assert result.value_db > 50
        assert result.frequency == pytest.approx(1000, rel=0.1)
        assert result.frequencies is not None
        assert len(result.frequencies) > 0

    def test_psrr_registered(self):
        """Test that PSRR sweep is registered in MeasurementRegistry."""
        cls = MeasurementRegistry.get("psrr_sweep")
        assert cls is not None
        assert cls.name == "psrr_sweep"


# ============================================================================
# CMRR Measurement Tests
# ============================================================================


class TestCMRRMeasurement:
    """Tests for CMRR measurement."""

    def test_cmrr_basic(self, ac_dataset_cmrr):
        """Test basic CMRR measurement."""
        meas = CMRRSweepMeasurement(
            diff_output_node="vout_diff",
            cm_output_node="vout_cm",
            frequency=100,
        )
        result = meas.measure(ac_dataset_cmrr)

        # CMRR should be ~100dB at low frequencies (100 / 0.001)
        assert result.value > 90  # Allow tolerance
        assert result.value < 110
        assert result.unit == "dB"

    def test_cmrr_with_input_node(self, ac_dataset_cmrr):
        """Test CMRR with input node for gain calculation."""
        meas = CMRRSweepMeasurement(
            diff_output_node="vout_diff",
            cm_output_node="vout_cm",
            input_node="vin",
            frequency=100,
        )
        result = meas.measure(ac_dataset_cmrr)

        # Same result since input is 1V
        assert result.value > 90
        assert "diff_gain" in result.metadata
        assert "cm_gain" in result.metadata

    def test_cmrr_precomputed_gains(self, ac_dataset_cmrr):
        """Test CMRR with pre-computed gains."""
        meas = CMRRSweepMeasurement(
            diff_gain=1000,  # 60dB
            cm_gain=1,  # 0dB
        )
        result = meas.measure(ac_dataset_cmrr)  # Dataset ignored

        # CMRR = 60dB (1000/1)
        assert result.value == pytest.approx(60, rel=0.01)
        assert result.metadata["mode"] == "pre_computed"

    def test_cmrr_high_frequency(self, ac_dataset_cmrr):
        """Test CMRR at high frequency (should be lower)."""
        meas_low = CMRRSweepMeasurement(
            diff_output_node="vout_diff",
            cm_output_node="vout_cm",
            frequency=10,
        )
        meas_high = CMRRSweepMeasurement(
            diff_output_node="vout_diff",
            cm_output_node="vout_cm",
            frequency=500000,
        )

        result_low = meas_low.measure(ac_dataset_cmrr)
        result_high = meas_high.measure(ac_dataset_cmrr)

        # CMRR should decrease at high frequency
        assert result_high.value < result_low.value

    def test_measure_cmrr_convenience(self, ac_dataset_cmrr):
        """Test convenience function measure_cmrr."""
        result = measure_cmrr(ac_dataset_cmrr, "vout_diff", "vout_cm", frequency=100)

        assert isinstance(result, CMRRResult)
        assert result.value_db > 90
        assert result.diff_gain > 0
        assert result.cm_gain > 0

    def test_cmrr_registered(self):
        """Test that CMRR sweep is registered in MeasurementRegistry."""
        cls = MeasurementRegistry.get("cmrr_sweep")
        assert cls is not None
        assert cls.name == "cmrr_sweep"


# ============================================================================
# Noise Measurement Tests
# ============================================================================


class TestNoiseMeasurement:
    """Tests for noise analysis measurement."""

    def test_noise_analysis_basic(self, noise_dataset):
        """Test basic noise analysis."""
        meas = NoiseAnalysisMeasurement(
            noise_node="onoise",
            frequency=1000,
            freq_min=10,
            freq_max=100000,
        )
        result = meas.measure(noise_dataset)

        assert result.value > 0  # Integrated noise should be positive
        assert result.unit == "V rms"
        assert "spot_noise" in result.metadata
        assert result.metadata["spot_noise"] > 0

    def test_spot_noise(self, noise_dataset):
        """Test spot noise measurement."""
        meas = SpotNoiseMeasurement(noise_node="onoise", frequency=1000)
        result = meas.measure(noise_dataset)

        assert result.value > 0
        assert result.unit == "V/sqrt(Hz)"
        assert result.metadata["frequency"] == pytest.approx(1000, rel=0.1)

    def test_spot_noise_1f_region(self, noise_dataset):
        """Test that 1/f noise is higher at low frequency."""
        meas_low = SpotNoiseMeasurement(noise_node="onoise", frequency=10)
        meas_high = SpotNoiseMeasurement(noise_node="onoise", frequency=10000)

        result_low = meas_low.measure(noise_dataset)
        result_high = meas_high.measure(noise_dataset)

        # 1/f region: noise should be higher at low frequency
        assert result_low.value > result_high.value

    def test_integrated_noise(self, noise_dataset):
        """Test integrated noise measurement."""
        meas = IntegratedNoiseMeasurement(
            noise_node="onoise",
            freq_min=10,
            freq_max=100000,
        )
        result = meas.measure(noise_dataset)

        assert result.value > 0
        assert result.unit == "V rms"
        assert result.metadata["bandwidth"] == 100000 - 10

    def test_integrated_noise_bandwidth_effect(self, noise_dataset):
        """Test that wider bandwidth gives higher integrated noise."""
        meas_narrow = IntegratedNoiseMeasurement(noise_node="onoise", freq_min=100, freq_max=1000)
        meas_wide = IntegratedNoiseMeasurement(noise_node="onoise", freq_min=10, freq_max=100000)

        result_narrow = meas_narrow.measure(noise_dataset)
        result_wide = meas_wide.measure(noise_dataset)

        # Wider bandwidth should have more integrated noise
        assert result_wide.value > result_narrow.value

    def test_noise_with_gain(self, noise_dataset):
        """Test noise with gain for input-referred calculation."""
        meas = NoiseAnalysisMeasurement(
            noise_node="onoise",
            frequency=1000,
            freq_min=10,
            freq_max=100000,
            gain=100,
        )
        result = meas.measure(noise_dataset)

        assert "input_referred_noise" in result.metadata
        # Input-referred should be output / gain
        expected_input_referred = result.value / 100
        actual = result.metadata["input_referred_noise"]
        assert actual == pytest.approx(expected_input_referred, rel=0.01)

    def test_noise_spectrum_return(self, noise_dataset):
        """Test that spectrum is returned when requested."""
        meas = NoiseAnalysisMeasurement(
            noise_node="onoise",
            frequency=1000,
            freq_min=10,
            freq_max=100000,
            return_spectrum=True,
        )
        result = meas.measure(noise_dataset)

        assert "frequencies" in result.metadata
        assert "noise_density" in result.metadata
        assert len(result.metadata["frequencies"]) == 200
        assert len(result.metadata["noise_density"]) == 200

    def test_measure_noise_convenience(self, noise_dataset):
        """Test convenience function measure_noise."""
        result = measure_noise(
            noise_dataset,
            "onoise",
            frequency=1000,
            freq_min=10,
            freq_max=100000,
        )

        assert isinstance(result, NoiseResult)
        assert result.spot_noise > 0
        assert result.integrated_noise > 0
        assert result.frequencies is not None
        assert result.noise_density is not None

    def test_noise_registered(self):
        """Test that noise measurements are registered."""
        for name in ["noise_analysis", "spot_noise", "integrated_noise"]:
            cls = MeasurementRegistry.get(name)
            assert cls is not None, f"{name} not registered"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in measurements."""

    def test_psrr_missing_frequency_coord(self):
        """Test PSRR raises error when frequency coordinate missing."""
        dataset = xr.Dataset(
            {"vout": (["time"], np.zeros(10))},
            coords={"time": np.linspace(0, 1, 10)},
        )
        meas = PSRRSweepMeasurement("vout", "vdd", frequency=1000)

        with pytest.raises(ValueError, match="frequency"):
            meas.measure(dataset)

    def test_psrr_missing_signal(self, ac_dataset_psrr):
        """Test PSRR raises error when signal not found."""
        meas = PSRRSweepMeasurement("nonexistent", "vdd", frequency=1000)

        with pytest.raises(KeyError, match="nonexistent"):
            meas.measure(ac_dataset_psrr)

    def test_cmrr_missing_required_params(self, ac_dataset_cmrr):
        """Test CMRR raises error when missing required parameters."""
        meas = CMRRSweepMeasurement(
            diff_output_node="vout_diff",
            # Missing cm_output_node
        )

        with pytest.raises(ValueError, match="Must provide"):
            meas.measure(ac_dataset_cmrr)

    def test_noise_invalid_frequency_range(self, noise_dataset):
        """Test noise raises error for invalid frequency range."""
        meas = IntegratedNoiseMeasurement(
            noise_node="onoise",
            freq_min=1000000,  # Higher than max in dataset
            freq_max=2000000,
        )

        with pytest.raises(ValueError, match="No frequencies in range"):
            meas.measure(noise_dataset)


# ============================================================================
# Result Dataclass Tests
# ============================================================================


class TestResultDataclasses:
    """Tests for result dataclasses."""

    def test_psrr_result_str(self):
        """Test PSRRResult string representation."""
        result = PSRRResult(value_db=60.5, frequency=1000)
        assert "60.5" in str(result)
        assert "dB" in str(result)
        assert "1000" in str(result)

    def test_psrr_result_passed_property(self):
        """Test PSRRResult passed property."""
        good = PSRRResult(value_db=65, frequency=1000)
        bad = PSRRResult(value_db=50, frequency=1000)

        assert good.passed is True  # >60dB at 1kHz
        assert bad.passed is False

    def test_cmrr_result_str(self):
        """Test CMRRResult string representation."""
        result = CMRRResult(value_db=100, frequency=100)
        assert "100" in str(result)
        assert "dB" in str(result)

    def test_cmrr_result_passed_property(self):
        """Test CMRRResult passed property."""
        good = CMRRResult(value_db=90, frequency=100)
        bad = CMRRResult(value_db=70, frequency=100)

        assert good.passed is True  # >80dB at DC
        assert bad.passed is False

    def test_noise_result_str(self):
        """Test NoiseResult string representation."""
        result = NoiseResult(
            spot_noise=1e-9,
            integrated_noise=1e-6,
            frequency=1000,
            freq_min=10,
            freq_max=100000,
        )
        s = str(result)
        assert "Spot noise" in s
        assert "Integrated" in s
