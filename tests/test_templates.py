"""Tests for circuit templates (M4 DX improvement).

Tests the circuit template functions for filters and amplifiers.
"""

import math

import pytest
from spicelab.templates import (
    AmplifierResult,
    FilterResult,
    differential_amplifier,
    inverting_amplifier,
    non_inverting_amplifier,
    rc_highpass,
    rc_lowpass,
    rlc_bandpass,
    sallen_key_lowpass,
    summing_amplifier,
    voltage_divider,
    voltage_follower,
)


class TestRCLowpass:
    """Tests for RC lowpass filter template."""

    def test_creates_circuit(self):
        """Should create a valid circuit."""
        circuit = rc_lowpass(fc=1000)
        assert circuit.name == "RC_Filter"
        assert len(circuit._components) == 2

    def test_cutoff_frequency_calculation(self):
        """RC values should give correct cutoff frequency."""
        fc = 1000  # 1kHz
        impedance = 10_000  # 10kΩ
        circuit = rc_lowpass(fc=fc, impedance=impedance)

        # Find R and C components
        r_comp = next(c for c in circuit._components if "Resistor" in type(c).__name__)
        c_comp = next(c for c in circuit._components if "Capacitor" in type(c).__name__)

        # Use typed fields for numeric values
        R = r_comp.resistance
        C = c_comp.capacitance

        # fc = 1/(2π RC)
        calculated_fc = 1 / (2 * math.pi * R * C)
        assert abs(calculated_fc - fc) < 1  # Within 1Hz

    def test_generates_valid_netlist(self):
        """Should generate valid SPICE netlist."""
        circuit = rc_lowpass(fc=1000)
        netlist = circuit.build_netlist()

        assert "RC_Filter" in netlist
        assert "R1" in netlist
        assert "C1" in netlist
        assert ".end" in netlist


class TestRCHighpass:
    """Tests for RC highpass filter template."""

    def test_creates_circuit(self):
        """Should create a valid circuit."""
        circuit = rc_highpass(fc=100)
        assert circuit.name == "RC_Highpass"
        assert len(circuit._components) == 2

    def test_cutoff_frequency_calculation(self):
        """RC values should give correct cutoff frequency."""
        fc = 100  # 100Hz
        circuit = rc_highpass(fc=fc, impedance=10_000)

        r_comp = next(c for c in circuit._components if "Resistor" in type(c).__name__)
        c_comp = next(c for c in circuit._components if "Capacitor" in type(c).__name__)

        R = r_comp.resistance
        C = c_comp.capacitance

        calculated_fc = 1 / (2 * math.pi * R * C)
        assert abs(calculated_fc - fc) < 1

    def test_topology_is_highpass(self):
        """Capacitor should be in series (first in chain)."""
        circuit = rc_highpass(fc=100)
        netlist = circuit.build_netlist()

        # In highpass, C connects vin to vout, R connects vout to GND
        assert "C1 vin vout" in netlist
        assert "R1 vout 0" in netlist


class TestRLCBandpass:
    """Tests for RLC bandpass filter template."""

    def test_returns_filter_result(self):
        """Should return FilterResult with circuit and metadata."""
        result = rlc_bandpass(fc=1000, bandwidth=100)

        assert isinstance(result, FilterResult)
        assert result.cutoff_frequency == 1000
        assert result.q_factor == 10.0  # Q = fc / bandwidth

    def test_q_factor_calculation(self):
        """Q factor should equal fc/bandwidth."""
        fc = 5000
        bandwidth = 500
        result = rlc_bandpass(fc=fc, bandwidth=bandwidth)

        expected_q = fc / bandwidth
        assert result.q_factor == expected_q

    def test_components_dict(self):
        """Should provide access to components."""
        result = rlc_bandpass(fc=1000, bandwidth=100)

        assert "R1" in result.components
        assert "L1" in result.components
        assert "C1" in result.components

    def test_circuit_has_three_components(self):
        """RLC filter should have R, L, and C."""
        result = rlc_bandpass(fc=1000, bandwidth=100)
        assert len(result.circuit._components) == 3


class TestSallenKeyLowpass:
    """Tests for Sallen-Key lowpass filter template."""

    def test_returns_filter_result(self):
        """Should return FilterResult."""
        result = sallen_key_lowpass(fc=1000)
        assert isinstance(result, FilterResult)

    def test_default_butterworth_q(self):
        """Default Q should be 0.707 (Butterworth)."""
        result = sallen_key_lowpass(fc=1000)
        assert abs(result.q_factor - 0.707) < 0.001

    def test_custom_q_factor(self):
        """Should accept custom Q factor."""
        result = sallen_key_lowpass(fc=1000, q=1.0)
        assert result.q_factor == 1.0

    def test_has_four_components(self):
        """Sallen-Key should have 2 resistors and 2 capacitors."""
        result = sallen_key_lowpass(fc=1000)
        assert len(result.circuit._components) == 4

    def test_components_accessible(self):
        """Should provide access to R1, R2, C1, C2."""
        result = sallen_key_lowpass(fc=1000)
        assert "R1" in result.components
        assert "R2" in result.components
        assert "C1" in result.components
        assert "C2" in result.components


class TestVoltageDivider:
    """Tests for voltage divider template."""

    def test_creates_circuit(self):
        """Should create valid circuit."""
        circuit = voltage_divider(ratio=0.5)
        assert len(circuit._components) == 2

    def test_ratio_50_percent(self):
        """50% ratio should create equal resistors."""
        circuit = voltage_divider(ratio=0.5, total_resistance=10_000)

        resistors = [c for c in circuit._components if "Resistor" in type(c).__name__]
        values = [r.resistance for r in resistors]

        # Both should be 5kΩ for 50% division
        assert abs(values[0] - 5000) < 1
        assert abs(values[1] - 5000) < 1

    def test_ratio_25_percent(self):
        """25% ratio: R2/(R1+R2) = 0.25, so R2 = 0.25 * total."""
        circuit = voltage_divider(ratio=0.25, total_resistance=10_000)

        resistors = sorted(
            [c for c in circuit._components if "Resistor" in type(c).__name__],
            key=lambda r: r.resistance,
        )

        # R2 = 2.5kΩ (smaller), R1 = 7.5kΩ (larger)
        assert abs(resistors[0].resistance - 2500) < 1
        assert abs(resistors[1].resistance - 7500) < 1

    def test_invalid_ratio_raises(self):
        """Ratio outside (0,1) should raise ValueError."""
        with pytest.raises(ValueError):
            voltage_divider(ratio=0)

        with pytest.raises(ValueError):
            voltage_divider(ratio=1)

        with pytest.raises(ValueError):
            voltage_divider(ratio=1.5)

        with pytest.raises(ValueError):
            voltage_divider(ratio=-0.5)


class TestInvertingAmplifier:
    """Tests for inverting amplifier template."""

    def test_creates_circuit(self):
        """Should create valid circuit."""
        circuit = inverting_amplifier(gain=10)
        assert len(circuit._components) == 2

    def test_gain_10_resistor_values(self):
        """Gain=10 with Rin=10kΩ should give Rf=100kΩ."""
        circuit = inverting_amplifier(gain=10, input_impedance=10_000)

        resistors = {c.ref: c for c in circuit._components}
        assert abs(resistors["in"].resistance - 10_000) < 1
        assert abs(resistors["f"].resistance - 100_000) < 1

    def test_invalid_gain_raises(self):
        """Non-positive gain should raise ValueError."""
        with pytest.raises(ValueError):
            inverting_amplifier(gain=0)

        with pytest.raises(ValueError):
            inverting_amplifier(gain=-5)

    def test_netlist_has_vin_vminus_vout(self):
        """Netlist should have vin, vminus, vout nodes."""
        circuit = inverting_amplifier(gain=10)
        netlist = circuit.build_netlist()

        assert "vin" in netlist
        assert "vminus" in netlist
        assert "vout" in netlist


class TestNonInvertingAmplifier:
    """Tests for non-inverting amplifier template."""

    def test_creates_circuit(self):
        """Should create valid circuit."""
        circuit = non_inverting_amplifier(gain=11)
        assert len(circuit._components) == 2

    def test_gain_11_resistor_values(self):
        """Gain=11 with Rf=10kΩ: R1 = Rf/(gain-1) = 10k/10 = 1kΩ."""
        circuit = non_inverting_amplifier(gain=11, feedback_resistance=10_000)

        resistors = {c.ref: c for c in circuit._components}
        assert abs(resistors["1"].resistance - 1000) < 1
        assert abs(resistors["f"].resistance - 10_000) < 1

    def test_invalid_gain_raises(self):
        """Gain < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            non_inverting_amplifier(gain=0.5)

    def test_gain_2_creates_equal_resistors(self):
        """Gain=2: R1 = Rf/(2-1) = Rf."""
        circuit = non_inverting_amplifier(gain=2, feedback_resistance=10_000)

        resistors = {c.ref: c for c in circuit._components}
        assert abs(resistors["1"].resistance - 10_000) < 1
        assert abs(resistors["f"].resistance - 10_000) < 1


class TestFilterResultDataclass:
    """Tests for FilterResult dataclass."""

    def test_repr(self):
        """Should have readable repr."""
        result = rlc_bandpass(fc=1000, bandwidth=100)
        repr_str = repr(result)

        assert "FilterResult" in repr_str
        assert "1000" in repr_str

    def test_attributes(self):
        """Should have expected attributes."""
        result = sallen_key_lowpass(fc=1000)

        assert hasattr(result, "circuit")
        assert hasattr(result, "components")
        assert hasattr(result, "cutoff_frequency")
        assert hasattr(result, "q_factor")


# =============================================================================
# Complete Amplifier Templates (with OpAmp)
# =============================================================================


class TestVoltageFollower:
    """Tests for voltage follower (buffer) template."""

    def test_returns_amplifier_result(self):
        """Should return AmplifierResult."""
        result = voltage_follower()
        assert isinstance(result, AmplifierResult)

    def test_unity_gain(self):
        """Gain should be exactly 1.0."""
        result = voltage_follower()
        assert result.gain == 1.0

    def test_has_opamp(self):
        """Should include OpAmpIdeal component."""
        result = voltage_follower()
        assert "opamp" in result.components

    def test_has_power_supplies(self):
        """Should include Vcc and Vee power supplies."""
        result = voltage_follower()
        assert "Vcc" in result.components
        assert "Vee" in result.components

    def test_circuit_has_three_components(self):
        """Should have opamp + 2 power supplies."""
        result = voltage_follower()
        assert len(result.circuit._components) == 3

    def test_input_output_nets(self):
        """Should provide input and output nets."""
        result = voltage_follower()
        assert result.input_net.name == "vin"
        assert result.output_net.name == "vout"

    def test_generates_valid_netlist(self):
        """Should generate valid SPICE netlist."""
        result = voltage_follower()
        netlist = result.circuit.build_netlist()

        assert "Voltage_Follower" in netlist
        assert "OA1" in netlist or "E1" in netlist  # OpAmp reference


class TestDifferentialAmplifier:
    """Tests for differential amplifier template."""

    def test_returns_amplifier_result(self):
        """Should return AmplifierResult."""
        result = differential_amplifier(gain=10)
        assert isinstance(result, AmplifierResult)

    def test_gain_stored(self):
        """Should store specified gain."""
        result = differential_amplifier(gain=5)
        assert result.gain == 5

    def test_has_four_resistors(self):
        """Should have R1, R2, Rf, Rg."""
        result = differential_amplifier(gain=10)
        assert "R1" in result.components
        assert "R2" in result.components
        assert "Rf" in result.components
        assert "Rg" in result.components

    def test_resistor_ratios_for_gain(self):
        """Rf/R1 and Rg/R2 should equal gain."""
        gain = 10
        result = differential_amplifier(gain=gain, input_impedance=10_000)

        R1 = result.components["R1"].resistance
        Rf = result.components["Rf"].resistance

        assert abs(Rf / R1 - gain) < 0.01

    def test_has_opamp_and_supplies(self):
        """Should include opamp and power supplies."""
        result = differential_amplifier()
        assert "opamp" in result.components
        assert "Vcc" in result.components
        assert "Vee" in result.components

    def test_circuit_component_count(self):
        """Should have 7 components: 4 resistors + opamp + 2 supplies."""
        result = differential_amplifier()
        assert len(result.circuit._components) == 7


class TestSummingAmplifier:
    """Tests for summing amplifier template."""

    def test_returns_amplifier_result(self):
        """Should return AmplifierResult."""
        result = summing_amplifier(num_inputs=2)
        assert isinstance(result, AmplifierResult)

    def test_inverting_gain(self):
        """Gain should be negative (inverting)."""
        result = summing_amplifier(gain=2)
        assert result.gain == -2

    def test_two_input_components(self):
        """2-input summer should have Rin1, Rin2, Rf."""
        result = summing_amplifier(num_inputs=2)
        assert "Rin1" in result.components
        assert "Rin2" in result.components
        assert "Rf" in result.components

    def test_three_input_components(self):
        """3-input summer should have Rin1, Rin2, Rin3."""
        result = summing_amplifier(num_inputs=3)
        assert "Rin1" in result.components
        assert "Rin2" in result.components
        assert "Rin3" in result.components

    def test_input_resistors_equal(self):
        """All input resistors should be equal."""
        result = summing_amplifier(num_inputs=3, input_impedance=10_000)

        R1 = result.components["Rin1"].resistance
        R2 = result.components["Rin2"].resistance
        R3 = result.components["Rin3"].resistance

        assert R1 == R2 == R3 == 10_000

    def test_feedback_resistor_gain(self):
        """Rf should equal gain * Rin for unity weighting."""
        gain = 5
        Rin = 10_000
        result = summing_amplifier(num_inputs=2, gain=gain, input_impedance=Rin)

        Rf = result.components["Rf"].resistance
        assert abs(Rf - gain * Rin) < 1

    def test_invalid_num_inputs_raises(self):
        """num_inputs < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            summing_amplifier(num_inputs=0)

    def test_component_count(self):
        """Should have correct number of components."""
        # 2 inputs: 2 Rin + Rf + opamp + 2 supplies = 6
        result = summing_amplifier(num_inputs=2)
        assert len(result.circuit._components) == 6

        # 4 inputs: 4 Rin + Rf + opamp + 2 supplies = 8
        result = summing_amplifier(num_inputs=4)
        assert len(result.circuit._components) == 8


class TestAmplifierResultDataclass:
    """Tests for AmplifierResult dataclass."""

    def test_repr(self):
        """Should have readable repr."""
        result = voltage_follower()
        repr_str = repr(result)

        assert "AmplifierResult" in repr_str
        assert "1.00" in repr_str  # gain

    def test_attributes(self):
        """Should have expected attributes."""
        result = differential_amplifier()

        assert hasattr(result, "circuit")
        assert hasattr(result, "components")
        assert hasattr(result, "gain")
        assert hasattr(result, "input_net")
        assert hasattr(result, "output_net")


# =============================================================================
# Advanced Filter Pattern Tests
# =============================================================================


class TestButterworthLowpass:
    """Tests for Butterworth lowpass filter template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import butterworth_lowpass

        assert callable(butterworth_lowpass)

    def test_2nd_order(self):
        """2nd-order Butterworth should create single stage."""
        from spicelab.templates import butterworth_lowpass

        result = butterworth_lowpass(fc=1000, order=2)

        assert result.cutoff_frequency == 1000
        assert result.q_factor is not None
        assert abs(result.q_factor - 0.7071) < 0.01  # Butterworth Q

    def test_4th_order(self):
        """4th-order should create two cascaded stages."""
        from spicelab.templates import butterworth_lowpass

        result = butterworth_lowpass(fc=1000, order=4)

        # 4th order = 2 stages, each with 2R + 2C = 8 components
        assert len(result.components) == 8

    def test_odd_order(self):
        """Odd orders should include first-order RC section."""
        from spicelab.templates import butterworth_lowpass

        result = butterworth_lowpass(fc=1000, order=3)

        # 3rd order = 1st order RC + 2nd order SK = 2 + 4 = 6 components
        assert len(result.components) == 6

    def test_1st_order(self):
        """1st-order should be simple RC."""
        from spicelab.templates import butterworth_lowpass

        result = butterworth_lowpass(fc=1000, order=1)

        # Only R and C
        assert len(result.components) == 2
        assert result.q_factor is None  # No complex poles

    def test_invalid_order_raises(self):
        """Orders outside 1-6 should raise ValueError."""
        from spicelab.templates import butterworth_lowpass

        with pytest.raises(ValueError):
            butterworth_lowpass(fc=1000, order=0)

        with pytest.raises(ValueError):
            butterworth_lowpass(fc=1000, order=7)

    def test_netlist_generation(self):
        """Should generate valid netlist."""
        from spicelab.templates import butterworth_lowpass

        result = butterworth_lowpass(fc=1000, order=4)
        netlist = result.circuit.build_netlist()

        assert "Butterworth_LPF" in netlist
        assert ".end" in netlist.lower()


class TestChebyshevLowpass:
    """Tests for Chebyshev lowpass filter template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import chebyshev_lowpass

        assert callable(chebyshev_lowpass)

    def test_2nd_order(self):
        """2nd-order Chebyshev should have higher Q than Butterworth."""
        from spicelab.templates import chebyshev_lowpass

        result = chebyshev_lowpass(fc=1000, order=2)

        assert result.cutoff_frequency == 1000
        assert result.q_factor is not None
        assert result.q_factor > 0.7071  # Higher Q than Butterworth

    def test_4th_order(self):
        """4th-order should create cascaded stages."""
        from spicelab.templates import chebyshev_lowpass

        result = chebyshev_lowpass(fc=1000, order=4)

        # Should have components for 2 stages
        assert len(result.components) == 8

    def test_ripple_parameter(self):
        """Should accept ripple_db parameter."""
        from spicelab.templates import chebyshev_lowpass

        # Should not raise
        result = chebyshev_lowpass(fc=1000, order=2, ripple_db=1.0)
        assert result.cutoff_frequency == 1000


class TestBesselLowpass:
    """Tests for Bessel lowpass filter template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import bessel_lowpass

        assert callable(bessel_lowpass)

    def test_2nd_order(self):
        """2nd-order Bessel should have lower Q than Butterworth."""
        from spicelab.templates import bessel_lowpass

        result = bessel_lowpass(fc=1000, order=2)

        assert result.cutoff_frequency == 1000
        assert result.q_factor is not None
        assert result.q_factor < 0.7071  # Lower Q than Butterworth

    def test_4th_order_components(self):
        """4th-order should have correct component count."""
        from spicelab.templates import bessel_lowpass

        result = bessel_lowpass(fc=1000, order=4)
        assert len(result.components) == 8


class TestCurrentMirror:
    """Tests for current mirror bias template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import current_mirror

        assert callable(current_mirror)

    def test_unity_ratio(self):
        """Unity ratio should give Iout = Iref."""
        from spicelab.templates import current_mirror

        result = current_mirror(reference_current=1e-3, mirror_ratio=1.0)

        assert abs(result.output_current - 1e-3) < 1e-9
        assert result.mirror_ratio == 1.0

    def test_ratio_2x(self):
        """2x ratio should give Iout = 2 * Iref."""
        from spicelab.templates import current_mirror

        result = current_mirror(reference_current=1e-3, mirror_ratio=2.0)

        assert abs(result.output_current - 2e-3) < 1e-9
        assert result.mirror_ratio == 2.0

    def test_components(self):
        """Should have Vcc, Iref, Iout components."""
        from spicelab.templates import current_mirror

        result = current_mirror(reference_current=1e-3)

        assert "Vcc" in result.components
        assert "Iref" in result.components
        assert "Iout" in result.components

    def test_bias_result_repr(self):
        """BiasResult should have readable repr."""
        from spicelab.templates import current_mirror

        result = current_mirror(reference_current=1e-3)
        repr_str = repr(result)

        assert "BiasResult" in repr_str
        assert "0.001" in repr_str or "1e-03" in repr_str.lower()


class TestBiasResultDataclass:
    """Tests for BiasResult dataclass."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import BiasResult

        assert BiasResult is not None

    def test_attributes(self):
        """Should have expected attributes."""
        from spicelab.templates import current_mirror

        result = current_mirror(reference_current=1e-3)

        assert hasattr(result, "circuit")
        assert hasattr(result, "components")
        assert hasattr(result, "output_current")
        assert hasattr(result, "mirror_ratio")


# =============================================================================
# Frequency Compensation Pattern Tests
# =============================================================================


class TestDominantPoleCompensation:
    """Tests for dominant pole compensation template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import dominant_pole_compensation

        assert callable(dominant_pole_compensation)

    def test_creates_rc_network(self):
        """Should create RC network with correct pole frequency."""
        from spicelab.templates import dominant_pole_compensation

        result = dominant_pole_compensation(pole_frequency=100, impedance=10_000)

        assert result.pole_frequency == 100
        assert "R_comp" in result.components
        assert "C_comp" in result.components

    def test_pole_frequency_calculation(self):
        """RC values should give correct pole frequency."""
        from spicelab.templates import dominant_pole_compensation

        fp = 1000
        result = dominant_pole_compensation(pole_frequency=fp, impedance=10_000)

        R = result.components["R_comp"].resistance
        C = result.components["C_comp"].capacitance

        # fp = 1/(2π RC)
        calculated_fp = 1 / (2 * math.pi * R * C)
        assert abs(calculated_fp - fp) < 1

    def test_netlist_generation(self):
        """Should generate valid netlist."""
        from spicelab.templates import dominant_pole_compensation

        result = dominant_pole_compensation(pole_frequency=100)
        netlist = result.circuit.build_netlist()

        assert "Dominant_Pole" in netlist
        assert "Rcomp" in netlist
        assert "Ccomp" in netlist


class TestLeadCompensation:
    """Tests for lead compensation template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import lead_compensation

        assert callable(lead_compensation)

    def test_creates_lead_network(self):
        """Should create lead network with zero and pole."""
        from spicelab.templates import lead_compensation

        result = lead_compensation(zero_frequency=1000, pole_frequency=10000)

        assert result.zero_frequency == 1000
        assert result.pole_frequency == 10000
        assert "R1" in result.components
        assert "R2" in result.components
        assert "C" in result.components

    def test_phase_boost_calculation(self):
        """Should calculate phase boost correctly."""
        from spicelab.templates import lead_compensation

        # 10:1 ratio gives about 55° phase boost
        result = lead_compensation(zero_frequency=1000, pole_frequency=10000)

        assert result.phase_margin_boost is not None
        assert 50 < result.phase_margin_boost < 60  # Around 55°

    def test_invalid_frequencies_raises(self):
        """Zero >= pole should raise ValueError."""
        from spicelab.templates import lead_compensation

        with pytest.raises(ValueError):
            lead_compensation(zero_frequency=10000, pole_frequency=1000)

        with pytest.raises(ValueError):
            lead_compensation(zero_frequency=1000, pole_frequency=1000)

    def test_component_values(self):
        """Component values should produce correct zero/pole."""
        from spicelab.templates import lead_compensation

        fz = 1000
        fp = 10000
        R1_val = 10_000

        result = lead_compensation(zero_frequency=fz, pole_frequency=fp, impedance=R1_val)

        R1 = result.components["R1"].resistance
        C = result.components["C"].capacitance

        # Verify zero: wz = 1/(R1*C)
        calculated_wz = 1 / (R1 * C)
        expected_wz = 2 * math.pi * fz
        assert abs(calculated_wz - expected_wz) / expected_wz < 0.01


class TestLeadLagCompensation:
    """Tests for lead-lag compensation template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import lead_lag_compensation

        assert callable(lead_lag_compensation)

    def test_creates_lead_lag_network(self):
        """Should create combined lead-lag network."""
        from spicelab.templates import lead_lag_compensation

        result = lead_lag_compensation(lead_zero_freq=1000, lead_pole_freq=10000, lag_pole_freq=10)

        # Should have 5 components: 2R + 1C for lead, 1R + 1C for lag
        assert len(result.components) == 5
        assert "R1_lead" in result.components
        assert "R2_lead" in result.components
        assert "C_lead" in result.components
        assert "R_lag" in result.components
        assert "C_lag" in result.components

    def test_frequencies_stored(self):
        """Should store lead zero and pole frequencies."""
        from spicelab.templates import lead_lag_compensation

        result = lead_lag_compensation(lead_zero_freq=1000, lead_pole_freq=10000, lag_pole_freq=10)

        assert result.zero_frequency == 1000
        assert result.pole_frequency == 10000

    def test_invalid_lead_frequencies_raises(self):
        """Lead zero >= lead pole should raise ValueError."""
        from spicelab.templates import lead_lag_compensation

        with pytest.raises(ValueError):
            lead_lag_compensation(lead_zero_freq=10000, lead_pole_freq=1000, lag_pole_freq=10)

    def test_has_phase_boost(self):
        """Should calculate phase boost from lead portion."""
        from spicelab.templates import lead_lag_compensation

        result = lead_lag_compensation(lead_zero_freq=1000, lead_pole_freq=10000, lag_pole_freq=10)

        assert result.phase_margin_boost is not None
        assert result.phase_margin_boost > 0


class TestMillerCompensation:
    """Tests for Miller compensation template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import miller_compensation

        assert callable(miller_compensation)

    def test_creates_miller_network(self):
        """Should create Miller compensation capacitor."""
        from spicelab.templates import miller_compensation

        result = miller_compensation(pole_frequency=10, gain=100)

        assert result.pole_frequency == 10
        assert "C_miller" in result.components
        assert "R_out" in result.components

    def test_capacitor_value_scaling(self):
        """Higher gain should result in smaller capacitor."""
        from spicelab.templates import miller_compensation

        result_low_gain = miller_compensation(pole_frequency=10, gain=10)
        result_high_gain = miller_compensation(pole_frequency=10, gain=100)

        C_low = result_low_gain.components["C_miller"].capacitance
        C_high = result_high_gain.components["C_miller"].capacitance

        # Higher gain = smaller capacitor (Miller multiplication)
        assert C_high < C_low
        assert abs(C_low / C_high - 10) < 0.1  # 10x gain ratio = 10x C ratio

    def test_netlist_generation(self):
        """Should generate valid netlist."""
        from spicelab.templates import miller_compensation

        result = miller_compensation(pole_frequency=10)
        netlist = result.circuit.build_netlist()

        assert "Miller_Comp" in netlist
        assert "Cmiller" in netlist


class TestCompensationResultDataclass:
    """Tests for CompensationResult dataclass."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import CompensationResult

        assert CompensationResult is not None

    def test_repr_with_pole_only(self):
        """Repr should show pole frequency."""
        from spicelab.templates import dominant_pole_compensation

        result = dominant_pole_compensation(pole_frequency=100)
        repr_str = repr(result)

        assert "CompensationResult" in repr_str
        assert "fp=" in repr_str
        # 100 is displayed as 1e+02 by :.2g format
        assert "100" in repr_str or "1e+02" in repr_str

    def test_repr_with_pole_and_zero(self):
        """Repr should show both pole and zero when present."""
        from spicelab.templates import lead_compensation

        result = lead_compensation(zero_frequency=1000, pole_frequency=10000)
        repr_str = repr(result)

        assert "fp=" in repr_str
        assert "fz=" in repr_str

    def test_attributes(self):
        """Should have expected attributes."""
        from spicelab.templates import lead_compensation

        result = lead_compensation(zero_frequency=1000, pole_frequency=10000)

        assert hasattr(result, "circuit")
        assert hasattr(result, "components")
        assert hasattr(result, "pole_frequency")
        assert hasattr(result, "zero_frequency")
        assert hasattr(result, "phase_margin_boost")


# =============================================================================
# ADC/DAC Building Block Tests
# =============================================================================


class TestR2RDacLadder:
    """Tests for R-2R DAC ladder template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import r2r_dac_ladder

        assert callable(r2r_dac_ladder)

    def test_4bit_dac(self):
        """4-bit DAC should create correct structure."""
        from spicelab.templates import r2r_dac_ladder

        result = r2r_dac_ladder(bits=4)

        assert result.resolution_bits == 4
        assert len(result.input_nets) == 4
        # Should have bit input names b0-b3
        assert result.input_nets[0].name == "b0"
        assert result.input_nets[3].name == "b3"

    def test_resistor_values(self):
        """Should use R and 2R resistor values."""
        from spicelab.templates import r2r_dac_ladder

        R = 10_000
        result = r2r_dac_ladder(bits=4, r_value=R)

        # Check that we have both R and 2R resistors
        resistors = [c for c in result.circuit._components if "Resistor" in type(c).__name__]
        values = {r.resistance for r in resistors}

        assert R in values or any(abs(v - R) < 1 for v in values)
        assert 2 * R in values or any(abs(v - 2 * R) < 1 for v in values)

    def test_component_count(self):
        """Should have correct number of components."""
        from spicelab.templates import r2r_dac_ladder

        # 4-bit: 4 x 2R (bit inputs) + 3 x R (series) + 1 x 2R (term) = 8
        result = r2r_dac_ladder(bits=4)
        assert len(result.circuit._components) == 8

        # 8-bit: 8 x 2R + 7 x R + 1 x 2R = 16
        result = r2r_dac_ladder(bits=8)
        assert len(result.circuit._components) == 16

    def test_invalid_bits_raises(self):
        """Bits outside 1-8 should raise ValueError."""
        from spicelab.templates import r2r_dac_ladder

        with pytest.raises(ValueError):
            r2r_dac_ladder(bits=0)

        with pytest.raises(ValueError):
            r2r_dac_ladder(bits=9)

    def test_netlist_generation(self):
        """Should generate valid netlist."""
        from spicelab.templates import r2r_dac_ladder

        result = r2r_dac_ladder(bits=4)
        netlist = result.circuit.build_netlist()

        assert "R2R_DAC" in netlist
        assert "vout" in netlist


class TestSampleAndHold:
    """Tests for sample and hold template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import sample_and_hold

        assert callable(sample_and_hold)

    def test_creates_sh_circuit(self):
        """Should create S/H with switch and capacitor."""
        from spicelab.templates import sample_and_hold

        result = sample_and_hold()

        assert "R_switch" in result.components
        assert "C_hold" in result.components

    def test_capacitor_value(self):
        """Should use specified hold capacitance."""
        from spicelab.templates import sample_and_hold

        C = 100e-12
        result = sample_and_hold(hold_capacitance=C)

        C_hold = result.components["C_hold"].capacitance
        assert abs(C_hold - C) < 1e-15

    def test_resolution_bits_zero(self):
        """Analog S/H should have 0 resolution bits."""
        from spicelab.templates import sample_and_hold

        result = sample_and_hold()
        assert result.resolution_bits == 0

    def test_input_output_nets(self):
        """Should provide input and output nets."""
        from spicelab.templates import sample_and_hold

        result = sample_and_hold()

        assert len(result.input_nets) == 1
        assert result.input_nets[0].name == "vin"
        assert result.output_net.name == "vhold"

    def test_netlist_generation(self):
        """Should generate valid netlist."""
        from spicelab.templates import sample_and_hold

        result = sample_and_hold()
        netlist = result.circuit.build_netlist()

        assert "Sample_Hold" in netlist
        assert "Rsw" in netlist
        assert "Chold" in netlist


class TestComparatorBank:
    """Tests for comparator bank (flash ADC reference ladder) template."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import comparator_bank

        assert callable(comparator_bank)

    def test_3bit_flash(self):
        """3-bit flash should create 7 reference voltages."""
        from spicelab.templates import comparator_bank

        result = comparator_bank(bits=3)

        assert result.resolution_bits == 3
        # 2^3 - 1 = 7 comparators/reference levels
        assert len(result.input_nets) == 7

    def test_2bit_flash(self):
        """2-bit flash should create 3 reference voltages."""
        from spicelab.templates import comparator_bank

        result = comparator_bank(bits=2)

        assert result.resolution_bits == 2
        assert len(result.input_nets) == 3

    def test_reference_net_names(self):
        """Reference nets should be named vref1, vref2, etc."""
        from spicelab.templates import comparator_bank

        result = comparator_bank(bits=2)

        names = [n.name for n in result.input_nets]
        assert "vref1" in names
        assert "vref2" in names
        assert "vref3" in names

    def test_has_voltage_sources(self):
        """Should have high reference voltage source."""
        from spicelab.templates import comparator_bank

        result = comparator_bank(bits=2)
        assert "V_high" in result.components

        # V_low only present when vref_range has non-zero low
        result_with_offset = comparator_bank(bits=2, vref_range=(1.0, 5.0))
        assert "V_low" in result_with_offset.components

    def test_resistor_ladder(self):
        """Should create resistor divider ladder."""
        from spicelab.templates import comparator_bank

        result = comparator_bank(bits=2)

        # 2-bit = 3 comparators = 4 resistors
        resistors = [c for c in result.circuit._components if "Resistor" in type(c).__name__]
        assert len(resistors) == 4

    def test_invalid_bits_raises(self):
        """Bits outside 1-4 should raise ValueError."""
        from spicelab.templates import comparator_bank

        with pytest.raises(ValueError):
            comparator_bank(bits=0)

        with pytest.raises(ValueError):
            comparator_bank(bits=5)

    def test_netlist_generation(self):
        """Should generate valid netlist."""
        from spicelab.templates import comparator_bank

        result = comparator_bank(bits=3)
        netlist = result.circuit.build_netlist()

        assert "Comparator_Bank" in netlist


class TestConverterResultDataclass:
    """Tests for ConverterResult dataclass."""

    def test_import(self):
        """Should be importable."""
        from spicelab.templates import ConverterResult

        assert ConverterResult is not None

    def test_repr(self):
        """Should have readable repr."""
        from spicelab.templates import r2r_dac_ladder

        result = r2r_dac_ladder(bits=8)
        repr_str = repr(result)

        assert "ConverterResult" in repr_str
        assert "bits=8" in repr_str

    def test_attributes(self):
        """Should have expected attributes."""
        from spicelab.templates import r2r_dac_ladder

        result = r2r_dac_ladder(bits=4)

        assert hasattr(result, "circuit")
        assert hasattr(result, "components")
        assert hasattr(result, "resolution_bits")
        assert hasattr(result, "input_nets")
        assert hasattr(result, "output_net")
