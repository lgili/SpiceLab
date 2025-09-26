from __future__ import annotations

import math

import pytest
from spicelab.extensions import UniformADC, adc_step


def test_uniform_adc_samples_and_reconstructs() -> None:
    adc = UniformADC(bits=8, vref=2.5)
    code, voltage = adc.sample(1.0)
    assert 0 <= code < 256
    assert math.isclose(voltage, adc.to_voltage(code), rel_tol=1e-9)


def test_uniform_adc_bipolar_quantisation() -> None:
    adc = UniformADC(bits=4, vref=1.2, bipolar=True)
    code_neg, value_neg = adc.sample(-0.6)
    code_pos, value_pos = adc.sample(0.9)
    assert code_neg < code_pos
    assert pytest.approx(value_neg, rel=1e-6) == adc.to_voltage(code_neg)
    assert pytest.approx(value_pos, rel=1e-6) == adc.to_voltage(code_pos)


def test_adc_step_returns_code_and_voltage() -> None:
    adc = UniformADC(bits=6, vref=3.3)
    code, voltage = adc_step(2.0, adc)
    assert code == adc.to_code(2.0)
    assert voltage == adc.to_voltage(code)
