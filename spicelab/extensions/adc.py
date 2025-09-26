from __future__ import annotations

from dataclasses import dataclass

__all__ = ["UniformADC", "adc_step"]


@dataclass(frozen=True)
class UniformADC:
    """Uniform mid-tread ADC helper used by co-simulation callbacks."""

    bits: int
    vref: float
    bipolar: bool = False

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("ADC resolution must be positive")
        if self.vref <= 0:
            raise ValueError("ADC reference voltage must be positive")

    @property
    def _levels(self) -> int:
        return 1 << self.bits

    def _clamp(self, value: float) -> float:
        if self.bipolar:
            if value < -self.vref:
                return -self.vref
            if value > self.vref:
                return self.vref
            return value
        if value < 0.0:
            return 0.0
        if value > self.vref:
            return self.vref
        return value

    def to_code(self, value: float) -> int:
        """Convert an analog value to the nearest ADC code."""

        clamped = self._clamp(value)
        span = self.vref * 2 if self.bipolar else self.vref
        offset = clamped + self.vref if self.bipolar else clamped
        scale = (self._levels - 1) / span
        code = round(offset * scale)
        if code < 0:
            return 0
        if code >= self._levels:
            return self._levels - 1
        return code

    def to_voltage(self, code: int) -> float:
        """Convert an ADC code back to an analog value (ideal DAC)."""

        if code < 0 or code >= self._levels:
            raise ValueError("Code out of range for ADC")
        span = self.vref * 2 if self.bipolar else self.vref
        base = -self.vref if self.bipolar else 0.0
        return base + (code / (self._levels - 1)) * span

    def sample(self, value: float) -> tuple[int, float]:
        """Quantise a value returning ``(code, reconstructed_voltage)``."""

        code = self.to_code(value)
        return code, self.to_voltage(code)


def adc_step(value: float, adc: UniformADC) -> tuple[int, float]:
    """Quantise ``value`` using ``adc`` while returning code and analog value."""

    return adc.sample(value)
