"""Physical units and SI formatting for circuit parameters.

Phase 2 (P2.2): Unit enum with SPICE suffixes and conversion utilities.

Example:
    >>> from spicelab.core.units import Unit, format_si_value
    >>>
    >>> Unit.OHM.suffix
    'Ohm'
    >>> format_si_value(10_000, Unit.OHM)
    '10k'
    >>> format_si_value(1e-6, Unit.FARAD)
    '1u'
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Unit", "format_si_value", "parse_si_value"]


class Unit(Enum):
    """Physical units for circuit parameters with SPICE suffixes.

    Each unit has:
    - name: Python identifier (OHM, FARAD, etc.)
    - suffix: SPICE suffix string
    - si_multiplier: Multiplier to convert to SI base unit

    SPICE Engineering Notation:
        T = 10^12 (tera)
        G = 10^9  (giga)
        Meg = 10^6 (mega) - special case to avoid conflict with 'M' (milli)
        k = 10^3  (kilo)
        m = 10^-3 (milli)
        u = 10^-6 (micro) - 'u' used instead of 'μ'
        n = 10^-9 (nano)
        p = 10^-12 (pico)
        f = 10^-15 (femto)
    """

    # Electrical units
    OHM = ("Ohm", 1.0)  # Resistance (Ω)
    SIEMENS = ("S", 1.0)  # Conductance (S = 1/Ω)
    FARAD = ("F", 1.0)  # Capacitance
    HENRY = ("H", 1.0)  # Inductance
    VOLT = ("V", 1.0)  # Voltage
    AMPERE = ("A", 1.0)  # Current
    WATT = ("W", 1.0)  # Power
    JOULE = ("J", 1.0)  # Energy

    # Time/Frequency
    SECOND = ("s", 1.0)  # Time
    HERTZ = ("Hz", 1.0)  # Frequency

    # Temperature
    CELSIUS = ("°C", 1.0)  # Temperature (°C)
    KELVIN = ("K", 1.0)  # Temperature (K)

    # Dimensionless
    UNITLESS = ("", 1.0)  # Gain, ratio, etc.
    PERCENT = ("%", 0.01)  # Percentage (converts to 0-1)
    DECIBEL = ("dB", 1.0)  # Logarithmic ratio

    def __init__(self, suffix: str, si_multiplier: float) -> None:
        self.suffix = suffix
        self.si_multiplier = si_multiplier

    def __str__(self) -> str:
        return self.suffix

    def __repr__(self) -> str:
        return f"Unit.{self.name}"


# Engineering notation multipliers (SPICE-compatible)
_SI_PREFIXES = {
    "T": 1e12,
    "G": 1e9,
    "Meg": 1e6,  # Special: SPICE uses "Meg" to avoid conflict with "m" (milli)
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,  # SPICE uses 'u' instead of 'μ'
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

# Reverse mapping for formatting (prefer these when formatting)
_SI_PREFIXES_FORMAT = [
    ("T", 1e12),
    ("G", 1e9),
    ("Meg", 1e6),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
]


def format_si_value(value: float, unit: Unit, precision: int = 3) -> str:
    """Format value with SI prefix and unit suffix.

    Args:
        value: Numeric value in SI base units
        unit: Physical unit
        precision: Significant figures (default 3)

    Returns:
        SPICE-compatible string (e.g., "10k", "1.5Meg", "100n")

    Example:
        >>> format_si_value(10_000, Unit.OHM)
        '10k'
        >>> format_si_value(1.5e6, Unit.OHM)
        '1.5Meg'
        >>> format_si_value(100e-9, Unit.FARAD)
        '100n'
    """
    if value == 0:
        return f"0{unit.suffix}"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    # Find best SI prefix (largest multiplier where 1 <= scaled < 1000)
    for prefix, multiplier in _SI_PREFIXES_FORMAT:
        scaled = abs_value / multiplier
        if 1 <= scaled < 1000:
            # Format with appropriate precision
            if scaled >= 100:
                formatted = f"{scaled:.{max(0, precision-3)}f}"
            elif scaled >= 10:
                formatted = f"{scaled:.{max(0, precision-2)}f}"
            else:
                formatted = f"{scaled:.{max(0, precision-1)}f}"

            # Remove trailing zeros and decimal point if not needed
            formatted = formatted.rstrip("0").rstrip(".")

            return f"{sign}{formatted}{prefix}{unit.suffix}"

    # No prefix works - use plain formatting
    if abs_value >= 1e-15:
        formatted = f"{abs_value:.{precision}g}"
    else:
        formatted = f"{abs_value:.{precision}e}"

    return f"{sign}{formatted}{unit.suffix}"


def parse_si_value(text: str, unit: Unit | None = None) -> float:
    """Parse SPICE value string to float (SI units).

    Args:
        text: SPICE value string (e.g., "10k", "1.5Meg", "100nF")
        unit: Optional unit for validation (will strip unit suffix if present)

    Returns:
        Float value in SI base units

    Raises:
        ValueError: If string cannot be parsed

    Example:
        >>> parse_si_value("10k")
        10000.0
        >>> parse_si_value("1.5Meg")
        1500000.0
        >>> parse_si_value("100nF", Unit.FARAD)
        1e-07
    """
    text = text.strip()

    if not text:
        raise ValueError("Empty value string")

    # Strip unit suffix if present
    if unit and text.endswith(unit.suffix):
        text = text[: -len(unit.suffix)].rstrip()

    # Try to find SI prefix at end
    for prefix, multiplier in _SI_PREFIXES.items():
        if text.endswith(prefix):
            numeric_part = text[: -len(prefix)]
            try:
                value = float(numeric_part)
                return value * multiplier
            except ValueError:
                pass

    # No prefix found, parse as plain float
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse '{text}' as numeric value") from exc


def normalize_spice_suffix(text: str) -> str:
    """Normalize SPICE suffix to canonical form.

    Handles common variations:
    - 'meg' or 'MEG' → 'Meg'
    - 'K' or 'K' → 'k'
    - 'μ' or 'μ' → 'u'

    Args:
        text: Raw SPICE value string

    Returns:
        Normalized string
    """
    # Common normalizations
    text = text.replace("μ", "u")  # Greek mu → u
    text = text.replace("µ", "u")  # Micro sign → u

    # Case-sensitive prefix handling
    for wrong, right in [("meg", "Meg"), ("MEG", "Meg"), ("K", "k")]:
        text = text.replace(wrong, right)

    return text


# Backward compatibility with existing utils.units module
def to_float(text: str) -> float:
    """Legacy function for backward compatibility.

    Parses SPICE value without unit validation.
    """
    text = normalize_spice_suffix(text)
    return parse_si_value(text)
