"""Behavioral component models for system-level simulation.

This module provides ideal and behavioral component models useful for:
- System-level simulation before detailed design
- Quick "what-if" analysis
- Teaching and learning circuit concepts
- Control system modeling

Components:
- Ideal diode (near-zero forward voltage)
- Ideal switch (low Ron, high Roff)
- Voltage limiter (clamp to ±Vmax)
- Slew rate limiter
- Ideal transformer (coupled inductors)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..core.components import Diode, VSwitch
from .registry import register_component

# -----------------------------------------------------------------------------
# Ideal Diode
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IdealDiodeEntry:
    """Entry for an ideal diode model."""

    slug: str
    model_name: str
    description: str
    model_card: str

    def metadata(self) -> Mapping[str, object]:
        return {
            "model": self.model_name,
            "description": self.description,
            "model_card": self.model_card,
        }


_IDEAL_DIODES = [
    IdealDiodeEntry(
        slug="ideal",
        model_name="D_IDEAL",
        description="Ideal diode (Vf~1mV, very low Is)",
        model_card=".model D_IDEAL D(Is=1e-15 Rs=0.001 N=0.01 Cjo=0)",
    ),
    IdealDiodeEntry(
        slug="ideal_schottky",
        model_name="D_IDEAL_SCHOTTKY",
        description="Ideal Schottky diode (Vf~0.2V)",
        model_card=".model D_IDEAL_SCHOTTKY D(Is=1e-8 Rs=0.01 N=1.03 Cjo=0)",
    ),
    IdealDiodeEntry(
        slug="ideal_zener_5v1",
        model_name="D_IDEAL_Z5V1",
        description="Ideal Zener 5.1V (sharp breakdown)",
        model_card=".model D_IDEAL_Z5V1 D(Is=1e-15 Rs=0.1 N=1 Bv=5.1 Ibv=1m)",
    ),
]


def _make_ideal_diode_factory(entry: IdealDiodeEntry) -> Callable[..., Diode]:
    def factory(ref: str, *, model: str | None = None) -> Diode:
        return Diode(ref, model or entry.model_name)

    return factory


# -----------------------------------------------------------------------------
# Ideal Switch
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IdealSwitchEntry:
    """Entry for an ideal switch model."""

    slug: str
    model_name: str
    description: str
    model_card: str

    def metadata(self) -> Mapping[str, object]:
        return {
            "model": self.model_name,
            "description": self.description,
            "model_card": self.model_card,
        }


_IDEAL_SWITCHES = [
    IdealSwitchEntry(
        slug="ideal",
        model_name="SW_IDEAL",
        description="Ideal switch (Ron=1mΩ, Roff=1GΩ)",
        model_card=".model SW_IDEAL VSWITCH(RON=0.001 ROFF=1e9 VON=1 VOFF=0)",
    ),
    IdealSwitchEntry(
        slug="ideal_fast",
        model_name="SW_IDEAL_FAST",
        description="Ideal fast switch (Ron=10mΩ, Roff=100MΩ, narrow hysteresis)",
        model_card=".model SW_IDEAL_FAST VSWITCH(RON=0.01 ROFF=1e8 VON=0.6 VOFF=0.4)",
    ),
    IdealSwitchEntry(
        slug="relay",
        model_name="SW_RELAY",
        description="Relay-like switch (Ron=50mΩ, slow threshold)",
        model_card=".model SW_RELAY VSWITCH(RON=0.05 ROFF=1e7 VON=3 VOFF=1)",
    ),
]


def _make_ideal_switch_factory(entry: IdealSwitchEntry) -> Callable[..., VSwitch]:
    def factory(ref: str, *, model: str | None = None) -> VSwitch:
        return VSwitch(ref, model or entry.model_name)

    return factory


# -----------------------------------------------------------------------------
# Voltage Limiter (using B-source behavioral)
# -----------------------------------------------------------------------------

# Note: Voltage limiters typically need behavioral sources (B elements)
# which require more complex SPICE syntax. For now, we provide model cards
# that can be manually added. A full implementation would require a new
# component type for behavioral sources.


@dataclass(frozen=True)
class LimiterEntry:
    """Entry for a limiter model (informational - requires manual setup)."""

    slug: str
    description: str
    spice_example: str

    def metadata(self) -> Mapping[str, object]:
        return {
            "description": self.description,
            "spice_example": self.spice_example,
        }


_LIMITERS = [
    LimiterEntry(
        slug="voltage_clamp_5v",
        description="Voltage clamp to ±5V using back-to-back Zeners",
        spice_example="""* Voltage clamp using back-to-back Zeners
D1 in out DZ5V
D2 out in DZ5V
.model DZ5V D(Bv=5 Ibv=1m)""",
    ),
    LimiterEntry(
        slug="slew_rate",
        description="Slew rate limiter (conceptual - use behavioral source)",
        spice_example="""* Slew rate limiter using behavioral source
* B1 out 0 V=sdt(V(in), 0, 1e6)  ; NGSpice syntax
* Or use RC network for simple slew limiting""",
    ),
]


# -----------------------------------------------------------------------------
# Coupled Inductors (for ideal transformer)
# Note: These require K statements in SPICE, handled by Circuit.add_coupling()
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformerEntry:
    """Entry for transformer/coupled inductor info."""

    slug: str
    description: str
    primary_inductance: float
    turns_ratio: float
    coupling: float

    def metadata(self) -> Mapping[str, object]:
        return {
            "description": self.description,
            "primary_inductance": self.primary_inductance,
            "turns_ratio": self.turns_ratio,
            "coupling": self.coupling,
            "secondary_inductance": self.primary_inductance / (self.turns_ratio**2),
        }


_TRANSFORMERS = [
    TransformerEntry(
        slug="ideal_1to1",
        description="Ideal 1:1 isolation transformer (k=0.999)",
        primary_inductance=100e-3,  # 100mH
        turns_ratio=1.0,
        coupling=0.999,
    ),
    TransformerEntry(
        slug="ideal_10to1",
        description="Ideal 10:1 step-down transformer (k=0.999)",
        primary_inductance=100e-3,
        turns_ratio=10.0,
        coupling=0.999,
    ),
    TransformerEntry(
        slug="flyback_5to1",
        description="Flyback transformer 5:1 (k=0.98, typical leakage)",
        primary_inductance=500e-6,  # 500µH
        turns_ratio=5.0,
        coupling=0.98,
    ),
]


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------


def _register_defaults() -> None:
    """Register all behavioral models."""
    # Ideal diodes
    for entry in _IDEAL_DIODES:
        register_component(
            f"behavioral.diode.{entry.slug}",
            _make_ideal_diode_factory(entry),
            category="behavioral",
            metadata=entry.metadata(),
            overwrite=False,
        )

    # Ideal switches
    for entry in _IDEAL_SWITCHES:
        register_component(
            f"behavioral.switch.{entry.slug}",
            _make_ideal_switch_factory(entry),
            category="behavioral",
            metadata=entry.metadata(),
            overwrite=False,
        )

    # Limiters (informational only - no factory)
    for entry in _LIMITERS:
        register_component(
            f"behavioral.limiter.{entry.slug}",
            lambda ref, _e=entry: None,  # Placeholder - requires manual setup
            category="behavioral",
            metadata=entry.metadata(),
            overwrite=False,
        )

    # Transformers (informational only - requires L + K setup)
    for entry in _TRANSFORMERS:
        register_component(
            f"behavioral.transformer.{entry.slug}",
            lambda ref, _e=entry: None,  # Placeholder - use Circuit.add_coupling()
            category="behavioral",
            metadata=entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = [
    "_IDEAL_DIODES",
    "_IDEAL_SWITCHES",
    "_LIMITERS",
    "_TRANSFORMERS",
    "IdealDiodeEntry",
    "IdealSwitchEntry",
    "LimiterEntry",
    "TransformerEntry",
]
