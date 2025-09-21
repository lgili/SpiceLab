"""Switch component entries for the component library."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..core.components import ISwitch, VSwitch
from .registry import register_component


@dataclass(frozen=True)
class SwitchEntry:
    slug: str
    model_name: str
    description: str
    model_card: str
    switch_type: str  # "voltage" or "current"

    def metadata(self) -> Mapping[str, object]:
        return {
            "model": self.model_name,
            "model_card": self.model_card,
            "description": self.description,
            "type": self.switch_type,
        }


def _make_voltage_switch(entry: SwitchEntry) -> Callable[..., VSwitch]:
    def factory(ref: str, *, model: str | None = None) -> VSwitch:
        selected_model = model or entry.model_name
        return VSwitch(ref, selected_model)

    return factory


def _make_current_switch(entry: SwitchEntry) -> Callable[..., ISwitch]:
    def factory(ref: str, *, ctrl_vsrc: str = "VCTRL", model: str | None = None) -> ISwitch:
        selected_model = model or entry.model_name
        return ISwitch(ref, ctrl_vsrc=ctrl_vsrc, model=selected_model)

    return factory


_SWITCHES = [
    SwitchEntry(
        slug="vsw_spst_fast",
        model_name="SW_SPST_FAST",
        description="Voltage-controlled SPST switch (RON=0.1Ω, ROFF=1MΩ)",
        model_card=".model SW_SPST_FAST VSWITCH(RON=0.1 ROFF=1e6 VON=2 VOFF=0)",
        switch_type="voltage",
    ),
    SwitchEntry(
        slug="isw_spst_sense",
        model_name="ISW_SPST_SLOW",
        description="Current-controlled SPST switch for sensing",
        model_card=".model ISW_SPST_SLOW ISWITCH(RON=0.2 ROFF=5e6 ION=20mA IOFF=5mA)",
        switch_type="current",
    ),
]


def _register_defaults() -> None:
    for entry in _SWITCHES:
        name = f"switch.{entry.slug}"
        factory = (
            _make_voltage_switch(entry)
            if entry.switch_type == "voltage"
            else _make_current_switch(entry)
        )
        register_component(
            name,
            factory,
            category="switch",
            metadata=entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = ["_SWITCHES"]
