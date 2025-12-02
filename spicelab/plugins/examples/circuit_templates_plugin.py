"""Circuit Templates Plugin.

This plugin provides pre-built circuit templates for common designs,
making it easy to quickly create standard circuits.

Usage::

    from spicelab.plugins import PluginManager
    from spicelab.plugins.examples import CircuitTemplatesPlugin

    manager = PluginManager()
    plugin = manager.loader.load_from_class(CircuitTemplatesPlugin)
    manager.registry.register(plugin)
    manager.activate_plugin("circuit-templates")

    # Get the plugin instance
    templates = plugin

    # Create circuits from templates
    rc_filter = templates.create("rc_lowpass", cutoff_freq=1000)
    voltage_divider = templates.create("voltage_divider", r1="10k", r2="10k", vin=5.0)
    inverting_amp = templates.create("inverting_amplifier", gain=10, rin="10k")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from ..base import Plugin, PluginMetadata, PluginType


@dataclass
class TemplateInfo:
    """Information about a circuit template."""

    name: str
    description: str
    category: str
    parameters: dict[str, str]  # param_name -> description
    create_fn: Callable[..., Any]


class CircuitTemplatesPlugin(Plugin):
    """Plugin providing pre-built circuit templates.

    Features:
    - RC/RL/RLC filters (lowpass, highpass, bandpass)
    - Voltage dividers
    - Op-amp circuits (inverting, non-inverting, buffer)
    - Power supply circuits
    - Oscillators
    - Common test circuits

    All templates are parametric and can be customized.
    """

    def __init__(self) -> None:
        self._templates: dict[str, TemplateInfo] = {}
        self._register_builtin_templates()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="circuit-templates",
            version="1.0.0",
            description="Pre-built circuit templates for common designs",
            author="SpiceLab Team",
            plugin_type=PluginType.GENERIC,
            keywords=["templates", "circuits", "library", "examples"],
        )

    def activate(self) -> None:
        """Activate the plugin."""
        pass

    def deactivate(self) -> None:
        """Deactivate the plugin."""
        pass

    def _register_builtin_templates(self) -> None:
        """Register all built-in templates."""
        # Filters
        self._templates["rc_lowpass"] = TemplateInfo(
            name="rc_lowpass",
            description="RC Low-Pass Filter",
            category="filters",
            parameters={
                "cutoff_freq": "Cutoff frequency in Hz (default: 1000)",
                "r_value": "Resistance value (default: calculated)",
                "c_value": "Capacitance value (default: calculated)",
                "vin": "Input voltage (default: 1.0V)",
            },
            create_fn=self._create_rc_lowpass,
        )

        self._templates["rc_highpass"] = TemplateInfo(
            name="rc_highpass",
            description="RC High-Pass Filter",
            category="filters",
            parameters={
                "cutoff_freq": "Cutoff frequency in Hz (default: 1000)",
                "r_value": "Resistance value (default: calculated)",
                "c_value": "Capacitance value (default: calculated)",
                "vin": "Input voltage (default: 1.0V)",
            },
            create_fn=self._create_rc_highpass,
        )

        self._templates["rlc_bandpass"] = TemplateInfo(
            name="rlc_bandpass",
            description="RLC Band-Pass Filter",
            category="filters",
            parameters={
                "center_freq": "Center frequency in Hz",
                "q_factor": "Quality factor (default: 10)",
                "r_value": "Resistance value (default: calculated)",
            },
            create_fn=self._create_rlc_bandpass,
        )

        # Voltage dividers
        self._templates["voltage_divider"] = TemplateInfo(
            name="voltage_divider",
            description="Resistive Voltage Divider",
            category="basic",
            parameters={
                "r1": "Top resistor value (default: 10k)",
                "r2": "Bottom resistor value (default: 10k)",
                "vin": "Input voltage (default: 5.0V)",
            },
            create_fn=self._create_voltage_divider,
        )

        self._templates["voltage_divider_ratio"] = TemplateInfo(
            name="voltage_divider_ratio",
            description="Voltage Divider with specified ratio",
            category="basic",
            parameters={
                "ratio": "Output/Input voltage ratio (0 to 1)",
                "vin": "Input voltage (default: 5.0V)",
                "total_r": "Total resistance (default: 20k)",
            },
            create_fn=self._create_voltage_divider_ratio,
        )

        # Op-amp circuits
        self._templates["inverting_amplifier"] = TemplateInfo(
            name="inverting_amplifier",
            description="Inverting Op-Amp Amplifier",
            category="amplifiers",
            parameters={
                "gain": "Voltage gain (positive number, will be inverted)",
                "rin": "Input resistance (default: 10k)",
                "vcc": "Positive supply voltage (default: 15V)",
                "vee": "Negative supply voltage (default: -15V)",
            },
            create_fn=self._create_inverting_amp,
        )

        self._templates["non_inverting_amplifier"] = TemplateInfo(
            name="non_inverting_amplifier",
            description="Non-Inverting Op-Amp Amplifier",
            category="amplifiers",
            parameters={
                "gain": "Voltage gain (must be >= 1)",
                "r1": "Feedback resistor to ground (default: 10k)",
                "vcc": "Positive supply voltage (default: 15V)",
                "vee": "Negative supply voltage (default: -15V)",
            },
            create_fn=self._create_non_inverting_amp,
        )

        self._templates["voltage_follower"] = TemplateInfo(
            name="voltage_follower",
            description="Unity-Gain Buffer (Voltage Follower)",
            category="amplifiers",
            parameters={
                "vcc": "Positive supply voltage (default: 15V)",
                "vee": "Negative supply voltage (default: -15V)",
            },
            create_fn=self._create_voltage_follower,
        )

        # Oscillators
        self._templates["rc_oscillator"] = TemplateInfo(
            name="rc_oscillator",
            description="RC Phase-Shift Oscillator",
            category="oscillators",
            parameters={
                "frequency": "Target oscillation frequency in Hz",
                "vcc": "Supply voltage (default: 12V)",
            },
            create_fn=self._create_rc_oscillator,
        )

        # Test circuits
        self._templates["rc_transient_test"] = TemplateInfo(
            name="rc_transient_test",
            description="RC circuit for transient analysis testing",
            category="test",
            parameters={
                "tau": "Time constant in seconds (default: 1ms)",
                "vpulse": "Pulse voltage (default: 5V)",
            },
            create_fn=self._create_rc_transient_test,
        )

    # Template creation methods

    def _create_rc_lowpass(
        self,
        cutoff_freq: float = 1000,
        r_value: str | None = None,
        c_value: str | None = None,
        vin: float = 1.0,
        name: str = "rc_lowpass",
    ) -> Any:
        """Create RC low-pass filter."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Capacitor, Resistor, Vdc
        from spicelab.core.net import GND, Net

        # Calculate component values if not provided
        if r_value is None and c_value is None:
            # Default: use 10k resistor, calculate capacitor
            r_val = 10e3
            c_val = 1 / (2 * math.pi * cutoff_freq * r_val)
            r_value = "10k"
            c_value = f"{c_val:.3g}"
        elif r_value is None:
            c_val = self._parse_value(c_value)  # type: ignore[arg-type]
            r_val = 1 / (2 * math.pi * cutoff_freq * c_val)
            r_value = f"{r_val:.3g}"
        elif c_value is None:
            r_val = self._parse_value(r_value)
            c_val = 1 / (2 * math.pi * cutoff_freq * r_val)
            c_value = f"{c_val:.3g}"

        circuit = Circuit(name)
        v_in = Vdc("Vin", vin)
        r1 = Resistor("R1", r_value)
        c1 = Capacitor("C1", c_value)

        circuit.add(v_in, r1, c1)
        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(r1.ports[0], Net("in"))
        circuit.connect(r1.ports[1], Net("out"))
        circuit.connect(c1.ports[0], Net("out"))
        circuit.connect(c1.ports[1], GND)

        return circuit

    def _create_rc_highpass(
        self,
        cutoff_freq: float = 1000,
        r_value: str | None = None,
        c_value: str | None = None,
        vin: float = 1.0,
        name: str = "rc_highpass",
    ) -> Any:
        """Create RC high-pass filter."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Capacitor, Resistor, Vdc
        from spicelab.core.net import GND, Net

        if r_value is None and c_value is None:
            r_val = 10e3
            c_val = 1 / (2 * math.pi * cutoff_freq * r_val)
            r_value = "10k"
            c_value = f"{c_val:.3g}"
        elif r_value is None:
            c_val = self._parse_value(c_value)  # type: ignore[arg-type]
            r_val = 1 / (2 * math.pi * cutoff_freq * c_val)
            r_value = f"{r_val:.3g}"
        elif c_value is None:
            r_val = self._parse_value(r_value)
            c_val = 1 / (2 * math.pi * cutoff_freq * r_val)
            c_value = f"{c_val:.3g}"

        circuit = Circuit(name)
        v_in = Vdc("Vin", vin)
        c1 = Capacitor("C1", c_value)
        r1 = Resistor("R1", r_value)

        circuit.add(v_in, c1, r1)
        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(c1.ports[0], Net("in"))
        circuit.connect(c1.ports[1], Net("out"))
        circuit.connect(r1.ports[0], Net("out"))
        circuit.connect(r1.ports[1], GND)

        return circuit

    def _create_rlc_bandpass(
        self,
        center_freq: float = 1000,
        q_factor: float = 10,
        r_value: str | None = None,
        name: str = "rlc_bandpass",
    ) -> Any:
        """Create RLC band-pass filter."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Capacitor, Inductor, Resistor, Vac
        from spicelab.core.net import GND, Net

        # Calculate L and C from center_freq and Q
        omega = 2 * math.pi * center_freq

        if r_value is None:
            r_val = 1000  # Default 1k
            r_value = "1k"
        else:
            r_val = self._parse_value(r_value)

        # For series RLC: Q = (1/R) * sqrt(L/C) and omega_0 = 1/sqrt(LC)
        l_val = (q_factor * r_val) / omega
        c_val = 1 / (omega * omega * l_val)

        circuit = Circuit(name)
        v_in = Vac("Vin", ac_mag=1.0)
        r1 = Resistor("R1", r_value)
        l1 = Inductor("L1", f"{l_val:.3g}")
        c1 = Capacitor("C1", f"{c_val:.3g}")

        circuit.add(v_in, r1, l1, c1)
        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(r1.ports[0], Net("in"))
        circuit.connect(r1.ports[1], Net("n1"))
        circuit.connect(l1.ports[0], Net("n1"))
        circuit.connect(l1.ports[1], Net("out"))
        circuit.connect(c1.ports[0], Net("out"))
        circuit.connect(c1.ports[1], GND)

        return circuit

    def _create_voltage_divider(
        self,
        r1: str = "10k",
        r2: str = "10k",
        vin: float = 5.0,
        name: str = "voltage_divider",
    ) -> Any:
        """Create voltage divider."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Resistor, Vdc
        from spicelab.core.net import GND, Net

        circuit = Circuit(name)
        v_in = Vdc("Vin", vin)
        r_top = Resistor("R1", r1)
        r_bot = Resistor("R2", r2)

        circuit.add(v_in, r_top, r_bot)
        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(r_top.ports[0], Net("in"))
        circuit.connect(r_top.ports[1], Net("out"))
        circuit.connect(r_bot.ports[0], Net("out"))
        circuit.connect(r_bot.ports[1], GND)

        return circuit

    def _create_voltage_divider_ratio(
        self,
        ratio: float = 0.5,
        vin: float = 5.0,
        total_r: str = "20k",
        name: str = "voltage_divider_ratio",
    ) -> Any:
        """Create voltage divider with specified ratio."""
        total = self._parse_value(total_r)
        r2_val = total * ratio
        r1_val = total - r2_val

        return self._create_voltage_divider(
            r1=f"{r1_val:.3g}",
            r2=f"{r2_val:.3g}",
            vin=vin,
            name=name,
        )

    def _create_inverting_amp(
        self,
        gain: float = 10,
        rin: str = "10k",
        vcc: float = 15,
        vee: float = -15,
        name: str = "inverting_amp",
    ) -> Any:
        """Create inverting op-amp amplifier."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Resistor, Vdc
        from spicelab.core.net import GND, Net

        # Rf = gain * Rin
        rin_val = self._parse_value(rin)
        rf_val = gain * rin_val

        circuit = Circuit(name)

        # Components
        v_in = Vdc("Vin", 1.0)
        v_cc = Vdc("Vcc", vcc)
        v_ee = Vdc("Vee", abs(vee))
        r_in = Resistor("Rin", rin)
        r_f = Resistor("Rf", f"{rf_val:.3g}")

        circuit.add(v_in, v_cc, v_ee, r_in, r_f)

        # Connections (simplified - actual opamp needs subcircuit)
        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(v_cc.ports[0], Net("vcc"))
        circuit.connect(v_cc.ports[1], GND)
        circuit.connect(v_ee.ports[0], GND)
        circuit.connect(v_ee.ports[1], Net("vee"))
        circuit.connect(r_in.ports[0], Net("in"))
        circuit.connect(r_in.ports[1], Net("inv_in"))
        circuit.connect(r_f.ports[0], Net("inv_in"))
        circuit.connect(r_f.ports[1], Net("out"))

        # Note: Actual opamp would need to be added as subcircuit
        return circuit

    def _create_non_inverting_amp(
        self,
        gain: float = 2,
        r1: str = "10k",
        vcc: float = 15,
        vee: float = -15,
        name: str = "non_inverting_amp",
    ) -> Any:
        """Create non-inverting op-amp amplifier."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Resistor, Vdc
        from spicelab.core.net import GND, Net

        if gain < 1:
            raise ValueError("Non-inverting gain must be >= 1")

        # Gain = 1 + Rf/R1, so Rf = (gain-1) * R1
        r1_val = self._parse_value(r1)
        rf_val = (gain - 1) * r1_val

        circuit = Circuit(name)

        v_in = Vdc("Vin", 1.0)
        v_cc = Vdc("Vcc", vcc)
        v_ee = Vdc("Vee", abs(vee))
        r_1 = Resistor("R1", r1)
        r_f = Resistor("Rf", f"{rf_val:.3g}")

        circuit.add(v_in, v_cc, v_ee, r_1, r_f)

        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(v_cc.ports[0], Net("vcc"))
        circuit.connect(v_cc.ports[1], GND)
        circuit.connect(v_ee.ports[0], GND)
        circuit.connect(v_ee.ports[1], Net("vee"))
        circuit.connect(r_1.ports[0], Net("inv_in"))
        circuit.connect(r_1.ports[1], GND)
        circuit.connect(r_f.ports[0], Net("inv_in"))
        circuit.connect(r_f.ports[1], Net("out"))

        return circuit

    def _create_voltage_follower(
        self,
        vcc: float = 15,
        vee: float = -15,
        name: str = "voltage_follower",
    ) -> Any:
        """Create unity-gain buffer."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Vdc
        from spicelab.core.net import GND, Net

        circuit = Circuit(name)

        v_in = Vdc("Vin", 1.0)
        v_cc = Vdc("Vcc", vcc)
        v_ee = Vdc("Vee", abs(vee))

        circuit.add(v_in, v_cc, v_ee)

        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(v_cc.ports[0], Net("vcc"))
        circuit.connect(v_cc.ports[1], GND)
        circuit.connect(v_ee.ports[0], GND)
        circuit.connect(v_ee.ports[1], Net("vee"))

        return circuit

    def _create_rc_oscillator(
        self,
        frequency: float = 1000,
        vcc: float = 12,
        name: str = "rc_oscillator",
    ) -> Any:
        """Create RC phase-shift oscillator (simplified)."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Capacitor, Resistor, Vdc
        from spicelab.core.net import GND, Net

        # For 3-stage RC: f = 1 / (2*pi*R*C*sqrt(6))
        # Choose C, calculate R
        c_val = 10e-9  # 10nF
        r_val = 1 / (2 * math.pi * frequency * c_val * math.sqrt(6))

        circuit = Circuit(name)

        v_cc = Vdc("Vcc", vcc)
        r1 = Resistor("R1", f"{r_val:.3g}")
        r2 = Resistor("R2", f"{r_val:.3g}")
        r3 = Resistor("R3", f"{r_val:.3g}")
        c1 = Capacitor("C1", f"{c_val:.3g}")
        c2 = Capacitor("C2", f"{c_val:.3g}")
        c3 = Capacitor("C3", f"{c_val:.3g}")

        circuit.add(v_cc, r1, r2, r3, c1, c2, c3)

        circuit.connect(v_cc.ports[0], Net("vcc"))
        circuit.connect(v_cc.ports[1], GND)

        # Phase shift network
        circuit.connect(c1.ports[0], Net("in"))
        circuit.connect(c1.ports[1], Net("n1"))
        circuit.connect(r1.ports[0], Net("n1"))
        circuit.connect(r1.ports[1], GND)
        circuit.connect(c2.ports[0], Net("n1"))
        circuit.connect(c2.ports[1], Net("n2"))
        circuit.connect(r2.ports[0], Net("n2"))
        circuit.connect(r2.ports[1], GND)
        circuit.connect(c3.ports[0], Net("n2"))
        circuit.connect(c3.ports[1], Net("out"))
        circuit.connect(r3.ports[0], Net("out"))
        circuit.connect(r3.ports[1], GND)

        return circuit

    def _create_rc_transient_test(
        self,
        tau: float = 1e-3,
        vpulse: float = 5.0,
        name: str = "rc_transient_test",
    ) -> Any:
        """Create RC circuit for transient testing."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Capacitor, Resistor, Vpulse
        from spicelab.core.net import GND, Net

        # tau = RC, choose R = 1k
        r_val = 1000
        c_val = tau / r_val

        circuit = Circuit(name)

        v_in = Vpulse(
            "Vin",
            v1=0,
            v2=vpulse,
            td=0,
            tr="1n",
            tf="1n",
            pw=f"{tau * 10}",
            per=f"{tau * 20}",
        )
        r1 = Resistor("R1", f"{r_val}")
        c1 = Capacitor("C1", f"{c_val:.3g}")

        circuit.add(v_in, r1, c1)

        circuit.connect(v_in.ports[0], Net("in"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(r1.ports[0], Net("in"))
        circuit.connect(r1.ports[1], Net("out"))
        circuit.connect(c1.ports[0], Net("out"))
        circuit.connect(c1.ports[1], GND)

        return circuit

    def _parse_value(self, value_str: str) -> float:
        """Parse component value string to float."""
        multipliers = {
            "f": 1e-15,
            "p": 1e-12,
            "n": 1e-9,
            "u": 1e-6,
            "m": 1e-3,
            "k": 1e3,
            "meg": 1e6,
            "g": 1e9,
        }

        value_str = value_str.strip().lower()

        try:
            return float(value_str)
        except ValueError:
            pass

        for suffix, mult in multipliers.items():
            if value_str.endswith(suffix):
                try:
                    return float(value_str[: -len(suffix)]) * mult
                except ValueError:
                    pass

        raise ValueError(f"Cannot parse value: {value_str}")

    # Public API

    def create(self, template_name: str, **kwargs: Any) -> Any:
        """Create a circuit from a template.

        Args:
            template_name: Name of the template
            **kwargs: Template parameters

        Returns:
            Circuit instance

        Raises:
            KeyError: If template not found
        """
        if template_name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise KeyError(f"Template '{template_name}' not found. Available: {available}")

        template = self._templates[template_name]
        return template.create_fn(**kwargs)

    def list_templates(self, category: str | None = None) -> list[str]:
        """List available templates.

        Args:
            category: Optional category filter

        Returns:
            List of template names
        """
        if category:
            return [
                name
                for name, info in self._templates.items()
                if info.category == category
            ]
        return list(self._templates.keys())

    def get_template_info(self, template_name: str) -> TemplateInfo:
        """Get information about a template.

        Args:
            template_name: Name of the template

        Returns:
            TemplateInfo for the template
        """
        return self._templates[template_name]

    def list_categories(self) -> list[str]:
        """List available template categories."""
        return list(set(info.category for info in self._templates.values()))

    def register_template(
        self,
        name: str,
        description: str,
        category: str,
        parameters: dict[str, str],
        create_fn: Callable[..., Any],
    ) -> None:
        """Register a custom template.

        Args:
            name: Template name
            description: Template description
            category: Category name
            parameters: Dict of parameter names to descriptions
            create_fn: Function that creates the circuit
        """
        self._templates[name] = TemplateInfo(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            create_fn=create_fn,
        )
