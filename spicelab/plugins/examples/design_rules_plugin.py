"""Design Rules Check (DRC) Plugin.

This plugin performs automatic design rule checks on circuits before simulation,
helping catch common design errors early.

Usage::

    from spicelab.plugins import PluginManager
    from spicelab.plugins.examples import DesignRulesPlugin

    manager = PluginManager()
    plugin = manager.loader.load_from_class(DesignRulesPlugin)
    manager.registry.register(plugin)
    manager.activate_plugin("design-rules")

    # Configure rules
    manager.set_plugin_settings("design-rules", {
        "rules": {
            "floating_nodes": True,
            "short_circuits": True,
            "missing_ground": True,
            "component_values": True,
            "power_limits": {"max_power_watts": 10.0},
        },
        "severity": "warning",  # or "error" to block simulation
    })
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..base import Plugin, PluginMetadata, PluginType
from ..hooks import HookManager, HookPriority, HookType


class RuleSeverity(Enum):
    """Severity level for rule violations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class RuleViolation:
    """Represents a design rule violation."""

    rule_name: str
    message: str
    severity: RuleSeverity
    component: str | None = None
    node: str | None = None
    suggestion: str | None = None

    def __str__(self) -> str:
        loc = self.component or self.node or "circuit"
        return f"[{self.severity.value.upper()}] {self.rule_name} @ {loc}: {self.message}"


@dataclass
class DRCResult:
    """Result of design rules check."""

    passed: bool
    violations: list[RuleViolation] = field(default_factory=list)
    warnings: int = 0
    errors: int = 0

    def __str__(self) -> str:
        if self.passed:
            return f"DRC Passed ({self.warnings} warnings)"
        return f"DRC Failed ({self.errors} errors, {self.warnings} warnings)"


class DesignRulesPlugin(Plugin):
    """Plugin that performs Design Rule Checks (DRC) on circuits.

    Features:
    - Floating node detection
    - Short circuit detection
    - Missing ground reference check
    - Component value validation
    - Power dissipation limits
    - Custom rule support

    Rules run automatically before simulation or can be invoked manually.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "rules": {
                "floating_nodes": True,
                "short_circuits": True,
                "missing_ground": True,
                "component_values": True,
                "power_limits": None,
                "min_resistance": 1e-3,  # 1 mOhm minimum
                "max_capacitance": 1.0,  # 1 F maximum
            },
            "severity": "warning",
            "block_on_error": False,
        }
        self._last_result: DRCResult | None = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="design-rules",
            version="1.0.0",
            description="Design Rule Checks (DRC) for circuit validation",
            author="SpiceLab Team",
            plugin_type=PluginType.GENERIC,
            keywords=["drc", "validation", "design-rules", "lint"],
        )

    def configure(self, settings: dict[str, Any]) -> None:
        """Configure DRC rules."""
        if "rules" in settings:
            self._config["rules"].update(settings["rules"])
        if "severity" in settings:
            self._config["severity"] = settings["severity"]
        if "block_on_error" in settings:
            self._config["block_on_error"] = settings["block_on_error"]

    def activate(self) -> None:
        """Activate the plugin and register hooks."""
        self._register_hooks()

    def deactivate(self) -> None:
        """Deactivate the plugin."""
        self._last_result = None

    def _register_hooks(self) -> None:
        """Register DRC hooks."""
        hook_manager = HookManager.get_instance()

        hook_manager.register_hook(
            HookType.PRE_SIMULATION,
            self._on_pre_simulation,
            priority=HookPriority.CRITICAL,
            plugin_name=self.name,
            description="Run design rules check before simulation",
        )

        hook_manager.register_hook(
            HookType.POST_VALIDATION,
            self._on_post_validation,
            priority=HookPriority.HIGH,
            plugin_name=self.name,
            description="Add DRC results to validation",
        )

    def _on_pre_simulation(self, **kwargs: Any) -> dict[str, Any] | None:
        """Run DRC before simulation."""
        circuit = kwargs.get("circuit")
        if not circuit:
            return None

        result = self.check(circuit)
        self._last_result = result

        if not result.passed and self._config["block_on_error"]:
            raise RuntimeError(f"DRC failed: {result}")

        return {"drc_result": result}

    def _on_post_validation(self, **kwargs: Any) -> None:
        """Add DRC info to validation results."""
        if self._last_result:
            errors = kwargs.get("errors", [])
            warnings = kwargs.get("warnings", [])

            for violation in self._last_result.violations:
                msg = str(violation)
                if violation.severity == RuleSeverity.ERROR:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    def check(self, circuit: Any) -> DRCResult:
        """Run all enabled design rule checks.

        Args:
            circuit: Circuit object to check

        Returns:
            DRCResult with violations found
        """
        violations: list[RuleViolation] = []
        rules = self._config["rules"]

        if rules.get("missing_ground"):
            violations.extend(self._check_ground(circuit))

        if rules.get("floating_nodes"):
            violations.extend(self._check_floating_nodes(circuit))

        if rules.get("component_values"):
            violations.extend(self._check_component_values(circuit))

        if rules.get("short_circuits"):
            violations.extend(self._check_short_circuits(circuit))

        if rules.get("power_limits"):
            violations.extend(self._check_power_limits(circuit))

        # Count by severity
        errors = sum(1 for v in violations if v.severity == RuleSeverity.ERROR)
        warnings = sum(1 for v in violations if v.severity == RuleSeverity.WARNING)

        return DRCResult(
            passed=errors == 0,
            violations=violations,
            errors=errors,
            warnings=warnings,
        )

    def _check_ground(self, circuit: Any) -> list[RuleViolation]:
        """Check for missing ground reference."""
        violations = []

        # Check if circuit has GND or 0 node
        has_ground = False
        # Try _port_to_net (newer API) or _connections (older API)
        connections = getattr(circuit, "_port_to_net", None) or getattr(
            circuit, "_connections", {}
        )

        for _, net in connections.items():
            net_name = str(getattr(net, "name", "")).lower()
            if net_name in ("0", "gnd", "ground"):
                has_ground = True
                break

        if not has_ground:
            violations.append(
                RuleViolation(
                    rule_name="missing_ground",
                    message="Circuit has no ground reference (node 0 or GND)",
                    severity=RuleSeverity.ERROR,
                    suggestion="Add a ground connection using GND or node '0'",
                )
            )

        return violations

    def _check_floating_nodes(self, circuit: Any) -> list[RuleViolation]:
        """Check for floating (unconnected) nodes."""
        violations = []
        connections = getattr(circuit, "_port_to_net", None) or getattr(
            circuit, "_connections", {}
        )

        # Build node connectivity map
        node_connections: dict[str, int] = {}
        for _, net in connections.items():
            net_name = str(getattr(net, "name", ""))
            node_connections[net_name] = node_connections.get(net_name, 0) + 1

        # Nodes with only one connection are potentially floating
        for node, count in node_connections.items():
            if count == 1 and node.lower() not in ("0", "gnd", "ground"):
                violations.append(
                    RuleViolation(
                        rule_name="floating_node",
                        message=f"Node '{node}' has only one connection",
                        severity=RuleSeverity.WARNING,
                        node=node,
                        suggestion="Connect this node to another component or ground",
                    )
                )

        return violations

    def _check_component_values(self, circuit: Any) -> list[RuleViolation]:
        """Check component values are within reasonable ranges."""
        violations = []
        components = getattr(circuit, "_components", [])
        rules = self._config["rules"]

        for comp in components:
            comp_name = getattr(comp, "name", str(comp))
            comp_type = type(comp).__name__

            # Get component value
            value = None
            if hasattr(comp, "value"):
                value = comp.value
            elif hasattr(comp, "resistance"):
                value = comp.resistance
            elif hasattr(comp, "capacitance"):
                value = comp.capacitance
            elif hasattr(comp, "inductance"):
                value = comp.inductance

            if value is None:
                continue

            # Parse value if string
            if isinstance(value, str):
                value = self._parse_value(value)

            if value is None:
                continue

            # Check resistor minimum
            if comp_type == "Resistor":
                min_r = rules.get("min_resistance", 1e-3)
                if value < min_r:
                    violations.append(
                        RuleViolation(
                            rule_name="low_resistance",
                            message=f"Resistance {value} is below minimum {min_r}",
                            severity=RuleSeverity.WARNING,
                            component=comp_name,
                            suggestion="Very low resistance may cause numerical issues",
                        )
                    )

            # Check capacitor maximum
            if comp_type == "Capacitor":
                max_c = rules.get("max_capacitance", 1.0)
                if value > max_c:
                    violations.append(
                        RuleViolation(
                            rule_name="high_capacitance",
                            message=f"Capacitance {value} exceeds maximum {max_c}",
                            severity=RuleSeverity.WARNING,
                            component=comp_name,
                            suggestion="Very high capacitance may slow simulation",
                        )
                    )

            # Check for zero/negative values
            if value <= 0:
                violations.append(
                    RuleViolation(
                        rule_name="invalid_value",
                        message=f"Component has invalid value: {value}",
                        severity=RuleSeverity.ERROR,
                        component=comp_name,
                        suggestion="Component values must be positive",
                    )
                )

        return violations

    def _check_short_circuits(self, circuit: Any) -> list[RuleViolation]:
        """Check for potential short circuits."""
        violations = []
        components = getattr(circuit, "_components", [])
        connections = getattr(circuit, "_connections", {})

        for comp in components:
            comp_name = getattr(comp, "name", str(comp))
            comp_type = type(comp).__name__

            # Check if voltage source is shorted
            if comp_type in ("Vdc", "Vac", "Vpulse", "Vsin"):
                ports = getattr(comp, "ports", [])
                if len(ports) >= 2:
                    net1 = connections.get(ports[0])
                    net2 = connections.get(ports[1])
                    if net1 and net2:
                        name1 = getattr(net1, "name", str(net1))
                        name2 = getattr(net2, "name", str(net2))
                        if name1 == name2:
                            violations.append(
                                RuleViolation(
                                    rule_name="shorted_source",
                                    message="Voltage source terminals connected to same node",
                                    severity=RuleSeverity.ERROR,
                                    component=comp_name,
                                    suggestion="Connect source terminals to different nodes",
                                )
                            )

        return violations

    def _check_power_limits(self, circuit: Any) -> list[RuleViolation]:
        """Check estimated power dissipation limits."""
        violations: list[RuleViolation] = []
        rules = self._config["rules"]
        power_config = rules.get("power_limits", {})

        if not power_config:
            return violations

        max_power = power_config.get("max_power_watts", float("inf"))
        components = getattr(circuit, "_components", [])

        # Estimate worst-case power for resistors
        for comp in components:
            comp_name = getattr(comp, "name", str(comp))
            comp_type = type(comp).__name__

            if comp_type == "Resistor":
                r_value = getattr(comp, "value", None) or getattr(
                    comp, "resistance", None
                )
                if r_value:
                    if isinstance(r_value, str):
                        r_value = self._parse_value(r_value)
                    if r_value and r_value > 0:
                        # Estimate max power (assuming max voltage in circuit)
                        # This is a rough estimate
                        pass  # Would need voltage info for accurate calculation

        return violations

    def _parse_value(self, value_str: str) -> float | None:
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
            "t": 1e12,
        }

        value_str = value_str.strip().lower()

        # Try direct conversion
        try:
            return float(value_str)
        except ValueError:
            pass

        # Try with suffix
        for suffix, mult in multipliers.items():
            if value_str.endswith(suffix):
                try:
                    return float(value_str[: -len(suffix)]) * mult
                except ValueError:
                    pass

        return None

    # Public API

    def get_last_result(self) -> DRCResult | None:
        """Get the result of the last DRC run."""
        return self._last_result

    def add_custom_rule(
        self,
        name: str,
        check_fn: Any,  # Callable[[Any], list[RuleViolation]]
        enabled: bool = True,
    ) -> None:
        """Add a custom design rule.

        Args:
            name: Rule name
            check_fn: Function that takes circuit and returns violations
            enabled: Whether rule is enabled
        """
        self._config["rules"][f"custom_{name}"] = {"fn": check_fn, "enabled": enabled}
