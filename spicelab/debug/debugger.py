"""Simulation step-by-step debugger.

Provides interactive debugging capabilities for simulations:
- Step through simulation phases
- Inspect intermediate state
- Modify parameters before execution
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..core.types import AnalysisSpec, ResultHandle


class DebugPhase(Enum):
    """Phases of simulation execution."""

    INIT = auto()
    VALIDATE = auto()
    NETLIST = auto()
    ANALYSIS = auto()
    ENGINE = auto()
    EXECUTE = auto()
    RESULTS = auto()
    COMPLETE = auto()


@dataclass
class DebugStep:
    """Information about a single debug step.

    Represents the state at a particular phase of simulation.
    """

    phase: DebugPhase
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    can_modify: bool = False
    completed: bool = False

    def __str__(self) -> str:
        status = "✓" if self.completed else "○"
        return f"[{status}] {self.phase.name}: {self.description}"


class SimulationDebugger:
    """Step-by-step simulation debugger.

    Allows stepping through simulation phases with inspection and
    modification capabilities at each step.

    Example:
        >>> from spicelab.debug import SimulationDebugger
        >>> from spicelab.templates import rc_lowpass
        >>> from spicelab.core.types import AnalysisSpec
        >>>
        >>> circuit = rc_lowpass(fc=1000)
        >>> analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20,
        ...                                  "fstart": 1, "fstop": 1e6})]
        >>>
        >>> debugger = SimulationDebugger(circuit, analyses)
        >>>
        >>> # Step through manually
        >>> step = debugger.current_step()
        >>> print(step)
        [○] INIT: Initialize simulation context
        >>>
        >>> debugger.step()  # Move to next phase
        >>> print(debugger.current_step())
        [○] VALIDATE: Validate circuit structure
        >>>
        >>> # Inspect data at current step
        >>> print(debugger.inspect())
        {'circuit_name': 'RC_Filter', 'component_count': 2}
        >>>
        >>> # Run to completion
        >>> result = debugger.run_to_completion()

        >>> # Or use callbacks
        >>> def on_step(step: DebugStep):
        ...     print(f"Phase: {step.phase.name}")
        ...
        >>> debugger = SimulationDebugger(circuit, analyses, on_step=on_step)
        >>> result = debugger.run_to_completion()
    """

    def __init__(
        self,
        circuit: Any,
        analyses: Sequence[AnalysisSpec],
        *,
        engine: str = "ngspice",
        on_step: Callable[[DebugStep], None] | None = None,
        on_error: Callable[[DebugStep, Exception], None] | None = None,
    ) -> None:
        """Initialize the debugger.

        Args:
            circuit: Circuit to simulate
            analyses: Analysis specifications
            engine: Simulation engine (default "ngspice")
            on_step: Callback called at each step
            on_error: Callback called on errors
        """
        self._circuit = circuit
        self._analyses = list(analyses)
        self._engine = engine
        self._on_step = on_step
        self._on_error = on_error

        # Internal state
        self._current_phase = DebugPhase.INIT
        self._steps: list[DebugStep] = []
        self._netlist: str = ""
        self._result: ResultHandle | None = None
        self._error: Exception | None = None

        # Initialize steps
        self._init_steps()

    def _init_steps(self) -> None:
        """Initialize the step sequence."""
        self._steps = [
            DebugStep(
                phase=DebugPhase.INIT,
                description="Initialize simulation context",
                can_modify=True,
            ),
            DebugStep(
                phase=DebugPhase.VALIDATE,
                description="Validate circuit structure",
                can_modify=False,
            ),
            DebugStep(
                phase=DebugPhase.NETLIST,
                description="Generate SPICE netlist",
                can_modify=True,
            ),
            DebugStep(
                phase=DebugPhase.ANALYSIS,
                description="Configure analyses",
                can_modify=True,
            ),
            DebugStep(
                phase=DebugPhase.ENGINE,
                description="Initialize simulation engine",
                can_modify=False,
            ),
            DebugStep(
                phase=DebugPhase.EXECUTE,
                description="Execute simulation",
                can_modify=False,
            ),
            DebugStep(
                phase=DebugPhase.RESULTS,
                description="Process results",
                can_modify=False,
            ),
            DebugStep(
                phase=DebugPhase.COMPLETE,
                description="Simulation complete",
                can_modify=False,
            ),
        ]
        self._step_index = 0

    def current_step(self) -> DebugStep:
        """Get the current step."""
        return self._steps[self._step_index]

    def current_phase(self) -> DebugPhase:
        """Get the current phase."""
        return self._steps[self._step_index].phase

    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self._current_phase == DebugPhase.COMPLETE

    def has_error(self) -> bool:
        """Check if an error occurred."""
        return self._error is not None

    def get_error(self) -> Exception | None:
        """Get the error if one occurred."""
        return self._error

    def inspect(self) -> dict[str, Any]:
        """Inspect current step data.

        Returns:
            Dictionary with current step information
        """
        step = self.current_step()
        data: dict[str, Any] = {
            "phase": step.phase.name,
            "description": step.description,
            "can_modify": step.can_modify,
            "completed": step.completed,
        }

        if step.phase == DebugPhase.INIT:
            data["circuit_name"] = getattr(self._circuit, "name", "Unknown")
            data["component_count"] = len(getattr(self._circuit, "_components", []))
            data["engine"] = self._engine

        elif step.phase == DebugPhase.VALIDATE:
            data["validation_passed"] = self._error is None

        elif step.phase == DebugPhase.NETLIST:
            data["netlist_length"] = len(self._netlist)
            data["netlist_preview"] = (
                self._netlist[:500] + "..." if len(self._netlist) > 500 else self._netlist
            )

        elif step.phase == DebugPhase.ANALYSIS:
            data["analyses"] = [{"mode": a.mode, "args": a.args} for a in self._analyses]

        elif step.phase == DebugPhase.RESULTS and self._result:
            try:
                ds = self._result.dataset()
                data["variables"] = list(ds.data_vars)
                data["coordinates"] = list(ds.coords)
            except Exception:
                data["result_available"] = self._result is not None

        data.update(step.data)
        return data

    def step(self) -> DebugStep:
        """Execute current step and advance to next.

        Returns:
            The completed step

        Raises:
            RuntimeError: If already complete or error occurred
        """
        if self.is_complete():
            raise RuntimeError("Simulation already complete")
        if self.has_error():
            raise RuntimeError(f"Cannot continue after error: {self._error}")

        current = self.current_step()

        try:
            self._execute_step(current)
            current.completed = True

            if self._on_step:
                self._on_step(current)

            # Advance to next step
            if self._step_index < len(self._steps) - 1:
                self._step_index += 1
                self._current_phase = self._steps[self._step_index].phase

        except Exception as e:
            self._error = e
            if self._on_error:
                self._on_error(current, e)
            raise

        return current

    def _execute_step(self, step: DebugStep) -> None:
        """Execute a single step."""
        if step.phase == DebugPhase.INIT:
            # Nothing to do - just initialization
            step.data["initialized"] = True

        elif step.phase == DebugPhase.VALIDATE:
            # Run circuit validation
            from .dry_run import dry_run

            result = dry_run(self._circuit, self._analyses, engine=self._engine)
            step.data["validation_result"] = result
            if not result.valid:
                errors = [str(e) for e in result.errors]
                raise ValueError(f"Circuit validation failed: {errors}")

        elif step.phase == DebugPhase.NETLIST:
            # Generate netlist
            self._netlist = self._circuit.build_netlist()
            step.data["netlist"] = self._netlist

        elif step.phase == DebugPhase.ANALYSIS:
            # Validate analyses
            step.data["analysis_count"] = len(self._analyses)

        elif step.phase == DebugPhase.ENGINE:
            # Create simulator
            from ..engines.factory import create_simulator

            self._simulator = create_simulator(self._engine)
            step.data["engine_created"] = True

        elif step.phase == DebugPhase.EXECUTE:
            # Run simulation
            self._result = self._simulator.run(self._circuit, self._analyses, None, None)
            step.data["simulation_complete"] = True

        elif step.phase == DebugPhase.RESULTS:
            # Process results
            if self._result:
                try:
                    ds = self._result.dataset()
                    step.data["data_vars"] = len(ds.data_vars)
                except Exception:
                    pass

        elif step.phase == DebugPhase.COMPLETE:
            step.data["complete"] = True

    def run_to_phase(self, phase: DebugPhase) -> DebugStep:
        """Run until reaching a specific phase.

        Args:
            phase: Target phase to stop at

        Returns:
            The step at the target phase
        """
        while self.current_phase() != phase and not self.is_complete():
            self.step()
        return self.current_step()

    def run_to_completion(self) -> ResultHandle:
        """Run simulation to completion.

        Returns:
            Simulation results

        Raises:
            RuntimeError: If simulation fails
        """
        while not self.is_complete():
            self.step()

        if self._result is None:
            raise RuntimeError("Simulation did not produce results")

        return self._result

    def get_netlist(self) -> str:
        """Get the generated netlist.

        Must be called after NETLIST phase.
        """
        if not self._netlist:
            raise RuntimeError("Netlist not yet generated - step to NETLIST phase first")
        return self._netlist

    def get_result(self) -> ResultHandle | None:
        """Get simulation result if available."""
        return self._result

    def modify_analysis(self, index: int, **kwargs: Any) -> None:
        """Modify an analysis before execution.

        Can only be called during ANALYSIS phase.

        Args:
            index: Index of analysis to modify
            **kwargs: Parameters to update
        """
        if self.current_phase() != DebugPhase.ANALYSIS:
            raise RuntimeError("Can only modify analyses during ANALYSIS phase")

        if index < 0 or index >= len(self._analyses):
            raise IndexError(f"Analysis index {index} out of range")

        from ..core.types import AnalysisSpec

        old = self._analyses[index]
        new_args = {**old.args, **kwargs}
        self._analyses[index] = AnalysisSpec(old.mode, new_args)

    def get_all_steps(self) -> list[DebugStep]:
        """Get all steps in the debug sequence."""
        return self._steps.copy()

    def __str__(self) -> str:
        lines = ["Simulation Debugger Status:", "=" * 40]
        for i, step in enumerate(self._steps):
            marker = "→" if i == self._step_index else " "
            lines.append(f"{marker} {step}")
        if self._error:
            lines.append(f"\nError: {self._error}")
        return "\n".join(lines)


__all__ = ["SimulationDebugger", "DebugStep", "DebugPhase"]
