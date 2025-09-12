from __future__ import annotations

from dataclasses import dataclass

from ..core.circuit import Circuit
from ..io.raw_reader import TraceSet, parse_ngspice_ascii_raw_multi
from ..spice.registry import get_run_directives


@dataclass(frozen=True)
class StepNativeResult:
    tracesets: list[TraceSet]
    # Nota: parâmetros por step não são triviais de extrair do RAW; por ora retornamos só os plots.


def run_step_native(circuit: Circuit, directives: list[str]) -> StepNativeResult:
    """
    Roda um deck com diretivas .step nativas (já contidas em `directives`)
    e retorna todos os plots como lista de TraceSet.
    """
    net = circuit.build_netlist()
    run_directives = get_run_directives()
    res = run_directives(net, directives)
    if res.returncode != 0 or not res.artifacts.raw_path:
        raise RuntimeError("NGSpice failed for native .step run")
    sets = parse_ngspice_ascii_raw_multi(res.artifacts.raw_path)
    return StepNativeResult(tracesets=sets)
