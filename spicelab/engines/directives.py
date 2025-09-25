from __future__ import annotations

from collections.abc import Sequence

from ..core.types import AnalysisSpec

__all__ = ["spec_to_directives", "specs_to_directives"]


def spec_to_directives(spec: AnalysisSpec) -> list[str]:
    """Map an AnalysisSpec to one or more SPICE analysis directives.

    Supported modes: op, tran, ac, dc. Noise left for future (raises
    NotImplementedError). The function is intentionally engine-agnostic: engines
    differ mainly in execution, not directive syntax for the common analyses we
    target here.
    """
    mode = spec.mode
    a = spec.args
    if mode == "op":
        return [".op"]
    if mode == "tran":
        tstep = a.get("tstep")
        tstop = a.get("tstop")
        tstart = a.get("tstart")
        if tstep is None or tstop is None:
            raise ValueError("TRAN requires 'tstep' and 'tstop' args")
        if tstart is not None:
            return [f".tran {tstep} {tstop} {tstart}"]
        return [f".tran {tstep} {tstop}"]
    if mode == "ac":
        sweep_type = a.get("sweep_type", "dec")
        n = a.get("n")
        fstart = a.get("fstart")
        fstop = a.get("fstop")
        if n is None or fstart is None or fstop is None:
            raise ValueError("AC requires 'n', 'fstart', 'fstop' args")
        return [f".ac {sweep_type} {n} {fstart} {fstop}"]
    if mode == "dc":
        src = a.get("src")
        start = a.get("start")
        stop = a.get("stop")
        step = a.get("step")
        if src is None or start is None or stop is None or step is None:
            raise ValueError("DC requires 'src', 'start', 'stop', 'step' args")
        src_ref = src if str(src).upper().startswith("V") else f"V{src}"
        return [f".dc {src_ref} {start} {stop} {step}"]
    if mode == "noise":  # future work
        raise NotImplementedError("noise analysis mapping not implemented yet")
    raise NotImplementedError(f"Unsupported analysis mode: {mode}")


def specs_to_directives(specs: Sequence[AnalysisSpec]) -> list[str]:
    out: list[str] = []
    for s in specs:
        out.extend(spec_to_directives(s))
    return out
