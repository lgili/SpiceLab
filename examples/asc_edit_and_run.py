"""Example: load LTspice (.asc/.net/.cir), tweak a resistor, run and plot.

Behavior
- Uses the unified reader ``spicelab.io.read_ltspice`` so you can pass .asc, .net,
    or .cir. Unknown ASC symbols try to import via SpiceLine or fall back to
    ideal components where possible, logging a warning.
- Prefers NGSpice. If not installed, falls back to a tiny synthetic runner so
    the example still produces a plot for docs/CI.

Usage: python examples/asc_edit_and_run.py
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from spicelab.analysis import TRAN
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.io import read_ltspice
from spicelab.spice import ngspice_cli
from spicelab.spice.base import RunArtifacts, RunResult
from spicelab.spice.registry import get_active_adapter, set_run_directives

plt: Any | None
try:
    plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:  # pragma: no cover - optional
    plt = None

# output folder
OUT_DIR = Path("./examples_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# fallback fake runner (simple synthetic transient for demo)

ASCII_RAW_TEMPLATE = """Title:  transient
Date:   Thu Sep  1 12:00:00 2025
Plotname: Transient
Flags: real
No. Variables: 2
No. Points: {npoints}
Variables:
        0       time    time
        1       v(vout) voltage
Values:
{values}
"""


def _fake_transient_runner(netlist: str, directives: list[str]) -> RunResult:
    # Create a small ASCII RAW file with time and v(vout)
    t = np.linspace(0, 1e-3, 201)
    v = 1.0 * np.exp(-t / (100e-6)) * (1 - np.cos(2 * np.pi * 1e3 * t))

    lines = []
    for i, (ti, vi) in enumerate(zip(t, v, strict=False)):
        # NGSpice ASCII uses index then values; complex entries would be 're,im'
        lines.append(f"\t{i}\t{ti:.12g}\t{vi:.12g}")

    content = ASCII_RAW_TEMPLATE.format(npoints=len(t), values="\n".join(lines))

    td = tempfile.mkdtemp()
    raw = os.path.join(td, "sim.raw")
    with open(raw, "w", encoding="utf-8") as f:
        f.write(content)
    log = os.path.join(td, "ngspice.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write("[fake ngspice] ok\n")

    art = RunArtifacts(
        netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw, workdir=td
    )
    return RunResult(artifacts=art, returncode=0, stdout="", stderr="")


def load_and_modify(path: Path, ref_res: str | None = None, new_value: str = "2k") -> Circuit:
    """Load LTspice file (.asc/.net/.cir), change a resistor value and return Circuit."""
    circ = read_ltspice(path)

    # update resistor
    found = False
    for c in list(getattr(circ, "_components", [])):
        if isinstance(c, Resistor):
            if ref_res is None or c.ref == ref_res:
                c.value = new_value
                found = True
                break
    if not found:
        raise RuntimeError("No resistor found to modify in the circuit")
    return circ


def run_and_plot(circ: Circuit, *, show: bool = False) -> Path | None:
    # Prefer real ngspice; if unavailable, fall back to a tiny synthetic runner
    try:
        _ = ngspice_cli._which_ngspice()
    except Exception:
        set_run_directives(_fake_transient_runner)
        print("[info] ngspice not found; using synthetic fake runner for demo")

    # run transient: 1us step to 1ms
    netlist = circ.build_netlist()
    print("--- Generated netlist ---")
    print(netlist)
    print("-------------------------")

    adapter = get_active_adapter()
    directives = TRAN("1e-6", "1e-3", "0.0")._directives()
    res = adapter.run_directives(netlist, directives)
    if res.returncode != 0:
        # try to show the ngspice log for diagnostics
        try:
            with open(res.artifacts.log_path, encoding="utf-8", errors="ignore") as f:
                print("--- NGSpice log ---")
                print(f.read())
                print("-------------------")
        except Exception as exc:
            print("Could not read ngspice log:", exc)
        # also show the deck file that was passed to ngspice
        try:
            print("--- Deck file passed to ngspice ---")
            with open(res.artifacts.netlist_path, encoding="utf-8", errors="ignore") as f:
                for i, ln in enumerate(f, start=1):
                    print(f"{i:03d}: {ln.rstrip()}")
            print("-----------------------------------")
        except Exception as exc:
            print("Could not read deck file:", exc)
        # Fallback to fake runner to keep the example working
        print("[warn] ngspice run failed; falling back to synthetic runner")
        set_run_directives(_fake_transient_runner)
        adapter = get_active_adapter()
        res = adapter.run_directives(netlist, directives)

    # extract traces and plot
    time = np.asarray(res.traces["time"].values, dtype=float)
    vout = np.asarray(res.traces["v(vout)"].values, dtype=float)

    if plt is None:
        print("time samples:", time[:5])
        print("vout samples:", vout[:5])
        return None

    fig, ax = plt.subplots()
    ax.plot(time, vout)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Vout [V]")
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / "asc_edit_vout.png"
    fig.tight_layout()
    fig.savefig(out)
    if show:
        plt.show()
    # cleanup
    try:
        # res may not have .run when adapter is function; cleanup best-effort
        art = getattr(res, "artifacts", None)
        if art:
            ngspice_cli.cleanup_artifacts(art)
    except Exception:
        pass
    return out


def main() -> None:
    asc = Path("examples/PT1000_circuit_1.asc")
    circ = load_and_modify(asc, new_value="2k")
    out = run_and_plot(circ)
    print("Plot saved to:", out)


if __name__ == "__main__":
    main()
