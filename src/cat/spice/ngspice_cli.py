from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from ..utils.log import get_logger
from .base import RunArtifacts, RunResult

log = get_logger("cat.spice.ngspice")


def _find_ngspice() -> str | None:
    # permite override via env var
    env = os.environ.get("CAT_SPICE_NGSPICE")
    if env and shutil.which(env):
        return env
    return shutil.which("ngspice")


def run_op(netlist_text: str) -> RunResult:
    return run_directives(netlist_text, [".op", ".option filetype=ascii", ".save all"])


def run_tran(netlist_text: str, tstep: str, tstop: str) -> RunResult:
    return run_directives(
        netlist_text,
        [f".tran {tstep} {tstop}", ".option filetype=ascii", ".save all"],
    )


def run_directives(netlist_text: str, directives: list[str]) -> RunResult:
    exe = _find_ngspice()
    if not exe:
        raise RuntimeError("ngspice executable not found on PATH.")
    td = tempfile.mkdtemp(prefix="cat_ngspice_")
    net = os.path.join(td, "circuit.sp")
    logp = os.path.join(td, "out.log")
    rawp = os.path.join(td, "sim.raw")
    with open(net, "w", encoding="utf-8") as f:
        f.write(netlist_text.rstrip() + "\n")
        for d in directives:
            f.write(d.rstrip() + "\n")
    cmd = [exe, "-b", "-o", logp, "-r", rawp, net]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return RunResult(
        artifacts=RunArtifacts(netlist_path=net, log_path=logp, raw_path=rawp),
        returncode=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
    )
