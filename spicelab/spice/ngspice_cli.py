from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ..engines.exceptions import EngineBinaryNotFound
from .base import RunArtifacts, RunResult, SimulatorAdapter


def _which_ngspice() -> str:
    env_exe = os.environ.get("CAT_SPICE_NGSPICE") or os.environ.get("SPICELAB_NGSPICE")
    if env_exe:
        return env_exe
    attempted: list[str] = []
    exe = shutil.which("ngspice")
    if exe:
        return exe
    hints = [
        "macOS (Homebrew): brew install ngspice",
        "Ubuntu/Debian: sudo apt-get install ngspice",
        "Fedora: sudo dnf install ngspice",
        "Windows: download installer from https://sourceforge.net/projects/ngspice/files/",
        "Or set CAT_SPICE_NGSPICE / SPICELAB_NGSPICE with the full path.",
    ]
    raise EngineBinaryNotFound("ngspice", hints=hints, attempted=attempted)


def _strip_final_end(netlist: str) -> str:
    """Remove a trailing '.end' (case-insensitive) and blank lines from a netlist string.

    This allows Circuit.build_netlist() to include '.end' to satisfy tests and
    standalone usage, while the runner appends analysis directives and a single
    final '.end' after the control block without duplication.
    """
    lines = netlist.splitlines()
    # drop trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    # drop a single trailing .end (case-insensitive)
    if lines and lines[-1].strip().lower() == ".end":
        lines.pop()
        # also drop any additional blank lines that may precede the .end we removed
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines)


def _write_deck(
    workdir: Path,
    title: str,
    netlist: str,
    directives: Sequence[str],
) -> tuple[Path, Path]:
    """
    Escreve o deck .cir com:
      - netlist dos componentes
      - diretivas de análise (.op / .tran / .ac ...)
      - bloco .control com 'set filetype=ascii' (para RAW ASCII)
      - .end
    A execução da análise é disparada pelo ngspice (modo batch) e o arquivo .raw
    é forçado via parâmetro de linha de comando '-r'.
    """
    deck = workdir / "deck.cir"
    log = workdir / "ngspice.log"
    with deck.open("w", encoding="utf-8") as f:
        f.write(f"* {title}\n")
        # Avoid duplicated '.end' when appending directives/control/end here
        net_no_end = _strip_final_end(netlist)
        if net_no_end:
            f.write(net_no_end.rstrip() + "\n")
        # diretivas de análise (fora do .control)
        for line in directives:
            f.write(line.rstrip() + "\n")
        # garantir ASCII para o RAW
        f.write(".control\n")
        f.write("set filetype=ascii\n")
        f.write(".endc\n")
        f.write(".end\n")
    return deck, log


def run_directives(
    netlist: str,
    directives: Sequence[str],
    title: str = "cat_run",
    *,
    keep_workdir: bool = True,
) -> RunResult:
    """
    Roda NGSpice em modo batch com deck controlado e retorna artefatos.
    Forçamos:
      - RAW gerado em 'sim.raw' via '-r'
      - formato ASCII via 'set filetype=ascii' no bloco .control
    """
    exe = _which_ngspice()
    workdir = Path(tempfile.mkdtemp(prefix="cat_ng_"))
    deck, log = _write_deck(workdir, title=title, netlist=netlist, directives=directives)

    raw_out = workdir / "sim.raw"
    cmd = [exe, "-b", "-o", str(log), "-r", str(raw_out), str(deck)]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    raw_path: str | None = str(raw_out) if raw_out.exists() else None

    return RunResult(
        artifacts=RunArtifacts(
            netlist_path=str(deck),
            log_path=str(log),
            raw_path=raw_path,
            workdir=str(workdir) if keep_workdir else str(workdir),
        ),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


# --------------------------------------------------------------------------------------
# Adapter implementation
# --------------------------------------------------------------------------------------


class NgSpiceCLIAdapter(SimulatorAdapter):
    """Adapter that wraps the ngspice CLI helper functions."""

    def __init__(self, *, keep_workdir: bool = True) -> None:
        self.keep_workdir = keep_workdir

    def run_directives(self, netlist_text: str, directives: Sequence[str]) -> RunResult:
        return run_directives(netlist_text, directives, keep_workdir=self.keep_workdir)


# Convenience singleton used by the registry
DEFAULT_ADAPTER = NgSpiceCLIAdapter()


# Conveniências
def run_tran(
    netlist: str,
    tstep: str,
    tstop: str,
    tstart: str | None = None,
    *,
    keep_workdir: bool = True,
) -> RunResult:
    lines: list[str] = []
    if tstart:
        lines.append(f".tran {tstep} {tstop} {tstart}")
    else:
        lines.append(f".tran {tstep} {tstop}")
    return run_directives(netlist, lines, title="tran", keep_workdir=keep_workdir)


def run_op(netlist: str, *, keep_workdir: bool = True) -> RunResult:
    return run_directives(netlist, [".op"], title="op", keep_workdir=keep_workdir)


def run_ac(
    netlist: str,
    sweep_type: str,
    n: int,
    fstart: float,
    fstop: float,
    *,
    keep_workdir: bool = True,
) -> RunResult:
    lines = [f".ac {sweep_type} {n} {fstart} {fstop}"]
    return run_directives(netlist, lines, title="ac", keep_workdir=keep_workdir)


def cleanup_artifacts(art: RunArtifacts) -> None:
    """Remove a pasta de trabalho se ela tiver sido criada por este runner (cat_ng_*)."""
    wd = art.workdir
    if not wd:
        return
    p = Path(wd)
    try:
        if p.is_dir() and p.name.startswith("cat_ng_"):
            # remove somente se for diretório temporário nosso
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
