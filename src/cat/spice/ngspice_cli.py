from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunArtifacts:
    workdir: Path
    netlist_path: Path
    raw_path: str | None
    log_path: Path


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    artifacts: RunArtifacts


def _which_ngspice() -> str:
    exe = shutil.which("ngspice")
    if not exe:
        raise RuntimeError("ngspice not found in PATH.")
    return exe


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
        f.write(netlist.rstrip() + "\n")
        # diretivas de análise (fora do .control)
        for line in directives:
            f.write(line.rstrip() + "\n")
        # garantir ASCII para o RAW
        f.write(".control\n")
        f.write("set filetype=ascii\n")
        f.write(".endc\n")
        f.write(".end\n")
    return deck, log


def run_directives(netlist: str, directives: Sequence[str], title: str = "cat_run") -> RunResult:
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
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        artifacts=RunArtifacts(
            workdir=workdir,
            netlist_path=deck,
            raw_path=raw_path,
            log_path=log,
        ),
    )


# Conveniências
def run_tran(netlist: str, tstep: str, tstop: str, tstart: str | None = None) -> RunResult:
    lines: list[str] = []
    if tstart:
        lines.append(f".tran {tstep} {tstop} {tstart}")
    else:
        lines.append(f".tran {tstep} {tstop}")
    return run_directives(netlist, lines, title="tran")


def run_op(netlist: str) -> RunResult:
    return run_directives(netlist, [".op"], title="op")


def run_ac(netlist: str, sweep_type: str, n: int, fstart: float, fstop: float) -> RunResult:
    lines = [f".ac {sweep_type} {n} {fstart} {fstop}"]
    return run_directives(netlist, lines, title="ac")
