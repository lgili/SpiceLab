from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .base import RunArtifacts, RunResult, SimulatorAdapter


def _which_ltspice() -> str:
    """Locate the LTspice CLI binary across platforms.

    Tries, in order:
      - environment variable SPICELAB_LTSPICE (explicit path)
      - common macOS app bundle binary
      - LTspice on PATH (Windows/Linux if available)
    """

    env_exe = os.environ.get("SPICELAB_LTSPICE")
    if env_exe and Path(env_exe).exists():
        return env_exe

    # macOS LTspice app bundle (LTspice 24)
    mac_paths = [
        "/Applications/LTspice.app/Contents/MacOS/LTspice",
        "/Applications/LTspice.app/Contents/MacOS/LTspiceXVII",
        "/Applications/LTspice.app/Contents/MacOS/LTspice64",
    ]
    for p in mac_paths:
        if Path(p).exists():
            return p

    exe = shutil.which("ltspice") or shutil.which("LTspice") or shutil.which("XVIIx64.exe")
    if not exe:
        raise RuntimeError(
            "LTspice binary not found. Set SPICELAB_LTSPICE or install LTspice and add it to PATH."
        )
    return exe


def _write_deck(
    workdir: Path,
    title: str,
    netlist: str,
    directives: Sequence[str],
) -> tuple[Path, Path]:
    """Write a .cir deck with analysis directives appended."""

    deck = workdir / "deck.cir"
    log = workdir / "ltspice.log"
    with deck.open("w", encoding="utf-8") as f:
        f.write(f"* {title}\n")
        f.write(netlist.rstrip() + "\n")
        for line in directives:
            f.write(line.rstrip() + "\n")
        f.write(".end\n")
    return deck, log


def run_directives(
    netlist: str,
    directives: Sequence[str],
    title: str = "spicelab_lt",
    *,
    keep_workdir: bool = True,
) -> RunResult:
    """Run LTspice headless on a generated deck.

    We request ASCII RAW with '-ascii'. Exact flags can vary per version/OS; if not
    supported, LTspice will still produce the default .raw which may be binary.
    """

    exe = _which_ltspice()
    workdir = Path(tempfile.mkdtemp(prefix="spicelab_lt_"))
    deck, log = _write_deck(workdir, title=title, netlist=netlist, directives=directives)

    # Expected RAW path: same base as deck, or as specified by LTspice default
    raw_out = workdir / "deck.raw"

    # Common CLI: -b (batch), -Run, -ascii to force ASCII raw
    cmd = [exe, "-b", "-Run", "-ascii", str(deck)]
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

    # Some LTspice builds write logs to stdout/stderr only
    with log.open("w", encoding="utf-8") as lf:
        if proc.stdout:
            lf.write(proc.stdout)
        if proc.stderr:
            lf.write("\n" + proc.stderr)

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


class LtSpiceCLIAdapter(SimulatorAdapter):
    def __init__(self, *, keep_workdir: bool = True) -> None:
        self.keep_workdir = keep_workdir

    def run_directives(self, netlist_text: str, directives: Sequence[str]) -> RunResult:
        return run_directives(netlist_text, directives, keep_workdir=self.keep_workdir)


DEFAULT_ADAPTER = LtSpiceCLIAdapter()
