from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RunArtifacts:
    netlist_path: str
    log_path: str
    raw_path: str | None
    # Optional: working directory used for this run (when applicable)
    workdir: str | None = None


@dataclass
class RunResult:
    artifacts: RunArtifacts
    returncode: int
    stdout: str
    stderr: str


class SimulatorAdapter(Protocol):
    def run_op(self, netlist_text: str) -> RunResult: ...
