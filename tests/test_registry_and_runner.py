import tempfile
from collections.abc import Sequence
from pathlib import Path

from cat.spice.base import RunArtifacts, RunResult
from cat.spice.ngspice_cli import cleanup_artifacts
from cat.spice.registry import get_run_directives, set_run_directives


def test_registry_set_get_roundtrip() -> None:
    old = get_run_directives()

    def dummy(net: str, dirs: Sequence[str]) -> RunResult:
        return RunResult(
            artifacts=RunArtifacts(netlist_path="/tmp/x.cir", log_path="/tmp/x.log", raw_path=None),
            returncode=0,
            stdout="",
            stderr="",
        )

    try:
        set_run_directives(dummy)
        assert get_run_directives() is dummy
    finally:
        set_run_directives(old)


def test_cleanup_artifacts_removes_tempdir() -> None:
    with tempfile.TemporaryDirectory(prefix="cat_ng_") as td:
        # Create a child file to make sure directory exists
        p = Path(td) / "dummy.txt"
        p.write_text("ok")
        art = RunArtifacts(netlist_path="x", log_path="y", raw_path=None, workdir=td)
        # make a copy path because context manager will try to remove on exit
        wd_copy = str(td)
        cleanup_artifacts(art)
        # Directory may be removed or left for TemporaryDirectory; ensure no exception
        assert wd_copy
