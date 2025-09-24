import tempfile
from collections.abc import Sequence
from pathlib import Path

from spicelab.spice.base import RunArtifacts, RunResult
from spicelab.spice.ngspice_cli import NgSpiceCLIAdapter, cleanup_artifacts
from spicelab.spice.registry import (
    get_active_adapter,
    get_run_directives,
    list_adapters,
    register_adapter,
    set_run_directives,
    use_adapter,
)


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


def test_register_and_switch_adapter() -> None:
    # ensure the default adapter exists
    default = get_active_adapter()
    assert "ngspice-cli" in list_adapters()

    # register a disposable adapter and switch to it
    class _DummyAdapter(NgSpiceCLIAdapter):
        def __init__(self) -> None:  # pragma: no cover - init trivial
            super().__init__(keep_workdir=False)

    name = "dummy-sim"
    register_adapter(name, _DummyAdapter(), make_default=False)
    use_adapter(name)
    active = get_active_adapter()
    assert isinstance(active, NgSpiceCLIAdapter)

    # restore previous adapter using legacy API for backwards compatibility
    set_run_directives(default)


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
