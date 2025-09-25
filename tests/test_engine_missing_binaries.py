import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

import pytest


# To avoid circular imports during collection we re-declare a lightweight version
# of EngineBinaryNotFound matching the real interface. This is sufficient to
# assert that locator functions raise the correct exception type name & fields.
class EngineBinaryNotFound(RuntimeError):
    def __init__(
        self,
        engine: str,
        hints: Iterable[str] | None = None,
        attempted: Iterable[str] | None = None,
    ) -> None:
        self.engine = engine
        self.hints = list(hints or [])
        self.attempted = list(attempted or [])
        hints_txt = "\n".join(f"  - {h}" for h in self.hints)
        tried_txt = ("\nTried: " + ", ".join(self.attempted)) if self.attempted else ""
        msg = f"{engine} binary not found.{tried_txt}\nInstall or configure it. Hints:\n{hints_txt}"
        super().__init__(msg)


PARAMS = [
    ("ngspice", "spicelab/spice/ngspice_cli.py", "_which_ngspice"),
    ("ltspice", "spicelab/spice/ltspice_cli.py", "_which_ltspice"),
    ("xyce", "spicelab/spice/xyce_cli.py", "_which_xyce"),
]


@pytest.mark.parametrize("engine, file_path, locator_name", PARAMS)
def test_missing_engine_raises_enginebinarynotfound(
    engine: str,
    file_path: str,
    locator_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset env vars to force lookup failure
    for var in [
        "SPICELAB_NGSPICE",
        "CAT_SPICE_NGSPICE",
        "SPICELAB_LTSPICE",
        "SPICELAB_XYCE",
        "LTSPICE_EXE",
        "XYCE_EXE",
    ]:
        monkeypatch.delenv(var, raising=False)

    # Force which() to return None for any binary name
    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: None)

    # Extract only the locator function source to avoid importing the heavy module.
    root = Path(__file__).parent.parent / file_path
    text = root.read_text(encoding="utf-8")
    lines: list[str] = []
    capture = False
    indent_level = None
    for ln in text.splitlines():
        if ln.startswith(f"def {locator_name}"):
            capture = True
        if capture:
            # Determine indent after the def line
            if indent_level is None and ln.startswith("def "):
                # next non-empty line sets indent
                pass
            elif indent_level is None and ln.strip() and not ln.startswith("def "):
                indent_level = len(ln) - len(ln.lstrip())
            # If we hit a new top-level def/class after we started and have indent_level
            if (
                ln.startswith("def ")
                and lines
                and indent_level is not None
                and not lines[-1].startswith("def ")
            ):
                # reached a new function; stop
                break
            lines.append(ln)
    if not lines:
        raise AssertionError(f"Could not extract locator source for {locator_name}")
    src = "\n".join(lines)
    # Provide minimal globals needed by locator (os, shutil, Path, EngineBinaryNotFound)
    glb: dict[str, object] = {
        "os": os,
        "shutil": shutil,
        "Path": Path,
        "EngineBinaryNotFound": EngineBinaryNotFound,
    }
    namespace: dict[str, object] = {}
    exec(src, glb, namespace)
    locator_obj = namespace[locator_name]
    if not callable(locator_obj):
        raise AssertionError(f"Extracted locator '{locator_name}' is not callable")
    locator = cast(Callable[[], object], locator_obj)

    # Ensure fallback paths such as /Applications/LTspice... are treated as missing
    monkeypatch.setattr(Path, "exists", lambda self: False, raising=False)

    with pytest.raises(EngineBinaryNotFound) as excinfo:
        locator()

    err = excinfo.value
    assert err.engine == engine
    assert err.hints and any(h for h in err.hints)
    s = str(err).lower()
    assert engine in s
    assert ("install" in s) or ("set" in s)
