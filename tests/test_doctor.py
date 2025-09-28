from __future__ import annotations

from pathlib import Path

import pytest
from spicelab import doctor


def test_collect_diagnostics_honours_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "ngspice"
    exe.write_text("", encoding="utf-8")
    lib = tmp_path / "libngspice.so"
    lib.write_text("", encoding="utf-8")

    monkeypatch.setenv("SPICELAB_NGSPICE", str(exe))
    monkeypatch.setenv("SPICELAB_NGSPICE_SHARED", str(lib))
    monkeypatch.setenv("SPICELAB_LTSPICE", str(tmp_path / "lt.exe"))

    monkeypatch.setattr(doctor.shutil, "which", lambda name: None, raising=False)
    monkeypatch.setattr(
        "spicelab.spice.ngspice_shared_backend._candidate_paths",
        lambda: [str(lib)],
        raising=False,
    )

    results = doctor.collect_diagnostics()
    statuses = {item.target: item.status for item in results}
    assert statuses.get("ngspice (SPICELAB_NGSPICE)") == "ok"
    assert statuses.get("libngspice") == "ok"

    text = doctor.format_cli(results)
    assert "spicelab environment check" in text
    assert "ngspice" in text
