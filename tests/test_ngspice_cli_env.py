import pytest
from spicelab.spice.ngspice_cli import _which_ngspice


def test_which_ngspice_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAT_SPICE_NGSPICE", "/custom/ngspice")
    assert _which_ngspice() == "/custom/ngspice"
