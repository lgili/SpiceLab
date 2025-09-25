import shutil

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines.factory import create_simulator

# Simple RC transient circuit: Vdc -> R -> C -> GND; probe node between R and C


def build_rc() -> Circuit:
    c = Circuit("rc_test")
    V1 = Vdc("1", "5")  # 5V source
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "1u")
    c.add(V1, R1, C1)
    # Ports: For simplicity assume first two ports defined logically in components (source: p,n)
    c.connect(V1.ports[0], R1.ports[0])  # V1.p -> R1.a (node vout)
    c.connect(R1.ports[1], C1.ports[0])  # R1.b -> C1.a (same node)
    c.connect(C1.ports[1], GND)  # C1.b -> GND
    c.connect(V1.ports[1], GND)  # V1.n -> GND
    return c


# Determine engines available on this system so test skips gracefully.
ENGINES = []
if shutil.which("ngspice"):
    ENGINES.append("ngspice")
if shutil.which("ltspice") or shutil.which("LTspice") or shutil.which("XVIIx64.exe"):
    ENGINES.append("ltspice")
if shutil.which("Xyce") or shutil.which("xyce"):
    ENGINES.append("xyce")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.engine
def test_basic_rc_tran_shapes(engine: str) -> None:
    if not ENGINES:  # pragma: no cover - defensive
        pytest.skip("no external engines installed")
    sim = create_simulator(engine)
    circuit = build_rc()
    tran = AnalysisSpec(mode="tran", args={"tstep": "1u", "tstop": "1m"})
    res = sim.run(circuit, [tran])
    ds = res.dataset()
    assert "analysis_modes" in ds.attrs
    assert "tran" in ds.attrs.get("analysis_modes", [])
    # Expect at least time coord and one voltage variable
    assert any(c in ds.coords for c in ("time", "freq"))
    volt_vars = [v for v in ds.data_vars if v.startswith("V(")]
    assert volt_vars, "expected at least one voltage variable"

    feats = sim.features()
    assert feats.supports_noise is True
    attrs = res.attrs()
    assert attrs["engine_features"]["supports_noise"] is True
