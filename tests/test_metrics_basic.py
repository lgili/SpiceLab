import shutil
from typing import Any

import pytest
from spicelab.analysis import (
    TRAN,
    gain_db_from_traces,
    overshoot_pct,
    peak,
    settling_time,
)
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND


def _rc_tran(vstep: float = 1.0) -> Circuit:
    c = Circuit("rc_basic_metrics")
    V1 = Vdc("1", vstep)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    # V1.p -> R1.a -> nó n1 (saída), R1.b -> C1.a, V1.n -> GND, C1.b -> GND
    c.connect(V1.ports[0], R1.ports[0])  # nó n1
    c.connect(R1.ports[1], C1.ports[0])  # nó n2 (pode não aparecer conforme netlist)
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


def _pick_voltage_trace_name(ts: Any) -> str:
    names = getattr(ts, "names", [])
    if "v(n2)" in names:
        return "v(n2)"
    if "v(n1)" in names:
        return "v(n1)"
    for nm in names:
        if isinstance(nm, str) and nm.startswith("v("):
            return nm
    raise AssertionError(f"Nenhum traço de tensão encontrado em {names!r}")


@pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice not installed")
def test_basic_metrics_rc() -> None:
    c = _rc_tran(1.0)
    res = TRAN("10us", "5ms").run(c)
    ts = res.traces
    yname = _pick_voltage_trace_name(ts)

    # ganho "dc" aproximado do degrau (helper simples) -> escalar
    gdb = gain_db_from_traces(ts, yname)
    assert isinstance(gdb, float)

    # pico -> escalar
    pk = peak(ts, yname)
    assert isinstance(pk, float)
    assert pk >= 0.0

    # overshoot em % -> escalar (RC ideal não overshoota)
    ovp = overshoot_pct(ts, yname)
    assert isinstance(ovp, float)
    assert ovp <= 2.0

    # settling
    st = settling_time(ts, yname, tol=0.02)
    assert st.t_settle is None or st.t_settle >= 0.0
