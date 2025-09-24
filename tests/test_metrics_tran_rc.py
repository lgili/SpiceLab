import shutil

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND

ng = shutil.which("ngspice")


def _rc_circuit(vstep: float = 1.0) -> Circuit:
    c = Circuit("rc_tran_metrics")
    V1 = Vdc("1", vstep)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # nó vin = V1+
    c.connect(R1.ports[1], C1.ports[0])  # nó vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


# @pytest.mark.skipif(not ng, reason="ngspice not installed")
# def test_tran_metrics_rc() -> None:
#     c = _rc_circuit(1.0)
#     res = TRAN("10us", "5ms").run(c)
#     ts = res.traces
#     # Métricas básicas
#     r = rise_time(ts, "v(n2)")  # nó vout — seu naming pode ser n2 conforme net builder
#     assert r.trise is not None and r.trise > 0
#     ov = overshoot(ts, "v(n2)")
#     # RC de 1ª ordem ideal não overshoota; toleramos ruído numérico
#     assert ov.overshoot < 0.02
#     st = settling_time(ts, "v(n2)", tol=0.02)
#     assert st.t_settle is not None and st.t_settle > 0
