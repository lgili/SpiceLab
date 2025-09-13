"""Components gallery: small, runnable snippets for most components.

Run as module:

  PYTHONPATH=src python -m examples.components_gallery --list
  PYTHONPATH=src python -m examples.components_gallery --demo op-rc
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from cat.analysis import AC, DC, OP, TRAN
from cat.core.circuit import Circuit
from cat.core.components import (
    VCCS,
    VCVS,
    AnalogMux8,
    Capacitor,
    Diode,
    OpAmpIdeal,
    Resistor,
    Vac,
    Vdc,
    Vpulse,
)
from cat.core.net import GND, Net


def demo_op_rc() -> None:
    c = Circuit("op_rc")
    vin = Net("vin")
    vout = Net("vout")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", 1000.0)
    C1 = Capacitor("1", 1e-6)
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(C1.ports[1], GND)
    r = OP().run(c)
    print("V(vout)=", float(r.traces["v(vout)"].values[-1]))


def demo_tran_step() -> None:
    c = Circuit("tran_step")
    vin = Net("vin")
    vout = Net("vout")
    Vp = Vpulse("1", 0.0, 1.0, 0.0, 1e-6, 1e-6, 1e-3, 2e-3)
    R = Resistor("1", 1000.0)
    C = Capacitor("1", 1e-6)
    c.add(Vp, R, C)
    c.connect(Vp.ports[0], vin)
    c.connect(Vp.ports[1], GND)
    c.connect(R.ports[0], vin)
    c.connect(R.ports[1], vout)
    c.connect(C.ports[0], vout)
    c.connect(C.ports[1], GND)
    r = TRAN("0.1ms", "5ms").run(c)
    print("last V(vout)=", float(r.traces["v(vout)"].values[-1]))


def demo_ac_divider() -> None:
    c = Circuit("ac_div")
    vin = Net("vin")
    vout = Net("vout")
    V1 = Vac("1", ac_mag=1.0)
    R1 = Resistor("1", 1000.0)
    R2 = Resistor("2", 2000.0)
    c.add(V1, R1, R2)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(R2.ports[0], vout)
    c.connect(R2.ports[1], GND)
    r = AC("lin", 1, 1.0, 1.0).run(c)
    print("|V(vout)|=", float(r.traces["v(vout)"].values[-1]))


def demo_dc_sweep() -> None:
    c = Circuit("dc_div")
    n1 = Net("n1")
    V1 = Vdc("1", 0.0)
    R1 = Resistor("1", 1000.0)
    R2 = Resistor("2", 2000.0)
    c.add(V1, R1, R2)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], n1)
    c.connect(R2.ports[0], n1)
    c.connect(R2.ports[1], GND)
    c.connect(V1.ports[1], GND)
    r = DC("1", 0.0, 5.0, 1.0).run(c)
    v0 = float(r.traces["v(n1)"].values[0])
    v1 = float(r.traces["v(n1)"].values[-1])
    print("V(n1) endpoints:", v0, v1)


def demo_op_oa_buffer() -> None:
    c = Circuit("oa_buf")
    vin = Net("vin")
    vout = Net("vout")
    V = Vdc("1", 1.0)
    u = OpAmpIdeal("1", 1e6)
    c.add(V, u)
    c.connect(V.ports[0], vin)
    c.connect(V.ports[1], GND)
    c.connect(u.ports[0], vin)
    c.connect(u.ports[2], vout)
    c.connect(u.ports[1], vout)
    r = OP().run(c)
    print("V(vout)=", float(r.traces["v(vout)"].values[-1]))


def demo_op_vcvs() -> None:
    c = Circuit("vcvs")
    vin = Net("vin")
    vout = Net("vout")
    V1 = Vdc("1", 1.0)
    E1 = VCVS("1", 2.0)
    RL = Resistor("L", 1000.0)
    c.add(V1, E1, RL)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(E1.ports[2], vin)
    c.connect(E1.ports[3], GND)
    c.connect(E1.ports[0], vout)
    c.connect(E1.ports[1], GND)
    c.connect(RL.ports[0], vout)
    c.connect(RL.ports[1], GND)
    print("Vout=", float(OP().run(c).traces["v(vout)"].values[-1]))


def demo_op_vccs() -> None:
    c = Circuit("vccs")
    vin = Net("vin")
    vout = Net("vout")
    V2 = Vdc("2", 1.0)
    G1 = VCCS("1", 1e-3)
    RL2 = Resistor("L", 1000.0)
    c.add(V2, G1, RL2)
    c.connect(V2.ports[0], vin)
    c.connect(V2.ports[1], GND)
    c.connect(G1.ports[2], vin)
    c.connect(G1.ports[3], GND)
    c.connect(G1.ports[0], GND)
    c.connect(G1.ports[1], vout)  # current into vout
    c.connect(RL2.ports[0], vout)
    c.connect(RL2.ports[1], GND)
    print("Vout=", float(OP().run(c).traces["v(vout)"].values[-1]))


def demo_op_diode() -> None:
    c = Circuit("d_fwd")
    n1 = Net("n1")
    V = Vdc("1", 1.0)
    R = Resistor("1", 1000.0)
    D = Diode("1", "D1")
    c.add_directive(".model D1 D(Is=1e-14)")
    c.add(V, R, D)
    c.connect(V.ports[0], R.ports[0])
    c.connect(R.ports[1], n1)
    c.connect(D.ports[0], n1)
    c.connect(D.ports[1], GND)
    c.connect(V.ports[1], GND)
    print("Vd=", float(OP().run(c).traces["v(n1)"].values[-1]))


def demo_mux() -> None:
    c = Circuit("mux")
    vin = Net("vin")
    vout2 = Net("vout2")
    V1 = Vdc("1", 5.0)
    M = AnalogMux8(ref="MU1", r_series=100.0, sel=2)
    RL = Resistor("L", 1000.0)
    c.add(V1, M, RL)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(M.ports[0], vin)
    # connect all outs; only out2 is loaded/wired
    for i in range(8):
        p = M.ports[1 + i]
        c.connect(p, vout2 if i == 2 else GND)
    c.connect(RL.ports[0], vout2)
    c.connect(RL.ports[1], GND)
    print("V(vout2)=", float(OP().run(c).traces["v(vout2)"].values[-1]))


DEMOS: dict[str, Callable[[], None]] = {
    "op-rc": demo_op_rc,
    "tran-step": demo_tran_step,
    "ac-divider": demo_ac_divider,
    "dc-sweep": demo_dc_sweep,
    "op-oa-buffer": demo_op_oa_buffer,
    "op-vcvs": demo_op_vcvs,
    "op-vccs": demo_op_vccs,
    "op-diode": demo_op_diode,
    "op-mux": demo_mux,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--demo", choices=sorted(DEMOS.keys()))
    args = ap.parse_args()
    if args.list:
        for k in sorted(DEMOS.keys()):
            print(k)
        return
    DEMOS[args.demo]()


if __name__ == "__main__":
    main()
