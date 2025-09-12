from __future__ import annotations

from ..core.circuit import Circuit
from ..core.components import OA, OpAmpIdeal, R, Resistor
from ..core.net import Net


def opamp_buffer(c: Circuit, inp: Net, out: Net, *, gain: float | str = 1e6) -> OpAmpIdeal:
    """Wire an ideal op-amp as a voltage follower (buffer).

    - inp -> OA.inp
    - OA.out -> out and OA.inn (feedback)
    """
    u = OA(gain)
    c.add(u)
    # inp (non-inverting)
    c.connect(u.ports[0], inp)
    # out node
    c.connect(u.ports[2], out)
    # feedback: out -> inn
    c.connect(u.ports[1], out)
    return u


def opamp_inverting(
    c: Circuit,
    inp: Net,
    out: Net,
    ref: Net,
    *,
    Rin: float | str,
    Rf: float | str,
    gain: float | str = 1e6,
) -> tuple[OpAmpIdeal, Resistor, Resistor]:
    """Wire an inverting amplifier using an ideal op-amp.

    Topology:
      - OA.inp tied to ref
      - Rin from inp to OA.inn
      - Rf from out to OA.inn
      - OA.out -> out
    """
    u = OA(gain)
    rin = R(Rin)
    rf = R(Rf)
    c.add(u, rin, rf)

    # Non-inverting at reference
    c.connect(u.ports[0], ref)

    # OA output to out
    c.connect(u.ports[2], out)

    # Rin: inp -> OA.inn
    c.connect(rin.ports[0], inp)
    c.connect(rin.ports[1], u.ports[1])

    # Rf: out -> OA.inn
    c.connect(rf.ports[0], out)
    c.connect(rf.ports[1], u.ports[1])

    return u, rin, rf
