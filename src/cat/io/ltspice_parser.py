from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.circuit import Circuit
from ..core.components import Capacitor, Resistor, Vac, Vdc  # sem Inductor por enquanto
from ..core.net import GND, Net


@dataclass(frozen=True)
class ParsedDeck:
    title: str | None
    circuit: Circuit


def _tok(line: str) -> list[str]:
    s = line.strip()
    if not s or s.startswith("*"):
        return []
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s.split()


def _is_directive(tokens: list[str]) -> bool:
    return bool(tokens) and tokens[0].startswith(".")


def _net_of(name: str, byname: dict[str, Net]) -> Net:
    if name == "0":
        return GND
    n = byname.get(name)
    if n is None:
        n = Net(name)
        byname[name] = n
    return n


def _parse_value(val: str) -> str:
    # Mantemos como string; conversões ficam para utils/units quando necessário
    return val


def from_spice_netlist(text: str, *, title: str | None = None) -> Circuit:
    """
    Converte um netlist SPICE (LTspice exportado via View→SPICE Netlist) em Circuit.
    MVP: R, C, V (DC/AC). Linhas desconhecidas são ignoradas com segurança.
    """
    c = Circuit(title or "imported")
    nets: dict[str, Net] = {}

    for raw in text.splitlines():
        t = _tok(raw)
        if not t or _is_directive(t):
            continue

        card = t[0]
        prefix = card[0].upper()
        ref = card[1:]  # 'R1' -> '1'

        if prefix == "R":
            # Rref n1 n2 value
            if len(t) < 4:
                continue
            _, n1, n2, val = t[:4]
            r = Resistor(ref, _parse_value(val))
            c.add(r)
            c.connect(r.ports[0], _net_of(n1, nets))
            c.connect(r.ports[1], _net_of(n2, nets))

        elif prefix == "C":
            if len(t) < 4:
                continue
            _, n1, n2, val = t[:4]
            cap = Capacitor(ref, _parse_value(val))
            c.add(cap)
            c.connect(cap.ports[0], _net_of(n1, nets))
            c.connect(cap.ports[1], _net_of(n2, nets))

        # elif prefix == "L":  # aguardando componente Inductor no core
        #     pass

        elif prefix == "V":
            _, nplus, nminus, *rest = t
            if not rest:
                continue

            if rest[0].upper() == "DC" and len(rest) >= 2:
                val = rest[1]
                vdc = Vdc(ref, _parse_value(val))
                c.add(vdc)
                c.connect(vdc.ports[0], _net_of(nplus, nets))
                c.connect(vdc.ports[1], _net_of(nminus, nets))

            elif rest[0].upper() == "AC" and len(rest) >= 2:
                mag = float(rest[1])
                phase = float(rest[2]) if len(rest) >= 3 else 0.0
                vac = Vac(ref, "", ac_mag=mag, ac_phase=phase)
                c.add(vac)
                c.connect(vac.ports[0], _net_of(nplus, nets))
                c.connect(vac.ports[1], _net_of(nminus, nets))

            else:
                val = rest[0]
                vdc = Vdc(ref, _parse_value(val))
                c.add(vdc)
                c.connect(vdc.ports[0], _net_of(nplus, nets))
                c.connect(vdc.ports[1], _net_of(nminus, nets))
        else:
            # Ignora outros dispositivos por enquanto (I, D, X, etc.)
            continue

    return c


def from_ltspice_file(path: str | Path) -> Circuit:
    """
    Lê arquivo de netlist SPICE (gerado pelo LTspice) e retorna Circuit.
    Observação: .ASC (schematic) não é um netlist — exporte via View→SPICE Netlist.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    first = text.splitlines()[0].strip() if text else ""
    title = first[1:].strip() if first.startswith("*") else p.stem
    return from_spice_netlist(text, title=title)
