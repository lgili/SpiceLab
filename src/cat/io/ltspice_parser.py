from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.circuit import Circuit
from ..core.components import (
    VPWL,
    VSIN,
    Capacitor,
    Component,
    Iac,
    Idc,
    Inductor,
    Ipulse,
    Ipwl,
    Isin,
    Resistor,
    Vac,
    Vdc,
    Vpulse,
)
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


def _collect_params_and_includes(text: str, base_dir: Path) -> tuple[str, dict[str, str]]:
    params: dict[str, str] = {}
    lines_out: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(".param"):
            try:
                body = s.split(None, 1)[1]
                for part in body.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k.strip()] = v.strip()
            except Exception:
                pass
            continue
        if low.startswith(".include"):
            try:
                inc = s.split(None, 1)[1].strip().strip('"')
                p = (base_dir / inc).resolve()
                if p.exists():
                    sub = p.read_text(encoding="utf-8", errors="ignore")
                    sub_expanded, sub_params = _collect_params_and_includes(sub, p.parent)
                    params.update(sub_params)
                    lines_out.extend(sub_expanded.splitlines())
                    continue
            except Exception:
                pass
            continue
        lines_out.append(raw)
    expanded = "\n".join(lines_out)
    for k, v in params.items():
        expanded = expanded.replace("{" + k + "}", v)
    return expanded, params


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

        elif prefix == "L":
            # Lref n1 n2 value
            if len(t) < 4:
                continue
            _, n1, n2, val = t[:4]
            ind = Inductor(ref, _parse_value(val))
            c.add(ind)
            c.connect(ind.ports[0], _net_of(n1, nets))
            c.connect(ind.ports[1], _net_of(n2, nets))

        elif prefix == "V":
            _, nplus, nminus, *rest = t
            if not rest:
                continue
            joined = " ".join(rest)
            up = joined.upper()
            if up.startswith("PULSE("):
                inside = joined[joined.find("(") + 1 : joined.rfind(")")]
                parts = [p for p in inside.replace(",", " ").split() if p]
                if len(parts) >= 7:
                    v1, v2, td, tr, tf, pw, per = parts[:7]
                    vp = Vpulse(ref, v1, v2, td, tr, tf, pw, per)
                    c.add(vp)
                    c.connect(vp.ports[0], _net_of(nplus, nets))
                    c.connect(vp.ports[1], _net_of(nminus, nets))
                    continue
            if up.startswith("SIN("):
                inside = joined[joined.find("(") + 1 : joined.rfind(")")]
                vs = VSIN(inside)
                # replace auto-ref with parsed ref
                vs.ref = ref
                c.add(vs)
                c.connect(vs.ports[0], _net_of(nplus, nets))
                c.connect(vs.ports[1], _net_of(nminus, nets))
                continue
            if up.startswith("PWL("):
                inside = joined[joined.find("(") + 1 : joined.rfind(")")]
                vpwl = VPWL(inside)
                vpwl.ref = ref
                c.add(vpwl)
                c.connect(vpwl.ports[0], _net_of(nplus, nets))
                c.connect(vpwl.ports[1], _net_of(nminus, nets))
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
        elif prefix == "I":
            # Corrente: Iref n+ n- DC <val> | AC <mag> [phase] | <val>
            _, nplus, nminus, *rest = t
            if not rest:
                continue
            joined = " ".join(rest)
            up2 = joined.upper()
            if up2.startswith("PULSE("):
                inside = joined[joined.find("(") + 1 : joined.rfind(")")]
                parts = [p for p in inside.replace(",", " ").split() if p]
                if len(parts) >= 7:
                    i1, i2, td, tr, tf, pw, per = parts[:7]
                    ip = Ipulse(ref, i1, i2, td, tr, tf, pw, per)
                    c.add(ip)
                    c.connect(ip.ports[0], _net_of(nplus, nets))
                    c.connect(ip.ports[1], _net_of(nminus, nets))
                    continue
            if up2.startswith("SIN("):
                inside = joined[joined.find("(") + 1 : joined.rfind(")")]
                isrc = Isin(ref, inside)
                c.add(isrc)
                c.connect(isrc.ports[0], _net_of(nplus, nets))
                c.connect(isrc.ports[1], _net_of(nminus, nets))
                continue
            if up2.startswith("PWL("):
                inside = joined[joined.find("(") + 1 : joined.rfind(")")]
                ipwl = Ipwl(ref, inside)
                c.add(ipwl)
                c.connect(ipwl.ports[0], _net_of(nplus, nets))
                c.connect(ipwl.ports[1], _net_of(nminus, nets))
                continue
            if rest[0].upper() == "DC" and len(rest) >= 2:
                val = rest[1]
                src_obj: Component = Idc(ref, _parse_value(val))
                c.add(src_obj)
                c.connect(src_obj.ports[0], _net_of(nplus, nets))
                c.connect(src_obj.ports[1], _net_of(nminus, nets))
            elif rest[0].upper() == "AC" and len(rest) >= 2:
                mag = float(rest[1])
                phase = float(rest[2]) if len(rest) >= 3 else 0.0
                src_obj = Iac(ref, "", ac_mag=mag, ac_phase=phase)
                c.add(src_obj)
                c.connect(src_obj.ports[0], _net_of(nplus, nets))
                c.connect(src_obj.ports[1], _net_of(nminus, nets))
            else:
                val = rest[0]
                src_obj = Idc(ref, _parse_value(val))
                c.add(src_obj)
                c.connect(src_obj.ports[0], _net_of(nplus, nets))
                c.connect(src_obj.ports[1], _net_of(nminus, nets))
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
    expanded, _ = _collect_params_and_includes(text, p.parent)
    return from_spice_netlist(expanded, title=title)


def ltspice_to_circuit(path: str | Path) -> Circuit:
    """Convenience wrapper that expands .include/.param and parses to Circuit."""
    return from_ltspice_file(path)
