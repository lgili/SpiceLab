from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.circuit import Circuit
from ..core.components import (
    CCCS,
    CCVS,
    VCCS,
    VCVS,
    VPWL,
    VSIN,
    Capacitor,
    Component,
    Diode,
    Iac,
    Idc,
    Inductor,
    Ipulse,
    Ipwl,
    Isin,
    ISwitch,
    Resistor,
    Vac,
    Vdc,
    Vpulse,
    VSwitch,
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


def _parse_subckts(text: str) -> tuple[str, dict[str, tuple[list[str], list[str]]]]:
    """Extract .SUBCKT blocks and return (text_without_blocks, subckt_map).

    subckt_map[name] = (pins, body_lines)
    """
    lines = text.splitlines()
    out: list[str] = []
    subckts: dict[str, tuple[list[str], list[str]]] = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        low = s.lower()
        if low.startswith(".subckt"):
            parts = s.split()
            if len(parts) >= 3:
                name = parts[1]
                pins = parts[2:]
                body: list[str] = []
                i += 1
                while i < len(lines):
                    ss = lines[i].strip()
                    if ss.lower().startswith(".ends"):
                        break
                    body.append(lines[i])
                    i += 1
                subckts[name] = (pins, body)
                # skip .ends
                while i < len(lines) and not lines[i].strip().lower().startswith(".ends"):
                    i += 1
                i += 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), subckts


def _expand_subckt_instance(card: str, pins: list[str], body: list[str]) -> list[str]:
    """Expand a single X instance line into flat device lines.

    card: e.g., "XU1 n1 n2 RC"
    pins: subckt formal pins
    body: lines inside subckt
    """
    t = card.split()
    inst_ref = t[0][1:]  # remove leading 'X'
    # Identify subckt name (last token) and split args vs params
    _subckt_name = t[-1]
    arg_tokens = t[1:-1]
    # Params: tokens containing '=' beyond the known positional pins
    pin_args = arg_tokens[: len(pins)]
    param_tokens = arg_tokens[len(pins) :]
    param_map: dict[str, str] = {}
    for tok in param_tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            param_map[k.strip()] = v.strip()
    pin_map: dict[str, str] = {}
    for i, p in enumerate(pins):
        if i < len(pin_args):
            pin_map[p] = pin_args[i]

    def _map_node(tok: str) -> str:
        if tok == "0":
            return tok
        if tok in pin_map:
            return pin_map[tok]
        return f"{inst_ref}_{tok}"

    def _rewrite_ref(tok: str) -> str:
        # Prefix ref with instance
        if not tok:
            return tok
        return f"{tok[0]}{inst_ref}_{tok[1:]}"

    def _node_count(prefix: str) -> int:
        return {"R": 2, "C": 2, "L": 2, "D": 2, "V": 2, "I": 2, "E": 4, "G": 4, "F": 2, "H": 2}.get(
            prefix, 0
        )

    out: list[str] = []
    for ln in body:
        s = ln.strip()
        if not s or s.startswith("*"):
            continue
        # Apply param substitution within the body line using {NAME} placeholders
        if param_map:
            for k, v in param_map.items():
                s = s.replace("{" + k + "}", v)
        parts = s.split()
        if not parts:
            continue
        dev = parts[0]
        pref = dev[0].upper()
        parts[0] = _rewrite_ref(dev)
        ncnt = _node_count(pref)
        # rewrite first ncnt node tokens
        for i in range(1, min(1 + ncnt, len(parts))):
            parts[i] = _map_node(parts[i])
        out.append(" ".join(parts))
    return out


def _expand_subckts_in_text(text: str) -> str:
    base, subckts = _parse_subckts(text)
    if not subckts:
        return text

    def expand_once(txt: str) -> tuple[str, bool]:
        changed = False
        out: list[str] = []
        for raw in txt.splitlines():
            s = raw.strip()
            if not s or s.startswith("*"):
                out.append(raw)
                continue
            t = s.split()
            if t[0][0].upper() == "X" and len(t) >= 2:
                # subckt name is the last token that does NOT contain '=' (params come after)
                subname = None
                for tok in reversed(t[1:]):
                    if "=" not in tok:
                        subname = tok
                        break
                if subname is None:
                    subname = t[-1]
                if subname in subckts:
                    pins, body = subckts[subname]
                    out.extend(_expand_subckt_instance(s, pins, body))
                    changed = True
                    continue
            out.append(raw)
        return "\n".join(out), changed

    cur = base
    # Iterate expansion to handle one level of nesting (or until no changes)
    for _ in range(3):  # small cap to prevent infinite loops
        cur, ch = expand_once(cur)
        if not ch:
            break
    return cur


def from_spice_netlist(text: str, *, title: str | None = None) -> Circuit:
    """
    Converte um netlist SPICE (LTspice exportado via View→SPICE Netlist) em Circuit.
    MVP: R, C, V (DC/AC). Linhas desconhecidas são ignoradas com segurança.
    """
    c = Circuit(title or "imported")
    nets: dict[str, Net] = {}

    for raw in text.splitlines():
        t = _tok(raw)
        if not t:
            continue
        if _is_directive(t):
            # Preserve .model directives into the Circuit
            if t[0].lower().startswith(".model"):
                # full raw line
                c.add_directive(raw)
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
            # Dispositivos adicionais
            if prefix == "S" and len(t) >= 6:
                # Sref p n cp cn model
                _, n1, n2, nc1, nc2, model = t[:6]
                s = VSwitch(ref, model)
                c.add(s)
                p, n, cp, cn = s.ports
                c.connect(p, _net_of(n1, nets))
                c.connect(n, _net_of(n2, nets))
                c.connect(cp, _net_of(nc1, nets))
                c.connect(cn, _net_of(nc2, nets))
                continue
            if prefix == "W" and len(t) >= 5:
                # Wref n+ n- Vsrc model
                _, n1, n2, vsrc, model = t[:5]
                w = ISwitch(ref, vsrc, model)
                c.add(w)
                p, n = w.ports
                c.connect(p, _net_of(n1, nets))
                c.connect(n, _net_of(n2, nets))
                continue
            if prefix == "E" and len(t) >= 6:
                # Eref n+ n- nc+ nc- gain
                _, n1, n2, nc1, nc2, gain = t[:6]
                e = VCVS(ref, gain)
                c.add(e)
                p, n, cp, cn = e.ports
                c.connect(p, _net_of(n1, nets))
                c.connect(n, _net_of(n2, nets))
                c.connect(cp, _net_of(nc1, nets))
                c.connect(cn, _net_of(nc2, nets))
                continue
            if prefix == "G" and len(t) >= 6:
                # Gref n+ n- nc+ nc- gm
                _, n1, n2, nc1, nc2, gm = t[:6]
                g = VCCS(ref, gm)
                c.add(g)
                p, n, cp, cn = g.ports
                c.connect(p, _net_of(n1, nets))
                c.connect(n, _net_of(n2, nets))
                c.connect(cp, _net_of(nc1, nets))
                c.connect(cn, _net_of(nc2, nets))
                continue
            if prefix == "F" and len(t) >= 5:
                # Fref n+ n- Vsrc gain
                _, n1, n2, vsrc, gain = t[:5]
                f = CCCS(ref, vsrc, gain)
                c.add(f)
                p, n = f.ports
                c.connect(p, _net_of(n1, nets))
                c.connect(n, _net_of(n2, nets))
                continue
            if prefix == "H" and len(t) >= 5:
                # Href n+ n- Vsrc r
                _, n1, n2, vsrc, rr = t[:5]
                h = CCVS(ref, vsrc, rr)
                c.add(h)
                p, n = h.ports
                c.connect(p, _net_of(n1, nets))
                c.connect(n, _net_of(n2, nets))
                continue
            if prefix == "D" and len(t) >= 4:
                # Dref a c model
                _, a, cc, model = t[:4]
                d = Diode(ref, model)
                c.add(d)
                c.connect(d.ports[0], _net_of(a, nets))
                c.connect(d.ports[1], _net_of(cc, nets))
                continue
            # Ignora outros dispositivos (X, etc.)
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
    # Flatten simple .SUBCKT instances (one-level nesting supported)
    expanded = _expand_subckts_in_text(expanded)
    return from_spice_netlist(expanded, title=title)


def ltspice_to_circuit(path: str | Path) -> Circuit:
    """Convenience wrapper that expands .include/.param and parses to Circuit."""
    return from_ltspice_file(path)
