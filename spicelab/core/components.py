from __future__ import annotations

from collections.abc import Callable

from .net import Port, PortRole

# Tipo do callback usado para mapear Port -> nome de nó no netlist
NetOf = Callable[[Port], str]


# --------------------------------------------------------------------------------------
# Base Component
# --------------------------------------------------------------------------------------
class Component:
    """Classe base de um componente de 2 terminais (ou fonte) no CAT."""

    ref: str
    value: str | float

    # As subclasses devem atribuir _ports no __init__
    _ports: tuple[Port, ...] = ()

    def __init__(self, ref: str, value: str | float = "") -> None:
        self.ref = ref
        self.value = value
        self._value_si_cache: float | None = None

    @property
    def value_si(self) -> float | None:
        """Return numeric SI value if parseable else None (lazy)."""
        if self._value_si_cache is not None:
            return self._value_si_cache
        from ..utils.units import to_float

        try:
            if isinstance(self.value, int | float):
                self._value_si_cache = float(self.value)
            elif isinstance(self.value, str) and self.value.strip():
                self._value_si_cache = to_float(self.value)
        except Exception:
            self._value_si_cache = None
        return self._value_si_cache

    @property
    def ports(self) -> tuple[Port, ...]:
        return self._ports

    def spice_card(self, net_of: NetOf) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover (depuração)
        return f"<{type(self).__name__} {self.ref} value={self.value!r}>"

    def __hash__(self) -> int:
        # Hash por identidade (robusto para objetos mutáveis)
        return id(self)


# --------------------------------------------------------------------------------------
# Componentes passivos
# --------------------------------------------------------------------------------------
class Resistor(Component):
    """Resistor de 2 terminais; portas: a (positivo), b (negativo)."""

    def __init__(self, ref: str, value: str | float = "") -> None:
        super().__init__(ref=ref, value=value)
        self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "b", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        a, b = self.ports
        return f"R{self.ref} {net_of(a)} {net_of(b)} {self.value}"


class Capacitor(Component):
    """Capacitor de 2 terminais; portas: a (positivo), b (negativo)."""

    def __init__(self, ref: str, value: str | float = "") -> None:
        super().__init__(ref=ref, value=value)
        self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "b", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        a, b = self.ports
        return f"C{self.ref} {net_of(a)} {net_of(b)} {self.value}"


# (Opcional) Se quiser adicionar Indutor no futuro:
# class Inductor(Component):
#     def __init__(self, ref: str, value: str | float = "") -> None:
#         super().__init__(ref=ref, value=value)
#         self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "b", PortRole.NEGATIVE))
#
#     def spice_card(self, net_of: NetOf) -> str:
#         a, b = self.ports
#         return f"L{self.ref} {net_of(a)} {net_of(b)} {self.value}"


# --------------------------------------------------------------------------------------
# Fontes
# --------------------------------------------------------------------------------------
class Vdc(Component):
    """Fonte de tensão DC; portas: p (positivo), n (negativo)."""

    def __init__(self, ref: str, value: str | float = "") -> None:
        super().__init__(ref=ref, value=value)
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        # Para DC, escrevemos o valor diretamente
        return f"V{self.ref} {net_of(p)} {net_of(n)} {self.value}"


class Vac(Component):
    """Fonte AC (small-signal) para .AC; portas: p (positivo), n (negativo).

    value: opcional, ignorado na carta SPICE (pode servir de rótulo).
    ac_mag: magnitude AC (tipicamente 1.0 V).
    ac_phase: fase em graus (opcional).
    """

    def __init__(
        self,
        ref: str,
        value: str | float = "",
        ac_mag: float = 1.0,
        ac_phase: float = 0.0,
    ) -> None:
        super().__init__(ref=ref, value=value)
        self.ac_mag = ac_mag
        self.ac_phase = ac_phase
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        if self.ac_phase:
            return f"V{self.ref} {net_of(p)} {net_of(n)} AC {self.ac_mag} {self.ac_phase}"
        return f"V{self.ref} {net_of(p)} {net_of(n)} AC {self.ac_mag}"


class Vpulse(Component):
    """Fonte de tensão PULSE(V1 V2 TD TR TF PW PER)."""

    def __init__(
        self,
        ref: str,
        v1: str | float,
        v2: str | float,
        td: str | float,
        tr: str | float,
        tf: str | float,
        pw: str | float,
        per: str | float,
    ) -> None:
        super().__init__(ref=ref, value="")
        self.v1, self.v2, self.td, self.tr, self.tf, self.pw, self.per = (
            v1,
            v2,
            td,
            tr,
            tf,
            pw,
            per,
        )
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return (
            f"V{self.ref} {net_of(p)} {net_of(n)} "
            f"PULSE({self.v1} {self.v2} {self.td} {self.tr} {self.tf} {self.pw} {self.per})"
        )


# --------------------------------------------------------------------------------------
# Helpers de criação com auto-ref (convenientes para notebooks/tests)
# --------------------------------------------------------------------------------------
_counter: dict[str, int] = {"R": 0, "C": 0, "V": 0, "L": 0, "I": 0}


def _next(prefix: str) -> str:
    _counter[prefix] = _counter.get(prefix, 0) + 1
    return str(_counter[prefix])


def R(value: str | float) -> Resistor:
    return Resistor(ref=_next("R"), value=value)


def C(value: str | float) -> Capacitor:
    return Capacitor(ref=_next("C"), value=value)


def V(value: str | float) -> Vdc:
    return Vdc(ref=_next("V"), value=value)


def VA(ac_mag: float = 1.0, ac_phase: float = 0.0, label: str | float = "") -> Vac:
    # label é apenas informativo; não aparece no card
    return Vac(ref=_next("V"), value=str(label), ac_mag=ac_mag, ac_phase=ac_phase)


def VP(
    v1: str | float,
    v2: str | float,
    td: str | float,
    tr: str | float,
    tf: str | float,
    pw: str | float,
    per: str | float,
) -> Vpulse:
    return Vpulse(ref=_next("V"), v1=v1, v2=v2, td=td, tr=tr, tf=tf, pw=pw, per=per)


# --------------------------------------------------------------------------------------
# Indutor e Fontes de Corrente
# --------------------------------------------------------------------------------------


class Inductor(Component):
    """Indutor de 2 terminais; portas: a (positivo), b (negativo)."""

    def __init__(self, ref: str, value: str | float = "") -> None:
        super().__init__(ref=ref, value=value)
        self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "b", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        a, b = self.ports
        return f"L{self.ref} {net_of(a)} {net_of(b)} {self.value}"


class Idc(Component):
    """Fonte de corrente DC; portas: p (de onde sai a corrente), n (retorno)."""

    def __init__(self, ref: str, value: str | float = "") -> None:
        super().__init__(ref=ref, value=value)
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"I{self.ref} {net_of(p)} {net_of(n)} {self.value}"


class Iac(Component):
    """Fonte de corrente AC (small-signal). value é apenas label, AC usa ac_mag/ac_phase."""

    def __init__(
        self, ref: str, value: str | float = "", ac_mag: float = 1.0, ac_phase: float = 0.0
    ) -> None:
        super().__init__(ref=ref, value=value)
        self.ac_mag = ac_mag
        self.ac_phase = ac_phase
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        if self.ac_phase:
            return f"I{self.ref} {net_of(p)} {net_of(n)} AC {self.ac_mag} {self.ac_phase}"
        return f"I{self.ref} {net_of(p)} {net_of(n)} AC {self.ac_mag}"


class Ipulse(Component):
    """Fonte de corrente PULSE(I1 I2 TD TR TF PW PER)."""

    def __init__(
        self,
        ref: str,
        i1: str | float,
        i2: str | float,
        td: str | float,
        tr: str | float,
        tf: str | float,
        pw: str | float,
        per: str | float,
    ) -> None:
        super().__init__(ref=ref, value="")
        self.i1, self.i2, self.td, self.tr, self.tf, self.pw, self.per = (
            i1,
            i2,
            td,
            tr,
            tf,
            pw,
            per,
        )
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return (
            f"I{self.ref} {net_of(p)} {net_of(n)} "
            f"PULSE({self.i1} {self.i2} {self.td} {self.tr} {self.tf} {self.pw} {self.per})"
        )


class Vsin(Component):
    """Fonte de tensão seno: SIN(args_raw). Mantém argumentos como string."""

    def __init__(self, ref: str, args_raw: str) -> None:
        super().__init__(ref=ref, value="")
        self.args_raw = args_raw
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"V{self.ref} {net_of(p)} {net_of(n)} SIN({self.args_raw})"


class Isin(Component):
    """Fonte de corrente seno: SIN(args_raw). Mantém argumentos como string."""

    def __init__(self, ref: str, args_raw: str) -> None:
        super().__init__(ref=ref, value="")
        self.args_raw = args_raw
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"I{self.ref} {net_of(p)} {net_of(n)} SIN({self.args_raw})"


class Vpwl(Component):
    """Fonte de tensão PWL(args_raw)."""

    def __init__(self, ref: str, args_raw: str) -> None:
        super().__init__(ref=ref, value="")
        self.args_raw = args_raw
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"V{self.ref} {net_of(p)} {net_of(n)} PWL({self.args_raw})"


class Ipwl(Component):
    """Fonte de corrente PWL(args_raw)."""

    def __init__(self, ref: str, args_raw: str) -> None:
        super().__init__(ref=ref, value="")
        self.args_raw = args_raw
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"I{self.ref} {net_of(p)} {net_of(n)} PWL({self.args_raw})"


# --------------------------------------------------------------------------------------
# Fontes controladas (E/G/F/H) e Diodo
# --------------------------------------------------------------------------------------


class VCVS(Component):
    """E: VCVS — Fonte de tensão controlada por tensão.

    Portas: p, n, cp, cn (saída e nós de controle). Valor = ganho (av).
    """

    def __init__(self, ref: str, gain: str | float) -> None:
        super().__init__(ref=ref, value=gain)
        self._ports = (
            Port(self, "p", PortRole.POSITIVE),
            Port(self, "n", PortRole.NEGATIVE),
            Port(self, "cp", PortRole.POSITIVE),
            Port(self, "cn", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p, n, cp, cn = self.ports
        return f"E{self.ref} {net_of(p)} {net_of(n)} {net_of(cp)} {net_of(cn)} {self.value}"


class VCCS(Component):
    """G: VCCS — Fonte de corrente controlada por tensão (transcondutância em S)."""

    def __init__(self, ref: str, gm: str | float) -> None:
        super().__init__(ref=ref, value=gm)
        self._ports = (
            Port(self, "p", PortRole.POSITIVE),
            Port(self, "n", PortRole.NEGATIVE),
            Port(self, "cp", PortRole.POSITIVE),
            Port(self, "cn", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p, n, cp, cn = self.ports
        return f"G{self.ref} {net_of(p)} {net_of(n)} {net_of(cp)} {net_of(cn)} {self.value}"


class CCCS(Component):
    """F: CCCS — Fonte de corrente controlada por corrente (ganho de corrente).

    Requer a referência de uma fonte de tensão controladora (nome SPICE, ex.: V1).
    """

    def __init__(self, ref: str, ctrl_vsrc: str, gain: str | float) -> None:
        super().__init__(ref=ref, value=gain)
        self.ctrl_vsrc = ctrl_vsrc
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"F{self.ref} {net_of(p)} {net_of(n)} {self.ctrl_vsrc} {self.value}"


class CCVS(Component):
    """H: CCVS — Fonte de tensão controlada por corrente (transresistência em ohms)."""

    def __init__(self, ref: str, ctrl_vsrc: str, r: str | float) -> None:
        super().__init__(ref=ref, value=r)
        self.ctrl_vsrc = ctrl_vsrc
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"H{self.ref} {net_of(p)} {net_of(n)} {self.ctrl_vsrc} {self.value}"


class Diode(Component):
    """D: Diodo mínimo — valor = nome do .model (string)."""

    def __init__(self, ref: str, model: str) -> None:
        super().__init__(ref=ref, value=model)
        self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "c", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        a, c = self.ports
        return f"D{self.ref} {net_of(a)} {net_of(c)} {self.value}"


class VSwitch(Component):
    """Voltage-controlled switch (S): Sref p n cp cn model.

    Requires a separate .model <model> VSWITCH(...) directive added to the Circuit.
    """

    def __init__(self, ref: str, model: str) -> None:
        super().__init__(ref=ref, value=model)
        self._ports = (
            Port(self, "p", PortRole.POSITIVE),
            Port(self, "n", PortRole.NEGATIVE),
            Port(self, "cp", PortRole.POSITIVE),
            Port(self, "cn", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p, n, cp, cn = self.ports
        return f"S{self.ref} {net_of(p)} {net_of(n)} {net_of(cp)} {net_of(cn)} {self.value}"


class ISwitch(Component):
    """Current-controlled switch (W): Wref p n Vsrc model (LTspice syntax).

    Requires a .model <model> ISWITCH(...) directive.
    """

    def __init__(self, ref: str, ctrl_vsrc: str, model: str) -> None:
        super().__init__(ref=ref, value=model)
        self.ctrl_vsrc = ctrl_vsrc
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"W{self.ref} {net_of(p)} {net_of(n)} {self.ctrl_vsrc} {self.value}"


# ==========================
# Tipadas (SIN / PWL)
# ==========================


def _fmt(x: str | float) -> str:
    return str(x)


class VsinT(Component):
    """Fonte de tensão seno tipada: SIN(VO VA FREQ [TD [THETA [PHASE]]])."""

    def __init__(
        self,
        ref: str,
        vo: str | float,
        va: str | float,
        freq: str | float,
        td: str | float = 0,
        theta: str | float = 0,
        phase: str | float = 0,
    ) -> None:
        super().__init__(ref=ref, value="")
        self.vo, self.va, self.freq = vo, va, freq
        self.td, self.theta, self.phase = td, theta, phase
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        args = [_fmt(self.vo), _fmt(self.va), _fmt(self.freq)]
        if self.td or self.theta or self.phase:
            args.append(_fmt(self.td))
        if self.theta or self.phase:
            args.append(_fmt(self.theta))
        if self.phase:
            args.append(_fmt(self.phase))
        return f"V{self.ref} {net_of(p)} {net_of(n)} SIN({' '.join(args)})"


class IsinT(Component):
    """Fonte de corrente seno tipada: SIN(IO IA FREQ [TD [THETA [PHASE]]])."""

    def __init__(
        self,
        ref: str,
        io: str | float,
        ia: str | float,
        freq: str | float,
        td: str | float = 0,
        theta: str | float = 0,
        phase: str | float = 0,
    ) -> None:
        super().__init__(ref=ref, value="")
        self.io, self.ia, self.freq = io, ia, freq
        self.td, self.theta, self.phase = td, theta, phase
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        args = [_fmt(self.io), _fmt(self.ia), _fmt(self.freq)]
        if self.td or self.theta or self.phase:
            args.append(_fmt(self.td))
        if self.theta or self.phase:
            args.append(_fmt(self.theta))
        if self.phase:
            args.append(_fmt(self.phase))
        return f"I{self.ref} {net_of(p)} {net_of(n)} SIN({' '.join(args)})"


class VpwlT(Component):
    """Fonte de tensão PWL tipada: PWL(t1 v1 t2 v2 ...)."""

    def __init__(self, ref: str, points: list[tuple[str | float, str | float]]) -> None:
        super().__init__(ref=ref, value="")
        self.points = points
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        flat: list[str] = []
        for t, v in self.points:
            flat.append(_fmt(t))
            flat.append(_fmt(v))
        return f"V{self.ref} {net_of(p)} {net_of(n)} PWL({' '.join(flat)})"


class IpwlT(Component):
    """Fonte de corrente PWL tipada: PWL(t1 i1 t2 i2 ...)."""

    def __init__(self, ref: str, points: list[tuple[str | float, str | float]]) -> None:
        super().__init__(ref=ref, value="")
        self.points = points
        self._ports = (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        flat: list[str] = []
        for t, i in self.points:
            flat.append(_fmt(t))
            flat.append(_fmt(i))
        return f"I{self.ref} {net_of(p)} {net_of(n)} PWL({' '.join(flat)})"


def L(value: str | float) -> Inductor:
    return Inductor(ref=_next("L"), value=value)


def I(value: str | float) -> Idc:  # noqa: E743 - single-letter helper kept for API symmetry
    return Idc(ref=_next("I"), value=value)


def IA(ac_mag: float = 1.0, ac_phase: float = 0.0, label: str | float = "") -> Iac:
    return Iac(ref=_next("I"), value=str(label), ac_mag=ac_mag, ac_phase=ac_phase)


def IP(
    i1: str | float,
    i2: str | float,
    td: str | float,
    tr: str | float,
    tf: str | float,
    pw: str | float,
    per: str | float,
) -> Ipulse:
    return Ipulse(ref=_next("I"), i1=i1, i2=i2, td=td, tr=tr, tf=tf, pw=pw, per=per)


def VSIN(args_raw: str) -> Vsin:
    return Vsin(ref=_next("V"), args_raw=args_raw)


def ISIN(args_raw: str) -> Isin:
    return Isin(ref=_next("I"), args_raw=args_raw)


def VPWL(args_raw: str) -> Vpwl:
    return Vpwl(ref=_next("V"), args_raw=args_raw)


def IPWL(args_raw: str) -> Ipwl:
    return Ipwl(ref=_next("I"), args_raw=args_raw)


def VSIN_T(
    vo: str | float,
    va: str | float,
    freq: str | float,
    td: str | float = 0,
    theta: str | float = 0,
    phase: str | float = 0,
) -> VsinT:
    return VsinT(ref=_next("V"), vo=vo, va=va, freq=freq, td=td, theta=theta, phase=phase)


def ISIN_T(
    io: str | float,
    ia: str | float,
    freq: str | float,
    td: str | float = 0,
    theta: str | float = 0,
    phase: str | float = 0,
) -> IsinT:
    return IsinT(ref=_next("I"), io=io, ia=ia, freq=freq, td=td, theta=theta, phase=phase)


def VPWL_T(points: list[tuple[str | float, str | float]]) -> VpwlT:
    return VpwlT(ref=_next("V"), points=points)


def IPWL_T(points: list[tuple[str | float, str | float]]) -> IpwlT:
    return IpwlT(ref=_next("I"), points=points)


# Helpers para controladas e diodo
def E(gain: str | float) -> VCVS:
    return VCVS(ref=_next("E"), gain=gain)


def G(gm: str | float) -> VCCS:
    return VCCS(ref=_next("G"), gm=gm)


def F(ctrl_vsrc: str, gain: str | float) -> CCCS:
    return CCCS(ref=_next("F"), ctrl_vsrc=ctrl_vsrc, gain=gain)


def H(ctrl_vsrc: str, r: str | float) -> CCVS:
    return CCVS(ref=_next("H"), ctrl_vsrc=ctrl_vsrc, r=r)


def D(model: str) -> Diode:
    return Diode(ref=_next("D"), model=model)


def SW(model: str) -> VSwitch:
    return VSwitch(ref=_next("S"), model=model)


def SWI(ctrl_vsrc: str, model: str) -> ISwitch:
    return ISwitch(ref=_next("W"), ctrl_vsrc=ctrl_vsrc, model=model)


# --------------------------------------------------------------------------------------
# Op-amp ideal (modelado como VCVS de alto ganho)
# --------------------------------------------------------------------------------------


class OpAmpIdeal(Component):
    """Op-amp ideal de 3 pinos (inp, inn, out) modelado por VCVS de alto ganho.

    Carta: E<ref> out 0 inp inn <gain>
    """

    def __init__(self, ref: str, gain: str | float = 1e6) -> None:
        super().__init__(ref=ref, value=str(gain))
        self._ports = (
            Port(self, "inp", PortRole.POSITIVE),
            Port(self, "inn", PortRole.NEGATIVE),
            Port(self, "out", PortRole.POSITIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        inp, inn, out = self.ports
        return f"E{self.ref} {net_of(out)} 0 {net_of(inp)} {net_of(inn)} {self.value}"


def OA(gain: str | float = 1e6) -> OpAmpIdeal:
    """Helper para op-amp ideal. Usa o prefixo 'E' para evitar colisão de refs."""
    return OpAmpIdeal(ref=_next("E"), gain=gain)


# --------------------------------------------------------------------------------------
# Analog multiplexer 1-to-8
# --------------------------------------------------------------------------------------


class AnalogMux8(Component):
    """Analog 1-to-8 multiplexer.

    Ports:
      - in: input node
      - out0..out7: outputs
      - en0..en7 (optional): active-high enable ports controlling each channel

    Modeling choices:
      - If `sel` is provided (int 0..7) the selected channel receives a series
        resistor `r_series` between `in` and `outN`. Other channels receive a
        large off-resistance (`off_resistance`).
      - If `enable_ports=True`, the component exposes 8 enable ports and the
        netlist will include voltage-controlled switches (S...) driven by the
        corresponding enable port. The switch model name defaults to
        `SW_<ref>` and a recommended `.model` should be added by the user.
    """

    def __init__(
        self,
        ref: str,
        r_series: str | float = 100,
        sel: int | None = None,
        enable_ports: bool = False,
        off_resistance: str | float = "1G",
        sw_model: str | None = None,
        emit_model: bool = False,
        as_subckt: bool = False,
    ) -> None:
        super().__init__(ref=ref, value="")
        ports: list[Port] = [Port(self, "in", PortRole.NODE)]
        for i in range(8):
            ports.append(Port(self, f"out{i}", PortRole.NODE))
        if enable_ports:
            for i in range(8):
                ports.append(Port(self, f"en{i}", PortRole.NODE))
        self._ports = tuple(ports)
        self.r_series = r_series
        self.sel = sel
        self.enable_ports = enable_ports
        self.off_resistance = off_resistance
        # If user doesn't provide sw_model, default to SW_<ref>
        self.sw_model = sw_model or f"SW_{self.ref}"
        self.emit_model = emit_model
        self.as_subckt = as_subckt
        # simple validation
        if self.sel is not None and not (0 <= self.sel < 8):
            raise ValueError("sel must be in 0..7")
        # if both sel and enable_ports are provided, we'll honor enable_ports

    def spice_card(self, net_of: NetOf) -> str:
        # indexing: 0 -> in, 1..8 -> out0..out7, optional enables follow
        in_port = self._ports[0]
        outs = [self._ports[1 + i] for i in range(8)]
        en_ports: list[Port] = []
        if self.enable_ports:
            en_ports = [self._ports[1 + 8 + i] for i in range(8)]

        lines: list[str] = []
        models: list[str] = []

        # If enable_ports, emit S switches + series resistor; otherwise emit
        # series resistors with selected one = r_series and others = off_resistance
        if self.enable_ports:
            # Use mid nodes to place resistor after switch to allow correct control
            for i, out in enumerate(outs):
                mid = f"{self.ref}_mid{i}"
                # S element: Sref p n cp cn model
                # Control is V(cp) - V(cn). Drive cp with enable, cn tied to ground (active-high)
                p_name = net_of(in_port)
                n_name = mid
                cp = net_of(en_ports[i])
                cn = "0"
                lines.append(f"S{self.ref}_{i} {p_name} {n_name} {cp} {cn} {self.sw_model}")
                lines.append(f"R{self.ref}_{i} {n_name} {net_of(out)} {self.r_series}")
            if self.emit_model:
                # simple SW model; these params are conservative defaults
                models.append(f".model {self.sw_model} SW(RON=1 ROFF=1e9 Vt=0.5)")
        else:
            for i, out in enumerate(outs):
                if self.sel is not None:
                    if i == self.sel:
                        lines.append(
                            f"R{self.ref}_{i} {net_of(in_port)} {net_of(out)} {self.r_series}"
                        )
                    else:
                        lines.append(
                            f"R{self.ref}_{i} {net_of(in_port)} {net_of(out)} {self.off_resistance}"
                        )
                else:
                    # No selection and no enables: all paths high-Z
                    lines.append(
                        f"R{self.ref}_{i} {net_of(in_port)} {net_of(out)} {self.off_resistance}"
                    )

        # Add a comment summarizing configuration
        header = (
            f"* AnalogMux8 {self.ref} r_series={self.r_series} sel={self.sel} "
            f"enable_ports={self.enable_ports} subckt={self.as_subckt}"
        )
        # If as_subckt is requested, wrap ports into .subckt ... .ends
        if self.as_subckt:
            # collect port names in order: in, out0..out7, en0..en7?
            port_names = ["in"] + [f"out{i}" for i in range(8)]
            if self.enable_ports:
                port_names += [f"en{i}" for i in range(8)]
            subckt_header = f".subckt M{self.ref} {' '.join(port_names)}"
            body = "\n".join(lines)
            parts = [header, subckt_header, body, ".ends"]
            if models:
                parts += models
            return "\n".join(parts)

        parts = [header] + lines
        if models:
            parts += models
        return "\n".join(parts)


def MUX8(r_series: str | float = 100, sel: int | None = None) -> AnalogMux8:
    """Convenience factory: create AnalogMux8 with auto-ref."""
    return AnalogMux8(ref=_next("M"), r_series=r_series, sel=sel)
