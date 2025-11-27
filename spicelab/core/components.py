from __future__ import annotations

from collections.abc import Callable

from .net import Port, PortRole
from .parameter import ParameterRef

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

    # SPICE card cache (M2 performance optimization)
    _spice_card_cache: dict[tuple[str, ...], str] | None = None

    def __init__(self, ref: str, value: str | float = "") -> None:
        self.ref = ref
        self.value = value
        self._value_si_cache: float | None = None
        self._spice_card_cache = None

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
    """Resistor de 2 terminais; portas: a (positivo), b (negativo).

    Args:
        ref: Reference designator (e.g., "1" for R1)
        value: Legacy stringly-typed value (backward compat)
        resistance: Typed resistance value (float or ParameterRef)

    Use either `value` OR `resistance`, not both.
    """

    def __init__(
        self,
        ref: str,
        value: str | float = "",
        resistance: float | ParameterRef | None = None,
    ) -> None:
        super().__init__(ref=ref, value=value)
        self.resistance = resistance
        self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "b", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        a, b = self.ports
        # Use typed field if present, otherwise fall back to value
        val = str(self.resistance) if self.resistance is not None else self.value
        return f"R{self.ref} {net_of(a)} {net_of(b)} {val}"


class Capacitor(Component):
    """Capacitor de 2 terminais; portas: a (positivo), b (negativo).

    Args:
        ref: Reference designator (e.g., "1" for C1)
        value: Legacy stringly-typed value (backward compat)
        capacitance: Typed capacitance value (float or ParameterRef)

    Use either `value` OR `capacitance`, not both.
    """

    def __init__(
        self,
        ref: str,
        value: str | float = "",
        capacitance: float | ParameterRef | None = None,
    ) -> None:
        super().__init__(ref=ref, value=value)
        self.capacitance = capacitance
        self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "b", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        a, b = self.ports
        # Use typed field if present, otherwise fall back to value
        val = str(self.capacitance) if self.capacitance is not None else self.value
        return f"C{self.ref} {net_of(a)} {net_of(b)} {val}"


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
_counter: dict[str, int] = {
    "R": 0,
    "C": 0,
    "V": 0,
    "L": 0,
    "I": 0,
    "D": 0,
    "E": 0,
    "F": 0,
    "G": 0,
    "H": 0,
    "J": 0,
    "K": 0,
    "M": 0,
    "O": 0,
    "Q": 0,
    "S": 0,
    "T": 0,
    "U": 0,
    "W": 0,
    "X": 0,
    "B": 0,
}


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
    """Indutor de 2 terminais; portas: a (positivo), b (negativo).

    Args:
        ref: Reference designator (e.g., "1" for L1)
        value: Legacy stringly-typed value (backward compat)
        inductance: Typed inductance value (float or ParameterRef)

    Use either `value` OR `inductance`, not both.
    """

    def __init__(
        self,
        ref: str,
        value: str | float = "",
        inductance: float | ParameterRef | None = None,
    ) -> None:
        super().__init__(ref=ref, value=value)
        self.inductance = inductance
        self._ports = (Port(self, "a", PortRole.POSITIVE), Port(self, "b", PortRole.NEGATIVE))

    def spice_card(self, net_of: NetOf) -> str:
        a, b = self.ports
        # Use typed field if present, otherwise fall back to value
        val = str(self.inductance) if self.inductance is not None else self.value
        return f"L{self.ref} {net_of(a)} {net_of(b)} {val}"


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


# --------------------------------------------------------------------------------------
# JFET Transistor
# --------------------------------------------------------------------------------------


class JFET(Component):
    """Junction Field-Effect Transistor (N-channel or P-channel).

    SPICE card: J<ref> <drain> <gate> <source> <model>

    Args:
        ref: Reference designator (e.g., "1" for J1)
        model: SPICE model name (e.g., "2N5457", "J2N5459")

    Ports:
        d: drain
        g: gate
        s: source
    """

    def __init__(self, ref: str, model: str) -> None:
        super().__init__(ref=ref, value=model)
        self._ports = (
            Port(self, "d", PortRole.POSITIVE),
            Port(self, "g", PortRole.NODE),
            Port(self, "s", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        d, g, s = self.ports
        return f"J{self.ref} {net_of(d)} {net_of(g)} {net_of(s)} {self.value}"


def JF(model: str) -> JFET:
    """Helper factory for JFET with auto-ref."""
    return JFET(ref=_next("J"), model=model)


# --------------------------------------------------------------------------------------
# Zener Diode
# --------------------------------------------------------------------------------------


class ZenerDiode(Component):
    """Zener diode for voltage reference/regulation.

    SPICE card: D<ref> <anode> <cathode> <model>

    Same as regular Diode but semantically distinct for clarity.

    Args:
        ref: Reference designator
        model: Zener model name (e.g., "1N4733" for 5.1V Zener)

    Ports:
        a: anode
        c: cathode (connected to reference voltage in reverse bias)
    """

    def __init__(self, ref: str, model: str) -> None:
        super().__init__(ref=ref, value=model)
        self._ports = (
            Port(self, "a", PortRole.POSITIVE),
            Port(self, "c", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        a, c = self.ports
        return f"D{self.ref} {net_of(a)} {net_of(c)} {self.value}"


def DZ(model: str) -> ZenerDiode:
    """Helper factory for ZenerDiode with auto-ref."""
    return ZenerDiode(ref=_next("D"), model=model)


# --------------------------------------------------------------------------------------
# Mutual Inductance (Coupled Inductors)
# --------------------------------------------------------------------------------------


class MutualInductance(Component):
    """Mutual inductance coupling between two inductors.

    SPICE card: K<ref> <L1_ref> <L2_ref> <coupling>

    This component does not have physical ports - it references existing inductors.

    Args:
        ref: Reference designator (e.g., "1" for K1)
        l1: Reference to first inductor (e.g., "L1")
        l2: Reference to second inductor (e.g., "L2")
        coupling: Coupling coefficient (0 to 1, typically 0.95-0.999)

    Example:
        # Create two inductors and couple them
        l1 = Inductor("1", "10m")
        l2 = Inductor("2", "10m")
        k1 = MutualInductance("1", l1="L1", l2="L2", coupling=0.99)
    """

    def __init__(
        self,
        ref: str,
        l1: str,
        l2: str,
        coupling: float,
    ) -> None:
        if not 0 <= coupling <= 1:
            raise ValueError(f"Coupling coefficient must be 0-1, got {coupling}")
        super().__init__(ref=ref, value=str(coupling))
        self.l1 = l1
        self.l2 = l2
        self.coupling = coupling
        # No physical ports - this references other components
        self._ports = ()

    def spice_card(self, net_of: NetOf) -> str:
        return f"K{self.ref} {self.l1} {self.l2} {self.coupling}"


def MK(l1: str, l2: str, coupling: float) -> MutualInductance:
    """Helper factory for MutualInductance with auto-ref."""
    return MutualInductance(ref=_next("K"), l1=l1, l2=l2, coupling=coupling)


# --------------------------------------------------------------------------------------
# Transformer (Ideal)
# --------------------------------------------------------------------------------------


class Transformer(Component):
    """Ideal transformer using coupled inductors.

    SPICE implementation uses two inductors coupled via K element.
    The turns ratio determines the inductance ratio (L2/L1 = n^2).

    Args:
        ref: Reference designator (e.g., "1" for XFMR1)
        turns_ratio: Secondary/Primary turns ratio (n = Ns/Np)
        l_primary: Primary inductance value (default "1m")
        coupling: Coupling coefficient (default 0.9999 for ideal)

    Ports:
        p1: Primary positive
        p2: Primary negative
        s1: Secondary positive
        s2: Secondary negative

    Example:
        # Create 1:10 step-up transformer
        xfmr = Transformer("1", turns_ratio=10)

    Note:
        The spice_card method returns multiple lines (L1, L2, K statements).
    """

    def __init__(
        self,
        ref: str,
        turns_ratio: float,
        l_primary: str = "1m",
        coupling: float = 0.9999,
    ) -> None:
        if turns_ratio <= 0:
            raise ValueError(f"Turns ratio must be positive, got {turns_ratio}")
        if not 0 < coupling <= 1:
            raise ValueError(f"Coupling must be 0 < k <= 1, got {coupling}")
        super().__init__(ref=ref, value=str(turns_ratio))
        self.turns_ratio = turns_ratio
        self.l_primary = l_primary
        self.coupling = coupling
        self._ports = (
            Port(self, "p1", PortRole.POSITIVE),
            Port(self, "p2", PortRole.NEGATIVE),
            Port(self, "s1", PortRole.POSITIVE),
            Port(self, "s2", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p1, p2, s1, s2 = self.ports
        # L2 = L1 * n^2 for ideal transformer
        l_sec = f"{float(self.l_primary.rstrip('m')) * self.turns_ratio**2}m"
        lines = [
            f"L{self.ref}p {net_of(p1)} {net_of(p2)} {self.l_primary}",
            f"L{self.ref}s {net_of(s1)} {net_of(s2)} {l_sec}",
            f"K{self.ref} L{self.ref}p L{self.ref}s {self.coupling}",
        ]
        return "\n".join(lines)


def XFMR(
    turns_ratio: float,
    l_primary: str = "1m",
    coupling: float = 0.9999,
) -> Transformer:
    """Helper factory for Transformer with auto-ref."""
    return Transformer(
        ref=_next("K"), turns_ratio=turns_ratio, l_primary=l_primary, coupling=coupling
    )


# --------------------------------------------------------------------------------------
# Transmission Lines
# --------------------------------------------------------------------------------------


class TLine(Component):
    """Lossless transmission line (T element).

    SPICE card: T<ref> <p1+> <p1-> <p2+> <p2-> Z0=<z0> TD=<td>

    Args:
        ref: Reference designator
        z0: Characteristic impedance (ohms)
        td: Time delay (e.g., "1n" for 1ns)

    Ports:
        p1p, p1n: Port 1 positive and negative
        p2p, p2n: Port 2 positive and negative
    """

    def __init__(
        self,
        ref: str,
        z0: str | float,
        td: str | float,
    ) -> None:
        super().__init__(ref=ref, value="")
        self.z0 = z0
        self.td = td
        self._ports = (
            Port(self, "p1p", PortRole.POSITIVE),
            Port(self, "p1n", PortRole.NEGATIVE),
            Port(self, "p2p", PortRole.POSITIVE),
            Port(self, "p2n", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p1p, p1n, p2p, p2n = self.ports
        return (
            f"T{self.ref} {net_of(p1p)} {net_of(p1n)} {net_of(p2p)} {net_of(p2n)} "
            f"Z0={self.z0} TD={self.td}"
        )


def TLINE(z0: str | float, td: str | float) -> TLine:
    """Helper factory for TLine with auto-ref."""
    return TLine(ref=_next("T"), z0=z0, td=td)


class TLineLossy(Component):
    """Lossy transmission line (O element) - LTRA model.

    SPICE card: O<ref> <p1+> <p1-> <p2+> <p2-> <model>

    Requires a .model statement with LTRA parameters.

    Args:
        ref: Reference designator
        model: LTRA model name

    Ports:
        p1p, p1n: Port 1 positive and negative
        p2p, p2n: Port 2 positive and negative
    """

    def __init__(self, ref: str, model: str) -> None:
        super().__init__(ref=ref, value=model)
        self._ports = (
            Port(self, "p1p", PortRole.POSITIVE),
            Port(self, "p1n", PortRole.NEGATIVE),
            Port(self, "p2p", PortRole.POSITIVE),
            Port(self, "p2n", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p1p, p1n, p2p, p2n = self.ports
        return f"O{self.ref} {net_of(p1p)} {net_of(p1n)} {net_of(p2p)} {net_of(p2n)} {self.value}"


def OLINE(model: str) -> TLineLossy:
    """Helper factory for TLineLossy with auto-ref."""
    return TLineLossy(ref=_next("O"), model=model)


class TLineRC(Component):
    """Uniform distributed RC line (U element) - URC model.

    SPICE card: U<ref> <n1> <n2> <n3> <model> L=<len>

    Args:
        ref: Reference designator
        model: URC model name
        length: Line length parameter

    Ports:
        n1: Node 1
        n2: Node 2
        n3: Node 3 (typically ground reference)
    """

    def __init__(self, ref: str, model: str, length: str | float = 1) -> None:
        super().__init__(ref=ref, value=model)
        self.length = length
        self._ports = (
            Port(self, "n1", PortRole.NODE),
            Port(self, "n2", PortRole.NODE),
            Port(self, "n3", PortRole.NODE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        n1, n2, n3 = self.ports
        return f"U{self.ref} {net_of(n1)} {net_of(n2)} {net_of(n3)} {self.value} L={self.length}"


def ULINE(model: str, length: str | float = 1) -> TLineRC:
    """Helper factory for TLineRC with auto-ref."""
    return TLineRC(ref=_next("U"), model=model, length=length)


# --------------------------------------------------------------------------------------
# Behavioral Sources (B element)
# --------------------------------------------------------------------------------------


class BVoltage(Component):
    """Behavioral voltage source with arbitrary expression.

    SPICE card: B<ref> <p> <n> V=<expression>

    The expression is passed directly to the SPICE engine.

    Args:
        ref: Reference designator
        expr: Voltage expression (e.g., "V(in)*2", "IF(V(ctrl)>2.5, 5, 0)")

    Ports:
        p: positive terminal
        n: negative terminal

    Example:
        # Voltage doubler
        b1 = BVoltage("1", expr="V(in)*2")

        # Conditional output
        b2 = BVoltage("2", expr="IF(V(ctrl)>2.5, 5, 0)")
    """

    def __init__(self, ref: str, expr: str) -> None:
        super().__init__(ref=ref, value=expr)
        self.expr = expr
        self._ports = (
            Port(self, "p", PortRole.POSITIVE),
            Port(self, "n", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"B{self.ref} {net_of(p)} {net_of(n)} V={self.expr}"


def BV(expr: str) -> BVoltage:
    """Helper factory for BVoltage with auto-ref."""
    return BVoltage(ref=_next("B"), expr=expr)


class BCurrent(Component):
    """Behavioral current source with arbitrary expression.

    SPICE card: B<ref> <p> <n> I=<expression>

    The expression is passed directly to the SPICE engine.

    Args:
        ref: Reference designator
        expr: Current expression (e.g., "I(Vref)*10", "V(in)/1k")

    Ports:
        p: positive terminal (current flows out)
        n: negative terminal (current flows in)

    Example:
        # Current mirror with gain
        b1 = BCurrent("1", expr="I(Vref)*10")

        # Voltage-to-current conversion
        b2 = BCurrent("2", expr="V(in)/1000")
    """

    def __init__(self, ref: str, expr: str) -> None:
        super().__init__(ref=ref, value=expr)
        self.expr = expr
        self._ports = (
            Port(self, "p", PortRole.POSITIVE),
            Port(self, "n", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"B{self.ref} {net_of(p)} {net_of(n)} I={self.expr}"


def BI(expr: str) -> BCurrent:
    """Helper factory for BCurrent with auto-ref."""
    return BCurrent(ref=_next("B"), expr=expr)


# Aliases for expression-based controlled sources
# BVoltage with V(node) expressions acts as VCVS
# BCurrent with V(node) expressions acts as VCCS
VCVSExpr = BVoltage  # Voltage-Controlled Voltage Source with expression
VCCSExpr = BCurrent  # Voltage-Controlled Current Source with expression


def EExpr(expr: str) -> BVoltage:
    """Create VCVS with arbitrary expression (E-source behavior via B element).

    Example:
        # Op-amp like gain stage
        e1 = EExpr("V(inp, inn) * 1e6")
    """
    return BVoltage(ref=_next("B"), expr=expr)


def GExpr(expr: str) -> BCurrent:
    """Create VCCS with arbitrary expression (G-source behavior via B element).

    Example:
        # Transconductance amplifier
        g1 = GExpr("V(in) * 0.001")  # gm = 1mS
    """
    return BCurrent(ref=_next("B"), expr=expr)


# --------------------------------------------------------------------------------------
# Subcircuit Instance
# --------------------------------------------------------------------------------------


class SubcktInstance(Component):
    """Subcircuit instance (X element).

    SPICE card: X<ref> <node1> <node2> ... <subckt_name> [param=value ...]

    Args:
        ref: Reference designator
        subckt_name: Name of the subcircuit to instantiate
        nodes: List of node names to connect to subcircuit ports
        params: Optional dict of parameter overrides

    Example:
        # Instantiate an op-amp subcircuit
        x1 = SubcktInstance("1", "LM741", ["inp", "inn", "vcc", "vee", "out"])

        # With parameters
        x2 = SubcktInstance("2", "RES_VAR", ["a", "b"], params={"R": "1k"})
    """

    def __init__(
        self,
        ref: str,
        subckt_name: str,
        nodes: list[str],
        params: dict[str, str | float] | None = None,
    ) -> None:
        super().__init__(ref=ref, value=subckt_name)
        self.subckt_name = subckt_name
        self.nodes = nodes
        self.params = params or {}
        # Create ports dynamically based on nodes
        self._ports = tuple(Port(self, f"n{i}", PortRole.NODE) for i in range(len(nodes)))
        # Store node names for spice_card
        self._node_names = nodes

    def spice_card(self, net_of: NetOf) -> str:
        # Use provided node names directly (they represent circuit nodes)
        nodes_str = " ".join(self._node_names)
        params_str = ""
        if self.params:
            params_str = " " + " ".join(f"{k}={v}" for k, v in self.params.items())
        return f"X{self.ref} {nodes_str} {self.subckt_name}{params_str}"


def XSUB(
    subckt_name: str,
    nodes: list[str],
    params: dict[str, str | float] | None = None,
) -> SubcktInstance:
    """Helper factory for SubcktInstance with auto-ref."""
    return SubcktInstance(ref=_next("X"), subckt_name=subckt_name, nodes=nodes, params=params)


# --------------------------------------------------------------------------------------
# Probe Components
# --------------------------------------------------------------------------------------


class CurrentProbe(Component):
    """Zero-volt voltage source for current measurement.

    SPICE card: V<ref> <p> <n> DC 0

    A zero-volt source allows measuring current through a branch
    without affecting circuit behavior. Current flows from p to n.

    Args:
        ref: Reference designator (e.g., "sense1")

    Ports:
        p: Positive terminal (current enters here)
        n: Negative terminal (current exits here)

    Example:
        # Measure current through a resistor
        probe = CurrentProbe("sense")
        # Connect in series with the component to measure
        # Access current as I(Vsense) in simulation results
    """

    def __init__(self, ref: str) -> None:
        super().__init__(ref=ref, value="0")
        self._ports = (
            Port(self, "p", PortRole.POSITIVE),
            Port(self, "n", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        p, n = self.ports
        return f"V{self.ref} {net_of(p)} {net_of(n)} DC 0"


def IPROBE(name: str | None = None) -> CurrentProbe:
    """Helper factory for CurrentProbe.

    Args:
        name: Optional name for the probe. If None, auto-generates.
    """
    ref = name if name else f"sense{_next('V')}"
    return CurrentProbe(ref=ref)


class VoltageProbe(Component):
    """Explicit voltage measurement point marker.

    This is a virtual component that doesn't generate a SPICE card.
    It serves as a marker for voltage measurement between two nodes.

    In SPICE, voltages are measured directly as V(node) or V(node1, node2).
    This class provides a semantic way to mark measurement points.

    Args:
        ref: Reference designator/name for the probe
        differential: If True, measures voltage between p and n.
                     If False, measures p relative to ground.

    Ports:
        p: Positive measurement point
        n: Negative/reference measurement point

    Example:
        # Single-ended measurement (relative to ground)
        vprobe = VoltageProbe("out")

        # Differential measurement
        vprobe_diff = VoltageProbe("diff", differential=True)
    """

    def __init__(self, ref: str, differential: bool = False) -> None:
        super().__init__(ref=ref, value="probe")
        self.differential = differential
        self._ports = (
            Port(self, "p", PortRole.POSITIVE),
            Port(self, "n", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: NetOf) -> str:
        # VoltageProbe is a virtual component - no SPICE element generated
        # Voltage measurement happens via .PROBE or direct node reference
        return f"* Voltage probe {self.ref}: V({net_of(self.ports[0])})"


def VPROBE(name: str, differential: bool = False) -> VoltageProbe:
    """Helper factory for VoltageProbe."""
    return VoltageProbe(ref=name, differential=differential)
