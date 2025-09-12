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
        return f"V{self.ref} {net_of(p)} {net_of(n)} SIN({ ' '.join(args) })"


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
        return f"I{self.ref} {net_of(p)} {net_of(n)} SIN({ ' '.join(args) })"


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
        return f"V{self.ref} {net_of(p)} {net_of(n)} PWL({ ' '.join(flat) })"


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
        return f"I{self.ref} {net_of(p)} {net_of(n)} PWL({ ' '.join(flat) })"


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
