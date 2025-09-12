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


def L(value: str | float) -> Inductor:
    return Inductor(ref=_next("L"), value=value)


def I(value: str | float) -> Idc:  # noqa: E743 - single-letter helper kept for API symmetry
    return Idc(ref=_next("I"), value=value)


def IA(ac_mag: float = 1.0, ac_phase: float = 0.0, label: str | float = "") -> Iac:
    return Iac(ref=_next("I"), value=str(label), ac_mag=ac_mag, ac_phase=ac_phase)
