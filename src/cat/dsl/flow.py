from __future__ import annotations

from dataclasses import dataclass

from ..core.circuit import Circuit
from ..core.components import Component
from ..core.net import Net


@dataclass
class Chain:
    items: list[Component]

    def __or__(self, other: Chain) -> Parallel:
        return Parallel([self, other])

    def to_circuit(self, name: str = "chain") -> tuple[Circuit, Net, Net]:
        """
        Monta os itens em série e retorna (circuit, net_in, net_out).
        Observação: não conectamos Net→Net; apenas retornamos o último nó como saída.
        """
        c = Circuit(name)
        for it in self.items:
            c.add(it)
        a = Net("in")
        last = a
        for comp in self.items:
            p0, p1 = comp.ports
            c.connect(p0, last)
            last = Net()
            c.connect(p1, last)
        b = last  # último nó é a saída
        return c, a, b


@dataclass
class Parallel:
    branches: list[Chain]

    def to_circuit(self, name: str = "parallel") -> tuple[Circuit, Net, Net]:
        """
        Monta branches em paralelo entre 'in' e 'out'.
        Cada branch vira uma cadeia série entre os mesmos nós de entrada/saída.
        """
        c = Circuit(name)
        a = Net("in")
        for ch in self.branches:
            # Adiciona componentes da branch
            for it in ch.items:
                c.add(it)
            # Conecta: a -- items... -- b
            prev = a
            for comp in ch.items:
                p0, p1 = comp.ports
                c.connect(p0, prev)
                mid = Net()
                c.connect(p1, mid)
                prev = mid
            # Ao final, 'prev' é o último nó da branch; ligue o último pino ao 'b'
            # Para isso, precisamos adicionar um "jumper": conectar o último pino ao 'b'
            # Como não há "wire" explícito, fazemos o último pino *diretamente* em b
            # alterando a última conexão para b em vez de um Net novo.
            # Implementação simples: refaz o último componente para terminar em b.
            # (Mais eficiente seria construir com b direto no loop)
            # Refator: reconstruir a branch usando b direto
        # Refator mais simples: reconstruir de fato
        c2 = Circuit(name)
        a2 = Net("in")
        b2 = Net("out")
        # Reconstroi de forma certa: cada branch usa o mesmo 'a2' e 'b2'
        for ch in self.branches:
            for it in ch.items:
                c2.add(it)
            prev = a2
            for j, comp in enumerate(ch.items):
                p0, p1 = comp.ports
                c2.connect(p0, prev)
                if j == len(ch.items) - 1:
                    c2.connect(p1, b2)
                else:
                    mid = Net()
                    c2.connect(p1, mid)
                    prev = mid
        return c2, a2, b2


def chain(*components: Component) -> Chain:
    return Chain(list(components))


# ===== Operador >> para encadear série =====


class Seq:
    def __init__(self, items: list[Component] | None = None) -> None:
        self.items: list[Component] = items or []

    def __rshift__(self, other: Component | Seq) -> Seq:
        if isinstance(other, Seq):
            return Seq(self.items + other.items)
        return Seq(self.items + [other])

    def to_circuit(self, name: str = "series") -> tuple[Circuit, Net, Net]:
        return Chain(self.items).to_circuit(name)


def S(first: Component) -> Seq:
    """Seed para usar: S(R(...)) >> C(...) >> ..."""
    return Seq([first])
