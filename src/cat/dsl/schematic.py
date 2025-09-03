from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ..core.circuit import Circuit


@contextmanager
def schematic(name: str) -> Iterator[Circuit]:
    c = Circuit(name)
    try:
        yield c
    finally:
        # no-op: deixa o usuário retornar o objeto Circuit se quiser
        ...
