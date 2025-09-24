from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

# Evita import cíclico em runtime; só para type-checkers
if TYPE_CHECKING:
    from .components import Component


class PortRole(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    NODE = auto()


@dataclass(frozen=True)
class Net:
    """Logical node. Name is optional and used for debug; '0' is reserved for GND."""

    name: str | None = None


GND = Net("0")


# hash por identidade (eq=False) para usar Port como chave em dict
@dataclass(frozen=True, eq=False)
class Port:
    owner: Component
    name: str
    role: PortRole
