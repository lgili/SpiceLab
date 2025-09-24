from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .net import Port


@runtime_checkable
class HasPorts(Protocol):
    @property
    def ports(self) -> tuple[Port, ...]: ...
