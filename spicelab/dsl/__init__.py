"""DSL helpers for spicelab.

Expose only the curated public surface to avoid star-imports which confuse linters.
"""

from .builder import CircuitBuilder
from .context import (
    IC,
    TEMP,
    C,
    Circuit,
    DesignContext,
    Directive,
    DSLContextError,
    L,
    Net,
    Option,
    Param,
    R,
    V,
    place,
)
from .flow import Chain, Parallel, S, Seq, chain

__all__ = [
    "C",
    "Chain",
    "Circuit",
    "CircuitBuilder",
    "DesignContext",
    "Directive",
    "DSLContextError",
    "IC",
    "L",
    "Net",
    "NetSpec",
    "Option",
    "Param",
    "Parallel",
    "R",
    "S",
    "Seq",
    "TEMP",
    "V",
    "chain",
    "place",
]
