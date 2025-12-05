"""Engine selection and discovery utilities (Phase 3).

This module provides smart engine selection based on:
1. Feature requirements (callbacks, noise, etc.)
2. Binary/library availability
3. User preferences

Example:
    >>> from spicelab.engines.selection import select_engine
    >>> from spicelab.engines.base import EngineFeatures
    >>>
    >>> # Auto-select best engine with callbacks support
    >>> sim = select_engine(
    ...     required=EngineFeatures("", supports_callbacks=True)
    ... )
    >>> # Returns NgSpiceSharedSimulator if libngspice available
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .base import EngineFeatures, Simulator
from .exceptions import EngineSharedLibraryNotFound
from .factory import create_simulator

if TYPE_CHECKING:
    pass

__all__ = [
    "select_engine",
    "discover_engines",
    "ENGINE_PREFERENCE",
]

# Default preference order (best to worst)
ENGINE_PREFERENCE = [
    "ngspice-shared",  # Fastest, supports callbacks
    "ngspice",  # Most compatible
    "xyce",  # Parallel support
    "ltspice",  # Windows-heavy, slower
]


def discover_engines() -> dict[str, bool]:
    """Discover which engines are available on this system.

    Returns:
        Dict mapping engine name to availability status.

    Example:
        >>> engines = discover_engines()
        >>> if engines["ngspice-shared"]:
        ...     print("libngspice is available")
    """
    available = {}

    # Check ngspice-shared (requires libngspice.so/.dll/.dylib)
    try:
        from ..spice.ngspice_shared_backend import NgSpiceShared  # type: ignore[attr-defined]

        NgSpiceShared()  # Try to load
        available["ngspice-shared"] = True
    except (ImportError, OSError, EngineSharedLibraryNotFound):
        available["ngspice-shared"] = False

    # Check ngspice CLI (requires ngspice binary)
    available["ngspice"] = shutil.which("ngspice") is not None

    # Check xyce (requires Xyce binary)
    available["xyce"] = shutil.which("Xyce") is not None or shutil.which("xyce") is not None

    # Check ltspice (requires wine + ltspice or native on Windows)
    available["ltspice"] = (
        shutil.which("ltspice") is not None
        or shutil.which("wine") is not None  # Assume LTspice installed under wine
    )

    return available


def select_engine(
    required: EngineFeatures | None = None,
    preference: Sequence[str] | None = None,
) -> Simulator:
    """Auto-select best available engine matching requirements.

    Selection algorithm:
    1. Iterate through preference list (or ENGINE_PREFERENCE default)
    2. For each engine, check if it's available
    3. Check if it satisfies required features
    4. Return first match

    Args:
        required: Feature requirements (callbacks, noise, etc.)
                  If None, any engine is acceptable.
        preference: Custom engine preference order.
                   If None, uses ENGINE_PREFERENCE default.

    Returns:
        Simulator instance for the best available engine.

    Raises:
        RuntimeError: If no engine satisfies requirements.

    Example:
        >>> # Auto-select (will use ngspice-shared if available)
        >>> sim = select_engine()
        >>>
        >>> # Require callbacks support
        >>> sim = select_engine(
        ...     required=EngineFeatures("", supports_callbacks=True)
        ... )
        >>>
        >>> # Custom preference (prefer Xyce)
        >>> sim = select_engine(preference=["xyce", "ngspice"])
    """
    preference_list = preference if preference is not None else ENGINE_PREFERENCE
    available = discover_engines()

    for engine_name in preference_list:
        # Skip if not available
        if not available.get(engine_name, False):
            continue

        # Try to create simulator
        try:
            sim = create_simulator(engine_name)
        except Exception:
            continue

        # Check feature requirements
        if required is None or sim.features().satisfies(required):
            return sim

    # No engine found
    if required:
        raise RuntimeError(
            f"No available engine satisfies requirements: {required}. "
            f"Available engines: {[k for k, v in available.items() if v]}"
        )
    else:
        raise RuntimeError(
            "No simulation engines available. " "Install ngspice, Xyce, or LTspice to use SpiceLab."
        )


def get_default_engine() -> str:
    """Get the name of the best available engine (for backward compat).

    Returns:
        Engine name string (e.g., "ngspice-shared", "ngspice").

    Example:
        >>> engine_name = get_default_engine()
        >>> sim = create_simulator(engine_name)
    """
    try:
        sim = select_engine()
        return sim.features().name
    except RuntimeError:
        # Fallback to "ngspice" if nothing available (will fail later)
        return "ngspice"
