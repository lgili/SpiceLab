"""Tests for Phase 3: Engine Improvements."""

from __future__ import annotations

import pytest
from spicelab.engines.base import EngineFeatures, Simulator
from spicelab.engines.factory import create_simulator
from spicelab.engines.selection import (
    ENGINE_PREFERENCE,
    discover_engines,
    get_default_engine,
    select_engine,
)


# ======================================================================================
# P3.1: Context Manager Tests
# ======================================================================================
def test_simulator_context_manager():
    """Simulators support context manager protocol."""
    try:
        sim = create_simulator("ngspice")
    except Exception:
        pytest.skip("ngspice not available")

    # Context manager protocol
    with sim as s:
        assert s is sim
        assert hasattr(s, "run")
        assert hasattr(s, "features")


def test_ngspice_shared_context_manager():
    """NgSpiceSharedSimulator implements context manager."""
    try:
        from spicelab.engines.ngspice_shared import NgSpiceSharedSimulator
    except ImportError:
        pytest.skip("ngspice-shared not available")

    sim = NgSpiceSharedSimulator()

    with sim as s:
        assert s is sim
        assert s._initialized is True

    # After exit, cleaned up
    assert sim._initialized is False


def test_ngspice_proc_context_manager():
    """NgSpiceProcSimulator implements context manager (no-op)."""
    from spicelab.engines.ngspice_proc import NgSpiceProcSimulator

    sim = NgSpiceProcSimulator()

    with sim as s:
        assert s is sim
        # No state change for process-based simulators


# ======================================================================================
# P3.2/P3.3: Feature Matching and Engine Selection
# ======================================================================================
def test_engine_features_satisfies_all_true():
    """EngineFeatures.satisfies() returns True when all features match."""
    engine = EngineFeatures(
        "ngspice-shared",
        supports_callbacks=True,
        supports_noise=True,
        supports_shared_lib=True,
    )

    required = EngineFeatures(
        "",
        supports_callbacks=True,
        supports_noise=True,
    )

    assert engine.satisfies(required) is True


def test_engine_features_satisfies_missing_feature():
    """EngineFeatures.satisfies() returns False when feature missing."""
    engine = EngineFeatures(
        "ngspice",
        supports_callbacks=False,
        supports_noise=True,
    )

    required = EngineFeatures(
        "",
        supports_callbacks=True,  # Engine doesn't have this
    )

    assert engine.satisfies(required) is False


def test_engine_features_satisfies_no_requirements():
    """EngineFeatures.satisfies() returns True when no features required."""
    engine = EngineFeatures("ngspice")
    required = EngineFeatures("")  # All features False

    assert engine.satisfies(required) is True


def test_discover_engines():
    """discover_engines() returns dict of available engines."""
    available = discover_engines()

    assert isinstance(available, dict)
    assert "ngspice" in available
    assert "ngspice-shared" in available
    assert "xyce" in available
    assert "ltspice" in available

    # Values are booleans
    for _engine, is_available in available.items():
        assert isinstance(is_available, bool)


def test_select_engine_no_requirements():
    """select_engine() returns best available engine when no requirements."""
    try:
        sim = select_engine()
    except RuntimeError:
        pytest.skip("No simulation engines available")

    assert isinstance(sim, Simulator)
    assert hasattr(sim, "run")
    assert hasattr(sim, "features")


def test_select_engine_with_callbacks_requirement():
    """select_engine() respects callback requirement."""
    required = EngineFeatures("", supports_callbacks=True)

    try:
        sim = select_engine(required=required)
    except RuntimeError as e:
        if "No available engine" in str(e):
            pytest.skip("No engine with callbacks available")
        raise

    # Selected engine must support callbacks
    assert sim.features().supports_callbacks is True


def test_select_engine_custom_preference():
    """select_engine() respects custom preference order."""
    available = discover_engines()

    # Find two available engines
    available_engines = [k for k, v in available.items() if v]
    if len(available_engines) < 1:
        pytest.skip("Need at least one engine for this test")

    # Prefer the first available
    preference = available_engines[:1]

    sim = select_engine(preference=preference)

    # Engine name might be aliased (e.g., "ngspice" -> "ngspice-cli")
    # Check that selected engine is from preference list (or its alias)
    engine_name = sim.features().name
    assert (
        engine_name in preference
        or engine_name == "ngspice-cli"
        and "ngspice" in preference
        or engine_name == "xyce-cli"
        and "xyce" in preference
    )


def test_select_engine_no_engines_available(monkeypatch):
    """select_engine() raises RuntimeError when no engines available."""

    # Mock discover_engines to return all False
    def mock_discover():
        return {
            "ngspice-shared": False,
            "ngspice": False,
            "xyce": False,
            "ltspice": False,
        }

    monkeypatch.setattr("spicelab.engines.selection.discover_engines", mock_discover)

    with pytest.raises(RuntimeError, match="No simulation engines available"):
        select_engine()


def test_select_engine_requirements_not_met(monkeypatch):
    """select_engine() raises RuntimeError when requirements not satisfied."""

    # Mock discover_engines to have ngspice (no callbacks)
    def mock_discover():
        return {"ngspice": True, "ngspice-shared": False, "xyce": False, "ltspice": False}

    monkeypatch.setattr("spicelab.engines.selection.discover_engines", mock_discover)

    # Require callbacks (ngspice doesn't have it)
    required = EngineFeatures("", supports_callbacks=True)

    with pytest.raises(RuntimeError, match="No available engine satisfies requirements"):
        select_engine(required=required)


def test_get_default_engine():
    """get_default_engine() returns name of best engine."""
    try:
        engine_name = get_default_engine()
    except RuntimeError:
        pytest.skip("No engines available")

    assert isinstance(engine_name, str)
    # Engine name might be aliased (ngspice -> ngspice-cli)
    assert engine_name in ENGINE_PREFERENCE or engine_name.replace("-cli", "") in ENGINE_PREFERENCE


def test_engine_preference_order():
    """ENGINE_PREFERENCE has expected order."""
    assert ENGINE_PREFERENCE[0] == "ngspice-shared"  # Fastest
    assert ENGINE_PREFERENCE[1] == "ngspice"  # Most compatible
    assert "xyce" in ENGINE_PREFERENCE
    assert "ltspice" in ENGINE_PREFERENCE


# ======================================================================================
# Integration Tests
# ======================================================================================
def test_select_and_use_engine():
    """Full workflow: select engine and use it with context manager."""
    try:
        sim = select_engine()
    except RuntimeError:
        pytest.skip("No engines available")

    # Use with context manager
    with sim as s:
        features = s.features()
        # Engine name might be aliased
        assert (
            features.name in ENGINE_PREFERENCE
            or features.name.replace("-cli", "") in ENGINE_PREFERENCE
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
