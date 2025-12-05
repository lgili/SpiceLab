"""Tests for condition sweep module."""

from __future__ import annotations

from spicelab.analysis.sweep import (
    ConditionResult,
    MonteCarloConditionResult,
    MonteCarloSweepResult,
    WcaConditionResult,
    WcaSweepResult,
    _apply_conditions_to_analyses,
)
from spicelab.core.types import AnalysisSpec


class TestConditionResult:
    """Tests for ConditionResult dataclass."""

    def test_basic_creation(self):
        """ConditionResult should store condition and label."""
        cr = ConditionResult(
            condition={"temp": 25},
            condition_label="temp=25",
        )
        assert cr.condition == {"temp": 25}
        assert cr.condition_label == "temp=25"

    def test_multiple_conditions(self):
        """ConditionResult should handle multiple conditions."""
        cr = ConditionResult(
            condition={"temp": 85, "vcc": 5.5},
            condition_label="temp=85, vcc=5.5",
        )
        assert cr.condition["temp"] == 85
        assert cr.condition["vcc"] == 5.5


class TestMonteCarloSweepResult:
    """Tests for MonteCarloSweepResult."""

    def test_empty_result(self):
        """MonteCarloSweepResult should handle empty results."""
        result = MonteCarloSweepResult(
            condition_results=[],
            conditions={"temp": []},
            n=100,
        )
        assert len(result) == 0

    def test_get_by_condition(self):
        """get_by_condition should find matching result."""
        # Create mock condition results
        cr1 = MonteCarloConditionResult(
            condition={"temp": 25},
            condition_label="temp=25",
            mc_result=None,  # type: ignore
        )
        cr2 = MonteCarloConditionResult(
            condition={"temp": 85},
            condition_label="temp=85",
            mc_result=None,  # type: ignore
        )

        result = MonteCarloSweepResult(
            condition_results=[cr1, cr2],
            conditions={"temp": [25, 85]},
            n=100,
        )

        found = result.get_by_condition(temp=85)
        assert found is cr2

        not_found = result.get_by_condition(temp=100)
        assert not_found is None

    def test_iteration(self):
        """MonteCarloSweepResult should be iterable."""
        cr1 = MonteCarloConditionResult(
            condition={"temp": 25},
            condition_label="temp=25",
            mc_result=None,  # type: ignore
        )
        cr2 = MonteCarloConditionResult(
            condition={"temp": 85},
            condition_label="temp=85",
            mc_result=None,  # type: ignore
        )

        result = MonteCarloSweepResult(
            condition_results=[cr1, cr2],
            conditions={"temp": [25, 85]},
            n=100,
        )

        items = list(result)
        assert len(items) == 2
        assert items[0] is cr1
        assert items[1] is cr2

    def test_indexing(self):
        """MonteCarloSweepResult should support indexing."""
        cr1 = MonteCarloConditionResult(
            condition={"temp": 25},
            condition_label="temp=25",
            mc_result=None,  # type: ignore
        )

        result = MonteCarloSweepResult(
            condition_results=[cr1],
            conditions={"temp": [25]},
            n=100,
        )

        assert result[0] is cr1


class TestWcaSweepResult:
    """Tests for WcaSweepResult."""

    def test_empty_result(self):
        """WcaSweepResult should handle empty results."""
        result = WcaSweepResult(
            condition_results=[],
            conditions={"temp": []},
        )
        assert len(result) == 0

    def test_get_by_condition(self):
        """get_by_condition should find matching result."""
        cr1 = WcaConditionResult(
            condition={"temp": -40},
            condition_label="temp=-40",
            wca_result=None,  # type: ignore
        )
        cr2 = WcaConditionResult(
            condition={"temp": 85},
            condition_label="temp=85",
            wca_result=None,  # type: ignore
        )

        result = WcaSweepResult(
            condition_results=[cr1, cr2],
            conditions={"temp": [-40, 85]},
        )

        found = result.get_by_condition(temp=-40)
        assert found is cr1


class TestApplyConditionsToAnalyses:
    """Tests for _apply_conditions_to_analyses helper."""

    def test_apply_temperature(self):
        """Should apply temperature to analysis args."""
        analyses = [AnalysisSpec(mode="op")]
        conditions = {"temp": 85}

        modified = _apply_conditions_to_analyses(analyses, conditions)

        assert len(modified) == 1
        assert modified[0].args is not None
        assert modified[0].args.get("temp") == 85

    def test_preserve_existing_args(self):
        """Should preserve existing args when adding temperature."""
        analyses = [AnalysisSpec(mode="op", args={"gmin": 1e-12})]
        conditions = {"temp": 25}

        modified = _apply_conditions_to_analyses(analyses, conditions)

        assert modified[0].args.get("gmin") == 1e-12
        assert modified[0].args.get("temp") == 25

    def test_no_temperature_condition(self):
        """Should not modify args if no temp condition."""
        analyses = [AnalysisSpec(mode="op")]
        conditions = {"vcc": 5.0}  # Not a temperature condition

        modified = _apply_conditions_to_analyses(analyses, conditions)

        # Args should be empty dict, temp should not be set
        assert modified[0].args.get("temp") is None

    def test_multiple_analyses(self):
        """Should apply conditions to all analyses."""
        analyses = [
            AnalysisSpec(mode="op"),
            AnalysisSpec(mode="tran", args={"stop": 1e-3}),
        ]
        conditions = {"temp": -40}

        modified = _apply_conditions_to_analyses(analyses, conditions)

        assert len(modified) == 2
        assert modified[0].args.get("temp") == -40
        assert modified[1].args.get("temp") == -40
        assert modified[1].args.get("stop") == 1e-3  # Preserved


class TestConditionCombinations:
    """Tests for multiple condition combinations."""

    def test_cartesian_product_conditions(self):
        """Should create results for all condition combinations."""
        # This is a unit test for the data structures
        # Full integration tests would require actual circuit simulation

        temps = [-40, 25, 85]
        vccs = [4.5, 5.0, 5.5]

        # Simulate what monte_carlo_sweep would produce
        import itertools

        combos = list(itertools.product(temps, vccs))
        assert len(combos) == 9  # 3 temps x 3 vccs

        condition_results = []
        for temp, vcc in combos:
            condition = {"temp": temp, "vcc": vcc}
            condition_label = f"temp={temp}, vcc={vcc}"
            cr = MonteCarloConditionResult(
                condition=condition,
                condition_label=condition_label,
                mc_result=None,  # type: ignore
            )
            condition_results.append(cr)

        result = MonteCarloSweepResult(
            condition_results=condition_results,
            conditions={"temp": temps, "vcc": vccs},
            n=100,
        )

        assert len(result) == 9

        # Find specific combination
        found = result.get_by_condition(temp=85, vcc=5.5)
        assert found is not None
        assert found.condition == {"temp": 85, "vcc": 5.5}
