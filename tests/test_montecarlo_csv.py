import os
from pathlib import Path
from typing import cast

import pytest
from spicelab.analysis.montecarlo import MonteCarloResult
from spicelab.analysis.result import AnalysisResult


def make_sample_table() -> MonteCarloResult:
    samples = [
        {"R1": 100.0, "V1": 5.0},
        {"R1": 110.0, "V1": 5.1},
    ]
    # MonteCarloResult.runs expects a list[AnalysisResult]; in tests we don't need
    # real AnalysisResult objects, so provide an empty list and cast to satisfy mypy
    runs = cast(list[AnalysisResult], [])
    manifest = [("R1", 100.0, "DummyDist(0.1)"), ("V1", 5.0, "DummyDist(0.0)")]
    return MonteCarloResult(samples=samples, runs=runs, mapping_manifest=manifest)


@pytest.mark.skipif(os.getenv("CI_REQUIRE_PANDAS") != "1", reason="requires pandas to be installed")
def test_save_samples_and_manifest_csv(tmp_path: Path) -> None:
    mc = make_sample_table()

    samples_path = tmp_path / "samples.csv"
    manifest_path = tmp_path / "manifest.csv"

    mc.save_samples_csv(str(samples_path))
    mc.save_manifest_csv(str(manifest_path))

    # quick sanity checks
    s_text = samples_path.read_text(encoding="utf-8")
    assert "R1" in s_text
    assert "V1" in s_text

    m_text = manifest_path.read_text(encoding="utf-8")
    assert "label" in m_text
    assert "DummyDist(0.1)" in m_text
