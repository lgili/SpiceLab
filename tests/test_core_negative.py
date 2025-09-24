from typing import cast

import pytest
from pydantic import ValidationError
from spicelab.core.types import AnalysisSpec, Probe, SweepSpec


def test_sweep_spec_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError):
        SweepSpec(variables={"R": []})  # empty list invalid


def test_sweep_spec_rejects_bad_type() -> None:
    with pytest.raises((ValueError, TypeError)):
        bad_values = cast(list[float | str], [object()])  # runtime invalid, static appeased
        SweepSpec(variables={"R": bad_values})


def test_probe_requires_target() -> None:
    with pytest.raises(ValidationError):
        Probe(kind="voltage", target="")


def test_analysis_spec_rejects_non_primitive_list_member() -> None:
    with pytest.raises(ValueError):
        AnalysisSpec(mode="ac", args={"n": [1, object()]})
