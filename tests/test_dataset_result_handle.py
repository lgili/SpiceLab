from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from spicelab.core.types import AnalysisSpec, ResultMeta
from spicelab.engines.result import DatasetResultHandle

xr = cast(Any, pytest.importorskip("xarray"))

if "polars" not in sys.modules:

    class _FakePolars(ModuleType):
        def __init__(self) -> None:
            super().__init__("polars")

        @staticmethod
        def from_pandas(pdf: Any) -> tuple[str, Any]:
            return ("polars", pdf)

        @staticmethod
        def DataFrame(rows: Any) -> tuple[str, Any]:
            return ("polars", rows)

    sys.modules["polars"] = _FakePolars()


def _handle() -> DatasetResultHandle:
    ds = xr.Dataset(
        {"V(out)": ("time", np.array([0.0, 1.0]))}, coords={"time": np.array([0.0, 1.0])}
    )
    meta = ResultMeta(
        engine="ngspice", analyses=[AnalysisSpec(mode="op", args={})], attrs={"foo": "bar"}
    )
    return DatasetResultHandle(ds, meta)


def test_dataset_result_handle_attrs_updates_dataset() -> None:
    handle = _handle()
    attrs = handle.attrs()
    assert attrs["engine"] == "ngspice"
    assert handle.dataset().attrs["foo"] == "bar"


def test_dataset_result_handle_to_polars(monkeypatch: pytest.MonkeyPatch) -> None:
    polars_module = cast(Any, ModuleType("polars"))
    frame_holder: dict[str, Any] = {}

    def _from_pandas(pdf: Any) -> tuple[str, Any]:
        frame_holder["df"] = pdf
        return ("polars", pdf)

    polars_module.from_pandas = _from_pandas
    monkeypatch.setitem(sys.modules, "polars", polars_module)
    handle = _handle()
    result = handle.to_polars()
    assert result[0] == "polars"
    assert "df" in frame_holder


def test_dataset_result_handle_to_pandas() -> None:
    handle = _handle()
    df = handle.to_pandas()
    assert hasattr(df, "index")


def test_dataset_result_handle_to_polars_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadDataset:
        def to_dataframe(self) -> None:
            raise RuntimeError("boom")

    meta = ResultMeta(engine="ngspice")
    handle = DatasetResultHandle(BadDataset(), meta)
    polars_module = cast(Any, ModuleType("polars"))

    def _raise(_: Any) -> None:
        raise RuntimeError("bad")

    polars_module.from_pandas = _raise
    monkeypatch.setitem(sys.modules, "polars", polars_module)
    with pytest.raises(RuntimeError):
        handle.to_polars()


def test_dataset_result_handle_to_pandas_missing() -> None:
    meta = ResultMeta(engine="ngspice")
    handle = DatasetResultHandle(object(), meta)
    with pytest.raises(RuntimeError):
        handle.to_pandas()
