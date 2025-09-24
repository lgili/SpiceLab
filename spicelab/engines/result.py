from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, cast

from ..core.types import ResultHandle, ResultMeta


class DatasetResultHandle(ResultHandle):
    """Concrete ResultHandle backed by an in-memory xarray.Dataset.

    The dataset is stored as `Any` to avoid a hard import-time dependency on xarray
    during type-checking. Conversions are performed lazily.
    """

    def __init__(self, dataset: Any, meta: ResultMeta) -> None:
        self._dataset = dataset
        self._meta = meta
        # Attach attrs on the dataset if it supports it (xarray.Dataset-like)
        try:
            ds_attrs = getattr(self._dataset, "attrs", None)
            if isinstance(ds_attrs, dict):
                analyses_serialized: list[dict[str, Any]] = []
                for a in meta.analyses:
                    if hasattr(a, "model_dump"):
                        analyses_serialized.append(a.model_dump())
                    else:  # dataclass fallback
                        analyses_serialized.append(asdict(a))
                probes_serialized: list[dict[str, Any]] = []
                for p in getattr(meta, "probes", []) or []:
                    if hasattr(p, "model_dump"):
                        probes_serialized.append(p.model_dump())
                    else:
                        try:
                            probes_serialized.append(asdict(p))
                        except Exception:
                            probes_serialized.append({"repr": repr(p)})
                ds_attrs.update(
                    {
                        "engine": meta.engine,
                        "engine_version": meta.engine_version,
                        "netlist_hash": meta.netlist_hash,
                        "analyses": analyses_serialized,
                        "probes": probes_serialized,
                        **meta.attrs,
                    }
                )
        except Exception:
            # best effort only
            pass

    def dataset(self) -> Any:
        return self._dataset

    def to_polars(self) -> Any:
        """Return results as a polars.DataFrame.

        Conversion path: xarray.Dataset -> pandas.DataFrame -> polars.DataFrame.
        """

        try:
            pl = importlib.import_module("polars")
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "polars is required for to_polars(); install with: pip install polars"
            ) from exc
        # xarray depends on pandas for DataFrame conversion
        try:
            to_df = getattr(self._dataset, "to_dataframe", None)
            if to_df is None:
                raise AttributeError("dataset has no to_dataframe()")
            pdf = cast(Any, to_df)()
        except Exception as exc:  # pragma: no cover - relies on xarray
            raise RuntimeError(
                "Underlying dataset does not support to_dataframe() (xarray required)"
            ) from exc
        return pl.from_pandas(pdf)

    def to_pandas(self) -> Any:
        """Return results as a pandas.DataFrame (requires xarray+pandas)."""
        to_df = getattr(self._dataset, "to_dataframe", None)
        if to_df is None:
            raise RuntimeError("Underlying dataset does not support to_dataframe()")
        return to_df()

    def attrs(self) -> Mapping[str, Any]:
        analyses_serialized: list[dict[str, Any]] = []
        for a in self._meta.analyses:
            if hasattr(a, "model_dump"):
                analyses_serialized.append(a.model_dump())
            else:
                analyses_serialized.append(asdict(a))
        probes_serialized: list[dict[str, Any]] = []
        for p in getattr(self._meta, "probes", []) or []:
            if hasattr(p, "model_dump"):
                probes_serialized.append(p.model_dump())
            else:
                try:
                    probes_serialized.append(asdict(p))
                except Exception:
                    probes_serialized.append({"repr": repr(p)})
        return {
            "engine": self._meta.engine,
            "engine_version": self._meta.engine_version,
            "netlist_hash": self._meta.netlist_hash,
            "analyses": analyses_serialized,
            "probes": probes_serialized,
            **self._meta.attrs,
        }


__all__ = ["DatasetResultHandle"]
