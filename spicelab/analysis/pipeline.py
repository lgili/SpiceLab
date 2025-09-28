from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ..orchestrator import Job, JobResult, run_job
from .measure import Spec, measure


def _flatten_params(combo: Mapping[str, Any], prefix: str | None = "param_") -> dict[str, Any]:
    if not combo:
        return {}
    if prefix is None:
        return dict(combo)
    return {f"{prefix}{k}": v for k, v in combo.items()}


def measure_job_result(
    result: JobResult,
    specs: Sequence[Spec],
    *,
    return_as: Literal["python", "polars"] = "python",
    param_prefix: str | None = "param_",
) -> Any:
    """Evaluate measurement ``specs`` for each run in a JobResult and aggregate rows.

    Each output row includes the measurement fields plus the sweep parameters from the
    corresponding combo, prefixed by ``param_prefix`` (set to None to avoid prefixing).
    The return type mirrors :func:`spicelab.analysis.measure.measure` (polars or list[dict]).
    """
    all_rows: list[dict[str, Any]] = []
    for run in result.runs:
        ds = run.handle.dataset()
        rows = measure(ds, specs, return_as="python")
        params = _flatten_params(run.combo, prefix=param_prefix)
        for r in rows:
            all_rows.append({**params, **r})
    if return_as == "python":
        return all_rows
    try:
        import polars as pl
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("polars is required when return_as='polars'") from exc
    # For polars, apply a stable column ordering: param_* first (sorted),
    # then measure fields via CLI's order
    from ..cli.measure import _order_columns as _cli_order_columns

    # Union of keys
    seen: set[str] = set()
    keys: list[str] = []
    for r in all_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    params_cols = sorted(
        [k for k in keys if isinstance(k, str) and k.startswith(str(param_prefix or "param_"))]
    )
    measure_cols = [k for k in keys if k not in params_cols]
    ordered_measure = _cli_order_columns([c for c in measure_cols if isinstance(c, str)])
    cols = params_cols + ordered_measure
    df = pl.DataFrame(all_rows)
    # Keep only existing columns, in the desired order
    existing = [c for c in cols if c in df.columns]
    return df.select(existing)


def run_and_measure(
    job: Job,
    specs: Sequence[Spec],
    *,
    cache_dir: str | None = ".spicelab_cache",
    workers: int = 1,
    reuse_cache: bool = True,
    return_as: Literal["python", "polars"] = "python",
    param_prefix: str | None = "param_",
) -> Any:
    """Execute a job with the orchestrator and measure outputs for each combo.

    Convenience wrapper around :func:`run_job` + :func:`measure_job_result`.
    """
    jr = run_job(job, cache_dir=cache_dir, workers=workers, reuse_cache=reuse_cache)
    return measure_job_result(jr, specs, return_as=return_as, param_prefix=param_prefix)


__all__ = ["measure_job_result", "run_and_measure"]
