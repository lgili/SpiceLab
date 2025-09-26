"""High-level orchestration helpers for sweeps, grids and caching."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .core.types import (
    AnalysisSpec,
    Probe,
    ResultHandle,
    SweepSpec,
    circuit_hash,
    ensure_analysis_spec,
    ensure_sweep_spec,
    stable_hash,
)
from .engines.base import Simulator
from .engines.factory import create_simulator

ComboDict = dict[str, str | float]


@dataclass(frozen=True)
class Job:
    circuit: object
    analyses: Sequence[AnalysisSpec]
    sweep: SweepSpec | None = None
    probes: Sequence[Probe] | None = None
    engine: str = "ngspice"
    combos: Sequence[ComboDict] | None = None


@dataclass(frozen=True)
class JobRun:
    combo: ComboDict
    handle: ResultHandle
    from_cache: bool = False


@dataclass(frozen=True)
class JobResult:
    job: Job
    runs: Sequence[JobRun]
    cache_dir: Path | None = None

    def handles(self) -> list[ResultHandle]:
        return [run.handle for run in self.runs]


def _component_map(circuit: object) -> dict[str, Any]:
    comps = cast(Sequence[Any], getattr(circuit, "_components", []))
    mapping: dict[str, Any] = {}
    for idx, comp in enumerate(comps):
        ref = getattr(comp, "ref", str(idx))
        mapping[str(ref)] = comp
    return mapping


def _apply_combo(circuit: object, combo: Mapping[str, str | float]) -> None:
    by_ref = _component_map(circuit)
    for ref, value in combo.items():
        if ref not in by_ref:
            raise KeyError(f"Component with ref '{ref}' not found in circuit")
        comp = cast(Any, by_ref[ref])
        if not hasattr(comp, "value"):
            raise AttributeError(f"Component '{ref}' does not support value overrides")
        comp.value = value


def _expand_sweep(sweep: SweepSpec | None) -> list[ComboDict]:
    if not sweep or not sweep.variables:
        return [ComboDict()]
    variables = sweep.variables
    keys = list(variables.keys())
    value_lists = [list(vals) for vals in variables.values()]
    combos: list[ComboDict] = []
    from itertools import product

    for values in product(*value_lists):
        combo = ComboDict()
        for key, value in zip(keys, values, strict=False):
            combo[str(key)] = value
        combos.append(combo)
    if not combos:
        combos.append(ComboDict())
    return combos


def _job_hash(job: Job, combos: Iterable[ComboDict]) -> str:
    analyses_dump = [spec.model_dump(mode="json") for spec in job.analyses]
    sweep_dump = job.sweep.model_dump(mode="json") if job.sweep else None
    probes_dump = [probe.model_dump(mode="json") for probe in job.probes] if job.probes else None
    extra = {
        "engine": job.engine,
        "analyses": analyses_dump,
        "sweep": sweep_dump,
        "probes": probes_dump,
        "combos": list(combos),
    }
    return circuit_hash(job.circuit, extra=extra)


def _cache_path(base: Path, job_hash: str, index: int, combo: ComboDict) -> Path:
    combo_hash = stable_hash(combo)
    return base / job_hash / f"{index:04d}_{combo_hash}.pkl"


def _load_cached(path: Path) -> ResultHandle | None:
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return cast(ResultHandle, pickle.load(fh))


def _store_cached(path: Path, handle: ResultHandle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(handle, fh)


def _run_single(
    circuit: object,
    analyses: Sequence[AnalysisSpec],
    combo: ComboDict,
    engine: str,
    probes: Sequence[Probe] | None,
) -> ResultHandle:
    circuit_copy = copy.deepcopy(circuit)
    if combo:
        _apply_combo(circuit_copy, combo)
    sim: Simulator = create_simulator(engine)
    probes_list = list(probes) if probes else None
    return sim.run(circuit_copy, analyses, None, probes_list)


def run_job(
    job: Job,
    *,
    cache_dir: str | Path | None = ".spicelab_cache",
    workers: int = 1,
    progress: bool | Callable[[int, int], None] | None = None,
    reuse_cache: bool = True,
) -> JobResult:
    analyses = [ensure_analysis_spec(a) for a in job.analyses]
    sweep = ensure_sweep_spec(job.sweep)
    combos_input = [dict(combo) for combo in job.combos] if job.combos else _expand_sweep(sweep)
    combos = [dict(combo) for combo in combos_input] if combos_input else [ComboDict()]
    job_norm = Job(
        circuit=job.circuit,
        analyses=analyses,
        sweep=sweep,
        probes=job.probes,
        engine=job.engine,
        combos=tuple(dict(combo) for combo in combos),
    )
    base_hash = _job_hash(job_norm, combos)
    cache_root: Path | None = None
    if reuse_cache and cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

    results: list[JobRun | None] = [None] * len(combos)

    def _notify(done: int) -> None:
        if not progress:
            return
        if callable(progress):
            try:
                progress(done, len(combos))
            except Exception:
                pass
            return
        try:
            import sys

            pct = int(round(100.0 * done / max(len(combos), 1)))
            sys.stderr.write(f"\rJOB: {done}/{len(combos)} ({pct}%)")
            sys.stderr.flush()
        except Exception:
            pass

    pending_indices: list[int] = []

    for idx, combo in enumerate(combos):
        cache_path: Path | None = None
        handle: ResultHandle | None = None
        if cache_root is not None:
            cache_path = _cache_path(cache_root, base_hash, idx, combo)
            handle = _load_cached(cache_path)
        if handle is not None:
            results[idx] = JobRun(combo=combo, handle=handle, from_cache=True)
            _notify(idx + 1)
            continue
        pending_indices.append(idx)

    if pending_indices:
        if workers <= 1:
            for idx in pending_indices:
                combo = combos[idx]
                handle = _run_single(job.circuit, analyses, combo, job.engine, job.probes)
                if cache_root is not None:
                    cache_path = _cache_path(cache_root, base_hash, idx, combo)
                    _store_cached(cache_path, handle)
                results[idx] = JobRun(combo=combo, handle=handle, from_cache=False)
                _notify(idx + 1)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        _run_single,
                        job.circuit,
                        analyses,
                        combos[idx],
                        job.engine,
                        job.probes,
                    ): idx
                    for idx in pending_indices
                }
                for future in future_map:
                    idx = future_map[future]
                    combo = combos[idx]
                    handle = future.result()
                    if cache_root is not None:
                        cache_path = _cache_path(cache_root, base_hash, idx, combo)
                        _store_cached(cache_path, handle)
                    results[idx] = JobRun(combo=combo, handle=handle, from_cache=False)
                    _notify(idx + 1)

    finalized = [run for run in results if run is not None]
    if len(finalized) != len(combos):
        raise RuntimeError("Some sweep runs did not complete")
    return JobResult(
        job=job_norm, runs=finalized, cache_dir=cache_root / base_hash if cache_root else None
    )


__all__ = ["Job", "JobRun", "JobResult", "run_job"]
