"""Parallel execution engine for batch simulations.

This module provides efficient parallel execution of simulations using
Python's multiprocessing capabilities, with support for:
- ProcessPoolExecutor for CPU-bound simulation tasks
- ThreadPoolExecutor for I/O-bound operations
- Progress tracking with optional tqdm integration
- Robust error handling with detailed job results
- Automatic resource cleanup

Example::

    from spicelab.parallel import ParallelExecutor

    def simulate(params):
        circuit = build_circuit(params)
        return run_simulation(circuit)

    executor = ParallelExecutor(max_workers=8)
    results = executor.map(simulate, [
        {"temp": -40},
        {"temp": 25},
        {"temp": 85},
    ])

    for result in results:
        if result.success:
            print(f"Job {result.job_id}: {result.value}")
        else:
            print(f"Job {result.job_id} failed: {result.error}")

"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class JobStatus(Enum):
    """Status of a parallel job."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class JobResult(Generic[R]):
    """Result of a single parallel job.

    Attributes:
        job_id: Unique identifier for the job (usually index)
        status: Current status of the job
        value: Result value if completed successfully
        error: Exception message if failed
        error_type: Type of exception if failed
        traceback: Full traceback if failed
        start_time: Unix timestamp when job started
        end_time: Unix timestamp when job finished
        duration_ms: Duration in milliseconds
        input_params: Original input parameters

    """

    job_id: int
    status: JobStatus
    value: R | None = None
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    duration_ms: float | None = None
    input_params: Any = None

    @property
    def success(self) -> bool:
        """Whether the job completed successfully."""
        return self.status == JobStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Whether the job failed."""
        return self.status == JobStatus.FAILED


@dataclass
class BatchResult(Generic[R]):
    """Results from a batch of parallel jobs.

    Attributes:
        results: List of individual job results
        total_jobs: Total number of jobs
        completed_jobs: Number of successfully completed jobs
        failed_jobs: Number of failed jobs
        total_duration_ms: Total wall-clock time in milliseconds
        avg_job_duration_ms: Average job duration in milliseconds

    """

    results: list[JobResult[R]]
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    total_duration_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def __post_init__(self) -> None:
        """Calculate statistics after initialization."""
        self.completed_jobs = sum(1 for r in self.results if r.success)
        self.failed_jobs = sum(1 for r in self.results if r.failed)
        self.cancelled_jobs = sum(
            1 for r in self.results if r.status == JobStatus.CANCELLED
        )

    @property
    def success_rate(self) -> float:
        """Fraction of jobs that completed successfully."""
        if self.total_jobs == 0:
            return 0.0
        return self.completed_jobs / self.total_jobs

    @property
    def avg_job_duration_ms(self) -> float:
        """Average duration per job in milliseconds."""
        durations = [r.duration_ms for r in self.results if r.duration_ms is not None]
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    def get_successful_results(self) -> list[R]:
        """Get values from all successful jobs."""
        return [r.value for r in self.results if r.success and r.value is not None]

    def get_failed_jobs(self) -> list[JobResult[R]]:
        """Get all failed job results."""
        return [r for r in self.results if r.failed]

    def get_result_by_id(self, job_id: int) -> JobResult[R] | None:
        """Get result for a specific job ID."""
        for r in self.results:
            if r.job_id == job_id:
                return r
        return None


# Type alias for progress callback
ProgressCallback = Callable[[int, int, JobResult[Any] | None], None]


def _default_progress_callback(
    completed: int, total: int, result: JobResult[Any] | None
) -> None:
    """Default progress callback that prints to stdout."""
    status = ""
    if result is not None:
        status = "✓" if result.success else "✗"
    print(f"\rProgress: {completed}/{total} {status}", end="", flush=True)
    if completed == total:
        print()  # Newline at end


def _worker_wrapper(
    func: Callable[[T], R],
    args: T,
    job_id: int,
) -> JobResult[R]:
    """Wrapper function that runs in worker process.

    Catches exceptions and returns JobResult with timing information.
    """
    start_time = time.time()
    try:
        result = func(args)
        end_time = time.time()
        return JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            value=result,
            start_time=start_time,
            end_time=end_time,
            duration_ms=(end_time - start_time) * 1000,
            input_params=args,
        )
    except Exception as e:
        end_time = time.time()
        return JobResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            start_time=start_time,
            end_time=end_time,
            duration_ms=(end_time - start_time) * 1000,
            input_params=args,
        )


class ParallelExecutor:
    """Execute functions in parallel using process or thread pools.

    This executor provides a high-level interface for running functions
    in parallel with automatic error handling, progress tracking, and
    result collection.

    Args:
        max_workers: Maximum number of parallel workers.
            Defaults to CPU count.
        use_processes: Use processes (True) or threads (False).
            Processes are better for CPU-bound work.
        progress: Show progress. Can be True (default callback),
            False (no progress), or a custom callback function.
        timeout: Timeout per job in seconds (None for no timeout).

    Example::

        executor = ParallelExecutor(max_workers=4)

        # Simple map
        results = executor.map(process_data, data_list)

        # With progress
        results = executor.map(simulate, params, progress=True)

        # Custom progress callback
        def my_progress(done, total, result):
            print(f"{done}/{total}")
        results = executor.map(simulate, params, progress=my_progress)

    """

    def __init__(
        self,
        max_workers: int | None = None,
        use_processes: bool = True,
        progress: bool | ProgressCallback = False,
        timeout: float | None = None,
    ) -> None:
        """Initialize the parallel executor."""
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.timeout = timeout

        # Progress callback
        if progress is True:
            self._progress_callback: ProgressCallback | None = _default_progress_callback
        elif progress is False:
            self._progress_callback = None
        else:
            self._progress_callback = progress

    def _get_executor(self) -> ProcessPoolExecutor | ThreadPoolExecutor:
        """Get the appropriate executor type."""
        if self.use_processes:
            # Use spawn for better compatibility
            ctx = multiprocessing.get_context("spawn")
            return ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
        else:
            return ThreadPoolExecutor(max_workers=self.max_workers)

    def map(
        self,
        func: Callable[[T], R],
        inputs: Sequence[T],
        progress: bool | ProgressCallback | None = None,
        ordered: bool = True,
    ) -> BatchResult[R]:
        """Execute function on all inputs in parallel.

        Args:
            func: Function to execute. Must be pickleable for process pools.
            inputs: Sequence of inputs to process.
            progress: Override default progress setting.
            ordered: Return results in input order (True) or completion order (False).

        Returns:
            BatchResult containing all job results.

        """
        # Determine progress callback
        if progress is None:
            callback = self._progress_callback
        elif progress is True:
            callback = _default_progress_callback
        elif progress is False:
            callback = None
        else:
            callback = progress

        total = len(inputs)
        if total == 0:
            return BatchResult(results=[], total_jobs=0)

        # Use tqdm if available and using default progress (not custom callback)
        tqdm_bar = None
        use_tqdm = callback is _default_progress_callback
        if use_tqdm:
            try:
                from tqdm import tqdm

                tqdm_bar = tqdm(total=total, desc="Processing", unit="job")
                callback = None  # Use tqdm instead of default callback
            except ImportError:
                pass

        results: list[JobResult[R]] = [
            JobResult(job_id=i, status=JobStatus.PENDING) for i in range(total)
        ]
        batch_start = time.time()

        try:
            with self._get_executor() as executor:
                # Submit all jobs
                futures: dict[Future[JobResult[R]], int] = {}
                for i, input_data in enumerate(inputs):
                    future = executor.submit(_worker_wrapper, func, input_data, i)
                    futures[future] = i

                # Collect results
                completed = 0
                for future in as_completed(futures, timeout=self.timeout):
                    job_id = futures[future]
                    try:
                        job_result = future.result()
                        results[job_id] = job_result
                    except Exception as e:
                        results[job_id] = JobResult(
                            job_id=job_id,
                            status=JobStatus.FAILED,
                            error=str(e),
                            error_type=type(e).__name__,
                            traceback=traceback.format_exc(),
                        )

                    completed += 1

                    # Update progress
                    if tqdm_bar is not None:
                        tqdm_bar.update(1)
                    elif callback is not None:
                        callback(completed, total, results[job_id])

        except KeyboardInterrupt:
            # Mark remaining jobs as cancelled
            for i, result in enumerate(results):
                if result.status == JobStatus.PENDING:
                    results[i] = JobResult(job_id=i, status=JobStatus.CANCELLED)
        finally:
            if tqdm_bar is not None:
                tqdm_bar.close()

        batch_end = time.time()

        batch_result = BatchResult(
            results=results,
            total_jobs=total,
            total_duration_ms=(batch_end - batch_start) * 1000,
            start_time=batch_start,
            end_time=batch_end,
        )

        return batch_result

    def run_batch(
        self,
        func: Callable[[T], R],
        inputs: Sequence[T],
        progress: bool | ProgressCallback | None = None,
    ) -> list[R | None]:
        """Execute function on all inputs and return values only.

        This is a simplified version of map() that returns just the result
        values (or None for failed jobs).

        Args:
            func: Function to execute.
            inputs: Sequence of inputs to process.
            progress: Show progress indicator.

        Returns:
            List of results in input order. Failed jobs return None.

        """
        batch = self.map(func, inputs, progress=progress)
        return [r.value for r in batch.results]

    def starmap(
        self,
        func: Callable[..., R],
        inputs: Sequence[tuple[Any, ...]],
        progress: bool | ProgressCallback | None = None,
    ) -> BatchResult[R]:
        """Execute function with unpacked arguments.

        Similar to itertools.starmap but parallel.

        Args:
            func: Function to call with unpacked arguments.
            inputs: Sequence of argument tuples.
            progress: Show progress indicator.

        Returns:
            BatchResult containing all job results.

        """

        def wrapper(args: tuple[Any, ...]) -> R:
            return func(*args)

        return self.map(wrapper, inputs, progress=progress)


def parallel_map(
    func: Callable[[T], R],
    inputs: Iterable[T],
    max_workers: int | None = None,
    progress: bool = False,
) -> list[R | None]:
    """Convenience function for simple parallel mapping.

    Args:
        func: Function to execute on each input.
        inputs: Inputs to process.
        max_workers: Number of workers (default: CPU count).
        progress: Show progress bar.

    Returns:
        List of results (None for failed jobs).

    Example::

        # Process files in parallel
        results = parallel_map(process_file, file_list, progress=True)

    """
    executor = ParallelExecutor(max_workers=max_workers, progress=progress)
    return executor.run_batch(func, list(inputs))


def parallel_for(
    func: Callable[[T], Any],
    inputs: Iterable[T],
    max_workers: int | None = None,
    progress: bool = False,
) -> int:
    """Execute function on all inputs, ignoring results.

    Useful for side-effect operations like file processing.

    Args:
        func: Function to execute on each input.
        inputs: Inputs to process.
        max_workers: Number of workers (default: CPU count).
        progress: Show progress bar.

    Returns:
        Number of successful executions.

    Example::

        # Process files in parallel
        success_count = parallel_for(save_result, results, progress=True)

    """
    executor = ParallelExecutor(max_workers=max_workers, progress=progress)
    batch = executor.map(func, list(inputs))
    return batch.completed_jobs
