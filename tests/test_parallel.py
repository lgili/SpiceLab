"""Tests for the parallel execution module."""

from __future__ import annotations

import time
from typing import Any

import pytest

from spicelab.parallel import (
    BatchResult,
    JobResult,
    JobStatus,
    ParallelExecutor,
)
from spicelab.parallel.executor import parallel_for, parallel_map


def simple_square(x: int) -> int:
    """Simple function for testing."""
    return x * x


def slow_square(x: int) -> int:
    """Slow function for testing parallelism."""
    time.sleep(0.1)
    return x * x


def failing_function(x: int) -> int:
    """Function that always fails."""
    raise ValueError(f"Intentional failure for {x}")


def sometimes_fails(x: int) -> int:
    """Function that fails for even numbers."""
    if x % 2 == 0:
        raise ValueError(f"Failed for even number {x}")
    return x * x


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self) -> None:
        """Test that all status values exist."""
        assert JobStatus.PENDING
        assert JobStatus.RUNNING
        assert JobStatus.COMPLETED
        assert JobStatus.FAILED
        assert JobStatus.CANCELLED


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful result."""
        result = JobResult(
            job_id=0,
            status=JobStatus.COMPLETED,
            value=42,
            duration_ms=100.0,
        )
        assert result.success
        assert not result.failed
        assert result.value == 42

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = JobResult(
            job_id=0,
            status=JobStatus.FAILED,
            error="Something went wrong",
            error_type="ValueError",
        )
        assert not result.success
        assert result.failed
        assert result.error == "Something went wrong"


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_success_rate_all_success(self) -> None:
        """Test success rate when all jobs succeed."""
        results = [
            JobResult(job_id=i, status=JobStatus.COMPLETED, value=i)
            for i in range(10)
        ]
        batch = BatchResult(results=results, total_jobs=10)
        assert batch.success_rate == 1.0

    def test_success_rate_partial(self) -> None:
        """Test success rate with partial success."""
        results = [
            JobResult(job_id=0, status=JobStatus.COMPLETED, value=0),
            JobResult(job_id=1, status=JobStatus.FAILED, error="fail"),
            JobResult(job_id=2, status=JobStatus.COMPLETED, value=2),
            JobResult(job_id=3, status=JobStatus.FAILED, error="fail"),
        ]
        batch = BatchResult(results=results, total_jobs=4)
        assert batch.success_rate == 0.5
        assert batch.completed_jobs == 2
        assert batch.failed_jobs == 2

    def test_get_successful_results(self) -> None:
        """Test extracting successful results."""
        results = [
            JobResult(job_id=0, status=JobStatus.COMPLETED, value=10),
            JobResult(job_id=1, status=JobStatus.FAILED, error="fail"),
            JobResult(job_id=2, status=JobStatus.COMPLETED, value=20),
        ]
        batch = BatchResult(results=results, total_jobs=3)
        successful = batch.get_successful_results()
        assert successful == [10, 20]

    def test_get_failed_jobs(self) -> None:
        """Test extracting failed jobs."""
        results = [
            JobResult(job_id=0, status=JobStatus.COMPLETED, value=10),
            JobResult(job_id=1, status=JobStatus.FAILED, error="error1"),
            JobResult(job_id=2, status=JobStatus.FAILED, error="error2"),
        ]
        batch = BatchResult(results=results, total_jobs=3)
        failed = batch.get_failed_jobs()
        assert len(failed) == 2
        assert failed[0].error == "error1"

    def test_get_result_by_id(self) -> None:
        """Test getting result by job ID."""
        results = [
            JobResult(job_id=0, status=JobStatus.COMPLETED, value=10),
            JobResult(job_id=1, status=JobStatus.COMPLETED, value=20),
        ]
        batch = BatchResult(results=results, total_jobs=2)

        result = batch.get_result_by_id(1)
        assert result is not None
        assert result.value == 20

        assert batch.get_result_by_id(99) is None

    def test_empty_batch(self) -> None:
        """Test empty batch result."""
        batch = BatchResult(results=[], total_jobs=0)
        assert batch.success_rate == 0.0
        assert batch.avg_job_duration_ms == 0.0


class TestParallelExecutor:
    """Tests for ParallelExecutor class."""

    def test_simple_map(self) -> None:
        """Test basic parallel mapping."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        batch = executor.map(simple_square, [1, 2, 3, 4, 5])

        assert batch.total_jobs == 5
        assert batch.completed_jobs == 5
        assert batch.success_rate == 1.0

        values = batch.get_successful_results()
        assert values == [1, 4, 9, 16, 25]

    def test_parallel_speedup(self) -> None:
        """Test that parallel execution is faster than serial."""
        executor = ParallelExecutor(max_workers=4, use_processes=False)

        # This should take ~0.4s with 4 workers vs ~1.6s serial
        inputs = list(range(16))

        start = time.time()
        batch = executor.map(slow_square, inputs)
        duration = time.time() - start

        assert batch.completed_jobs == 16
        # Should be significantly faster than 1.6s
        assert duration < 1.0

    def test_error_handling(self) -> None:
        """Test that errors are caught and reported."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        batch = executor.map(failing_function, [1, 2, 3])

        assert batch.total_jobs == 3
        assert batch.failed_jobs == 3
        assert batch.success_rate == 0.0

        for result in batch.results:
            assert result.failed
            assert "Intentional failure" in (result.error or "")
            assert result.error_type == "ValueError"

    def test_partial_failure(self) -> None:
        """Test mix of successful and failed jobs."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        batch = executor.map(sometimes_fails, [1, 2, 3, 4, 5])

        # Odd numbers succeed, even numbers fail
        assert batch.completed_jobs == 3  # 1, 3, 5
        assert batch.failed_jobs == 2  # 2, 4

        successful = batch.get_successful_results()
        assert sorted(successful) == [1, 9, 25]

    def test_empty_input(self) -> None:
        """Test with empty input list."""
        executor = ParallelExecutor(max_workers=2)
        batch = executor.map(simple_square, [])

        assert batch.total_jobs == 0
        assert batch.completed_jobs == 0

    def test_single_item(self) -> None:
        """Test with single item."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        batch = executor.map(simple_square, [5])

        assert batch.total_jobs == 1
        assert batch.completed_jobs == 1
        assert batch.results[0].value == 25

    def test_run_batch_returns_values(self) -> None:
        """Test run_batch returns just values."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        results = executor.run_batch(simple_square, [1, 2, 3])

        assert results == [1, 4, 9]

    def test_run_batch_with_failures(self) -> None:
        """Test run_batch returns None for failures."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        results = executor.run_batch(sometimes_fails, [1, 2, 3])

        # 1 -> 1, 2 -> None (failed), 3 -> 9
        assert results[0] == 1
        assert results[1] is None
        assert results[2] == 9

    def test_starmap(self) -> None:
        """Test starmap with tuple arguments."""

        def add(a: int, b: int) -> int:
            return a + b

        executor = ParallelExecutor(max_workers=2, use_processes=False)
        batch = executor.starmap(add, [(1, 2), (3, 4), (5, 6)])

        assert batch.completed_jobs == 3
        values = batch.get_successful_results()
        assert values == [3, 7, 11]

    def test_threads_vs_processes(self) -> None:
        """Test that both thread and process modes work."""
        # Thread mode
        executor_threads = ParallelExecutor(max_workers=2, use_processes=False)
        batch_threads = executor_threads.map(simple_square, [1, 2, 3])
        assert batch_threads.completed_jobs == 3

        # Process mode
        executor_procs = ParallelExecutor(max_workers=2, use_processes=True)
        batch_procs = executor_procs.map(simple_square, [1, 2, 3])
        assert batch_procs.completed_jobs == 3

    def test_timing_recorded(self) -> None:
        """Test that timing information is recorded."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        batch = executor.map(slow_square, [1, 2])

        for result in batch.results:
            assert result.duration_ms is not None
            assert result.duration_ms >= 100  # At least 100ms due to sleep
            assert result.start_time is not None
            assert result.end_time is not None

        assert batch.total_duration_ms > 0

    def test_input_params_preserved(self) -> None:
        """Test that input parameters are preserved in results."""
        executor = ParallelExecutor(max_workers=2, use_processes=False)
        inputs = [{"x": 1}, {"x": 2}, {"x": 3}]

        def process(params: dict[str, Any]) -> int:
            return params["x"] * 2

        batch = executor.map(process, inputs)

        for i, result in enumerate(batch.results):
            assert result.input_params == inputs[i]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_parallel_map(self) -> None:
        """Test parallel_map function."""
        results = parallel_map(simple_square, [1, 2, 3, 4], max_workers=2)
        assert results == [1, 4, 9, 16]

    def test_parallel_map_with_failures(self) -> None:
        """Test parallel_map with failures."""
        results = parallel_map(sometimes_fails, [1, 2, 3], max_workers=2)
        assert results[0] == 1
        assert results[1] is None
        assert results[2] == 9

    def test_parallel_for(self) -> None:
        """Test parallel_for function."""
        # parallel_for uses processes by default, which requires pickleable functions
        # Use a module-level function (simple_square) that can be pickled
        success_count = parallel_for(simple_square, [1, 2, 3, 4], max_workers=2)

        assert success_count == 4


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_custom_progress_callback(self) -> None:
        """Test custom progress callback is called."""
        progress_calls: list[tuple[int, int]] = []

        def track_progress(done: int, total: int, result: JobResult[Any] | None) -> None:
            progress_calls.append((done, total))

        executor = ParallelExecutor(max_workers=2, use_processes=False, progress=track_progress)
        batch = executor.map(simple_square, [1, 2, 3, 4])

        assert batch.completed_jobs == 4
        # Should have been called once per completion
        assert len(progress_calls) == 4
        # Last call should be (4, 4)
        assert progress_calls[-1] == (4, 4)

    def test_progress_disabled(self) -> None:
        """Test that progress can be disabled."""
        executor = ParallelExecutor(max_workers=2, use_processes=False, progress=False)
        batch = executor.map(simple_square, [1, 2, 3])

        # Should still work without progress
        assert batch.completed_jobs == 3

    def test_progress_override_in_map(self) -> None:
        """Test progress can be overridden per call."""
        calls: list[int] = []

        def callback(done: int, total: int, result: JobResult[Any] | None) -> None:
            calls.append(done)

        # Default no progress
        executor = ParallelExecutor(max_workers=2, use_processes=False, progress=False)

        # But enable for this call
        executor.map(simple_square, [1, 2, 3], progress=callback)

        assert len(calls) == 3


class TestEdgeCases:
    """Tests for edge cases."""

    def test_large_batch(self) -> None:
        """Test with large number of jobs."""
        executor = ParallelExecutor(max_workers=4, use_processes=False)
        batch = executor.map(simple_square, list(range(100)))

        assert batch.total_jobs == 100
        assert batch.completed_jobs == 100

    def test_worker_count_exceeds_jobs(self) -> None:
        """Test when workers exceed job count."""
        executor = ParallelExecutor(max_workers=10, use_processes=False)
        batch = executor.map(simple_square, [1, 2])

        assert batch.completed_jobs == 2

    def test_single_worker(self) -> None:
        """Test with single worker (serial execution)."""
        executor = ParallelExecutor(max_workers=1, use_processes=False)
        batch = executor.map(simple_square, [1, 2, 3])

        assert batch.completed_jobs == 3
        assert batch.get_successful_results() == [1, 4, 9]
