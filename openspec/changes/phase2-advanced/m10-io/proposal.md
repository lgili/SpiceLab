# M10: Performance & I/O Overhaul

**Status:** Proposed
**Priority:** 🟠 HIGH
**Estimated Duration:** 10-12 weeks
**Dependencies:** M2 (performance baseline), M1 (testing framework)

## Problem Statement

SpiceLab's current I/O implementation loads entire RAW files into memory, doesn't cache efficiently, runs simulations sequentially, and lacks GPU acceleration. This makes it impractical for large circuits, high-resolution simulations, and Monte Carlo analysis at scale.

### Current Gaps
- ❌ RAW file reader loads entire file into memory (OOM for >100MB files)
- ❌ No compressed cache (disk usage grows rapidly)
- ❌ Serial job execution (wastes multi-core CPUs)
- ❌ No GPU acceleration for FFT/signal processing
- ❌ No HDF5 result storage (inefficient for large datasets)
- ❌ No incremental simulation (re-runs unchanged subcircuits)
- ❌ No distributed caching (team collaboration limited)

### Impact
- **Scalability:** Cannot handle large circuits or long simulations
- **Performance:** 10-100x slower than possible
- **Disk Usage:** Terabytes for Monte Carlo campaigns
- **Team Collaboration:** No shared cache for faster builds

## Objectives

1. **Streaming RAW reader** with chunked memory-mapped I/O
2. **Compressed cache** using zstd (90% disk reduction)
3. **Parallel job execution** with ProcessPoolExecutor
4. **GPU acceleration** for FFT using CuPy
5. **HDF5 result storage** for efficient large datasets
6. **Incremental simulation** with dirty checking
7. **Distributed cache** with Redis backend
8. **Target: 100x faster, 90% less disk**

## Technical Design

### 1. Streaming RAW File Reader

```python
# spicelab/io/raw_streaming.py
import mmap
import numpy as np
from pathlib import Path

class StreamingRAWReader:
    """Memory-efficient chunked RAW file reader."""

    def __init__(self, raw_file: Path, chunk_size: int = 10_000):
        self.raw_file = raw_file
        self.chunk_size = chunk_size
        self._mmap = None
        self._file = None

    def __enter__(self):
        self._file = open(self.raw_file, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

    def read_variable_chunked(self, var_name: str, start: int = 0, end: int | None = None):
        """Read variable data in chunks (generator)."""
        # Parse header to find variable offset
        var_info = self._parse_header()[var_name]

        # Read data in chunks
        offset = var_info['offset'] + start * var_info['dtype'].itemsize
        n_points = end - start if end else var_info['n_points'] - start

        for i in range(0, n_points, self.chunk_size):
            chunk_size = min(self.chunk_size, n_points - i)
            chunk_offset = offset + i * var_info['dtype'].itemsize

            # Read chunk from mmap
            self._mmap.seek(chunk_offset)
            chunk_bytes = self._mmap.read(chunk_size * var_info['dtype'].itemsize)
            chunk_data = np.frombuffer(chunk_bytes, dtype=var_info['dtype'])

            yield chunk_data

    def read_to_hdf5(self, output_file: Path, compression='gzip'):
        """Stream RAW to HDF5 with compression."""
        import h5py

        with h5py.File(output_file, 'w') as h5:
            header = self._parse_header()

            for var_name, var_info in header.items():
                # Create dataset with compression
                ds = h5.create_dataset(
                    var_name,
                    shape=(var_info['n_points'],),
                    dtype=var_info['dtype'],
                    compression=compression,
                    chunks=(self.chunk_size,),
                )

                # Stream data
                offset = 0
                for chunk in self.read_variable_chunked(var_name):
                    ds[offset:offset+len(chunk)] = chunk
                    offset += len(chunk)
```

### 2. Compressed Cache with zstd

```python
# spicelab/cache/compressed.py
import zstandard as zstd
import pickle
from pathlib import Path

class CompressedCache:
    """zstd-compressed simulation result cache."""

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".spicelab" / "cache_compressed"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.compressor = zstd.ZstdCompressor(level=3)  # Fast compression
        self.decompressor = zstd.ZstdDecompressor()

    def save(self, key: str, data: xr.Dataset):
        """Save dataset with compression."""
        cache_file = self.cache_dir / f"{key}.zst"

        # Serialize to bytes
        data_bytes = pickle.dumps(data)

        # Compress
        compressed = self.compressor.compress(data_bytes)

        # Write
        cache_file.write_bytes(compressed)

        # Log compression ratio
        compression_ratio = len(compressed) / len(data_bytes)
        print(f"Compressed {len(data_bytes)} -> {len(compressed)} bytes ({compression_ratio:.1%})")

    def load(self, key: str) -> xr.Dataset | None:
        """Load compressed dataset."""
        cache_file = self.cache_dir / f"{key}.zst"

        if not cache_file.exists():
            return None

        # Read compressed
        compressed = cache_file.read_bytes()

        # Decompress
        data_bytes = self.decompressor.decompress(compressed)

        # Deserialize
        dataset = pickle.loads(data_bytes)

        return dataset
```

### 3. Parallel Job Execution

```python
# spicelab/parallel/executor.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

class ParallelSimulationExecutor:
    """Execute simulations in parallel."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or os.cpu_count()

    def run_batch(
        self,
        simulation_fn: Callable,
        parameter_sets: list[dict],
        progress: bool = True,
    ) -> list:
        """Run batch of simulations in parallel."""

        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(simulation_fn, params): i
                for i, params in enumerate(parameter_sets)
            }

            # Collect results
            if progress:
                from tqdm import tqdm
                futures_iter = tqdm(as_completed(futures), total=len(futures))
            else:
                futures_iter = as_completed(futures)

            for future in futures_iter:
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    print(f"Simulation {idx} failed: {e}")
                    results.append((idx, None))

        # Sort by index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

# Usage
executor = ParallelSimulationExecutor(max_workers=8)
results = executor.run_batch(
    simulation_fn=lambda params: run_simulation(circuit, analyses, **params),
    parameter_sets=[{'temp': t} for t in range(-40, 125, 10)],
)
```

### 4. GPU Acceleration with CuPy

```python
# spicelab/gpu/fft.py
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class GPUAcceleratedFFT:
    """GPU-accelerated FFT using CuPy."""

    @staticmethod
    def is_available() -> bool:
        return GPU_AVAILABLE

    @staticmethod
    def fft(data: np.ndarray, use_gpu: bool = True) -> np.ndarray:
        """Compute FFT (GPU if available)."""

        if use_gpu and GPU_AVAILABLE:
            # Transfer to GPU
            data_gpu = cp.asarray(data)

            # Compute FFT on GPU
            fft_gpu = cp.fft.fft(data_gpu)

            # Transfer back to CPU
            fft_result = cp.asnumpy(fft_gpu)
        else:
            # CPU fallback
            fft_result = np.fft.fft(data)

        return fft_result

    @staticmethod
    def benchmark():
        """Benchmark GPU vs CPU FFT."""
        import time

        n = 2**20  # 1M points
        data = np.random.randn(n)

        # CPU
        t0 = time.perf_counter()
        fft_cpu = np.fft.fft(data)
        cpu_time = time.perf_counter() - t0

        # GPU
        if GPU_AVAILABLE:
            data_gpu = cp.asarray(data)
            cp.cuda.Stream.null.synchronize()  # Warmup

            t0 = time.perf_counter()
            fft_gpu = cp.fft.fft(data_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.perf_counter() - t0

            print(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {cpu_time/gpu_time:.1f}x")
        else:
            print(f"CPU: {cpu_time:.3f}s, GPU: Not available")
```

### 5. HDF5 Result Storage

```python
# spicelab/io/hdf5_storage.py
import h5py
import xarray as xr

class HDF5ResultStorage:
    """Efficient HDF5-based result storage."""

    @staticmethod
    def save_dataset(dataset: xr.Dataset, output_file: Path, compression='gzip'):
        """Save xarray Dataset to HDF5."""
        dataset.to_netcdf(output_file, engine='h5netcdf', compression=compression)

    @staticmethod
    def load_dataset(input_file: Path) -> xr.Dataset:
        """Load xarray Dataset from HDF5."""
        return xr.open_dataset(input_file, engine='h5netcdf')

    @staticmethod
    def save_batch(datasets: dict[str, xr.Dataset], output_file: Path):
        """Save multiple datasets to single HDF5 file."""
        with h5py.File(output_file, 'w') as h5:
            for name, dataset in datasets.items():
                group = h5.create_group(name)
                # Save each data variable
                for var_name, data_array in dataset.data_vars.items():
                    group.create_dataset(
                        var_name,
                        data=data_array.values,
                        compression='gzip',
                    )
```

### 6. Incremental Simulation (Dirty Checking)

```python
# spicelab/incremental/dirty_check.py

class IncrementalSimulation:
    """Incremental simulation with change detection."""

    def __init__(self, circuit: Circuit, cache: CompressedCache):
        self.circuit = circuit
        self.cache = cache
        self._circuit_hash = None

    def needs_resimulation(self) -> bool:
        """Check if circuit changed since last simulation."""
        current_hash = self._compute_circuit_hash()

        if self._circuit_hash is None:
            # First run
            self._circuit_hash = current_hash
            return True

        if current_hash != self._circuit_hash:
            self._circuit_hash = current_hash
            return True

        return False

    def run_incremental(self, analyses: list[AnalysisSpec], **kwargs):
        """Run simulation only if needed."""
        cache_key = self._circuit_hash

        # Check cache
        cached_result = self.cache.load(cache_key)
        if cached_result is not None and not self.needs_resimulation():
            print("Using cached result")
            return cached_result

        # Run simulation
        print("Running simulation (circuit changed)")
        result = run_simulation(self.circuit, analyses, **kwargs)

        # Cache result
        self.cache.save(cache_key, result.dataset())

        return result

    def _compute_circuit_hash(self) -> str:
        """Compute hash of circuit state."""
        import hashlib

        netlist = self.circuit.build_netlist()
        return hashlib.sha256(netlist.encode()).hexdigest()
```

### 7. Distributed Cache with Redis

```python
# spicelab/cache/distributed.py
import redis
import pickle

class DistributedCache:
    """Redis-based distributed cache for team collaboration."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)

    def save(self, key: str, data: xr.Dataset, ttl: int = 86400):
        """Save dataset to Redis (with TTL)."""
        data_bytes = pickle.dumps(data)

        # Compress before storing
        import zlib
        compressed = zlib.compress(data_bytes, level=6)

        self.redis_client.setex(key, ttl, compressed)

    def load(self, key: str) -> xr.Dataset | None:
        """Load dataset from Redis."""
        compressed = self.redis_client.get(key)

        if compressed is None:
            return None

        # Decompress
        import zlib
        data_bytes = zlib.decompress(compressed)

        return pickle.loads(data_bytes)
```

## Implementation Plan

### Week 1-2: Streaming RAW Reader
### Week 3-4: Compressed Cache
### Week 5-6: Parallel Execution
### Week 7-8: GPU Acceleration
### Week 9: HDF5 Storage
### Week 10-11: Incremental Simulation
### Week 12: Distributed Cache & Documentation

## Success Metrics

- [ ] **100x faster** for large circuits
- [ ] **90% disk reduction** with compression
- [ ] Streaming RAW reader: <100MB memory for any file size
- [ ] Parallel execution: near-linear scaling to 16 cores
- [ ] GPU FFT: >10x faster than CPU
- [ ] Test coverage: **95%+**

## Dependencies

- M2 (performance baseline)
- M1 (testing)
- zstandard
- h5py
- CuPy (optional, GPU)
- redis (optional, distributed)

## References

- [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
- [zstandard](https://facebook.github.io/zstd/)
- [CuPy](https://cupy.dev/)
- [Redis](https://redis.io/)
