# Unified I/O

spicelab provides unified readers that normalise waveform outputs from
**NGSpice**, **LTspice**, and **Xyce** into a single `xarray.Dataset` structure with
consistent coordinates, canonical signal names, and provenance metadata.

## Goals
- Single entrypoint to parse RAW / PRN / CSV waveforms.
- Canonical coordinates: `time` or `freq`, plus optional `step` dimension for multi-plot (.step) outputs.
- Canonical signal naming: `V(node)`, `I(element)`.
- Metadata for reproducibility: engine, engine_version (when available), netlist hash, analysis arguments, log warnings/errors.

## Public API

```python
from spicelab.io import load_dataset, read_waveform

# High-level (recommended) with enrichment & normalization
ds = load_dataset(
	"run.raw",
	engine="ngspice",
	log="run.log",
	netlist_hash="abc123",
	analysis_args={"tstop": 1e-3},
	allow_binary=True,  # binary RAW now supported
)

# Lower-level dispatch (auto by extension, minimal enrichment)
ds2 = read_waveform("run.prn")
```

### `load_dataset(...)`
Parameters:
- `path`: waveform file (.raw / .prn / .csv)
- `engine`: optional explicit engine name (`ngspice|ltspice|xyce`)
- `log`: ngspice log file to harvest version, warnings, errors
- `netlist_hash`: stable hash of the netlist/circuit (see core hashing API)
- `analysis_args`: dict of parameters (e.g. tstep, tstop, npoints)
- `allow_binary`: if False and RAW detected as binary, raises (set True to parse)

Returns: `xarray.Dataset` with:
- Coordinates: `time` or `freq`, `step` (if multi-plot/.step)
- Data variables: signal traces (float arrays; AC magnitude by default, optional real/imag/phase expansion)
- Rich `attrs` metadata (see table below)

### Complex AC Components
By default AC analyses provide magnitude-only traces (original RAW magnitude). Pass `complex_components` to expand complex parts into additional variables:

```python
# All three canonical components
ds = load_dataset("ac_run.raw", engine="ngspice", complex_components=True, allow_binary=True)
list(ds.data_vars)
# ['V(out)', 'V(out)_real', 'V(out)_imag', 'V(out)_phase_deg', ...]

# Only real + imag (omit phase)
ds = load_dataset("ac_run.raw", complex_components=("real", "imag"))

# Any subset (case-insensitive): {'real','imag','phase'}
```

Rules:
- Magnitude (original value) is always preserved under the base signal name.
- Expanded channels are suffixed with `_real`, `_imag`, `_phase_deg`.
- Phase is in degrees; convert to radians as needed: `np.deg2rad(ds['V(out)_phase_deg'])`.
- Applied components recorded in `ds.attrs['complex_components']`.

### Device Current Normalization Extensions
Additional raw device quantity patterns are canonicalized:

| Raw Pattern | Canonical Output |
|-------------|------------------|
| `@R1[i]` | `I(R1)` |
| `@M1[id]` (drain current) | `Id(M1)` |
| `Ix(U1:...)` | `I(U1:...)` (existing) |

This complements existing voltage/current normalization (`V(node)`, `I(element)`). Drain currents keep the `Id(...)` form to distinguish from total element current when relevant.

### `read_waveform(path)`
Thin dispatch by extension, no metadata enrichment (use when you only need quick parsing).

## Multi-step (.step) Handling
ASCII RAW files with multiple plot blocks are merged along a `step` dimension:
```python
multi = load_dataset("sweep.raw", engine="ngspice")
assert "step" in multi.dims
```
Each original plot contributes one slice along `step`; the independent variable (`time` / `freq`) is shared.

## Normalization Rules
| Input Pattern            | Canonical |
|-------------------------|-----------|
| `v(out)`, `V(out)`      | `V(out)`  |
| `i(R1)`, `I(R1)`        | `I(R1)`   |
| `time`, `t`, `tempo`    | coord `time` |
| `freq`, `f`, `frequency`| coord `freq` |

If a coordinate alias appears only as a data variable it is lifted into a coordinate.

## Provenance Metadata
Attach run context so artifacts are reproducible:
```python
net_hash = "abc123ef45"  # produced by circuit_hash(circuit)
tran = load_dataset("rc_tran.raw", engine="ngspice", netlist_hash=net_hash, analysis_args={"tstop": 2e-3})
print(tran.attrs["netlist_hash"], tran.attrs["analysis_args"])  # => abc123ef45 {"tstop": 0.002}
```

## Helpers & Conversions
Waveforms can be converted, serialized and introspected via helper functions:

```python
from spicelab.io import (
	to_polars, to_pandas, dataset_to_long_polars,
	save_dataset, load_saved_dataset, list_signals
)

pl_df = to_polars(ds)                  # Wide form (one column per signal)
pd_df = to_pandas(ds)                  # Pandas wide form
long_df = dataset_to_long_polars(ds)   # Melted: coord columns + (signal,value)

save_dataset(ds, "waveform.nc")        # NetCDF
save_dataset(ds, "waveform.parquet", format="parquet")
reloaded = load_saved_dataset("waveform.nc")

sig_classes = list_signals(ds)
print(sig_classes["voltage"], sig_classes["current"])  # classification
```

### Metadata Attributes
The loader enriches datasets with provenance & structural metadata when available:

| Key | Description |
|-----|-------------|
| `engine` | Detected or provided engine (`ngspice`, `ltspice`, `xyce`) |
| `engine_version` | Heuristic or log-derived version string |
| `analysis` | Raw analysis name (e.g. `tran`, `ac`) inferred from plot header |
| `analysis_kind` | Canonical analysis alias (mirrors `analysis` today) |
| `raw_path` | Source path of waveform file |
| `stepped` | Boolean: multi-plot (.step) merged dataset |
| `steps` | Total number of steps (if stepped) |
| `step_index` | Index of this slice inside a multi-step set (per-plot before merge) |
| `netlist_hash` | Stable circuit/netlist hash provided by caller |
| `analysis_args` | Dict of simulation parameters (user supplied) |
| `log_warnings` / `log_errors` | Parsed ngspice log diagnostics |
| `provenance` | JSON blob with load timestamp (and future lineage) |
| `n_points` | Number of sample points in primary dimension |
| `current_signals` | List of canonical current signal names (`I(...)`) |
| `subckt_currents` | Subcircuit current subset (normalized `Ix(...) -> I(...)`) |
| `time_offset` | Offset extracted from LTspice binary header (if present) |
| `command` | LTspice original run command (binary RAW header) |
| `backannotation` | Backannotation info from LTspice header (if present) |
| `title` / `plot_title` | Original plot title text (when provided) |

Missing keys simply mean the information was not derivable for that file.

### Example End-to-End (doctest style)
```python
>>> from spicelab.io import load_dataset, to_pandas
>>> ds = load_dataset("tests/fixtures/rc_tran_ng.raw", engine="ngspice")
>>> "time" in ds.coords
True
>>> sorted(v for v in ds.data_vars)  # signals
['I(R1)', 'V(out)']
>>> df = to_pandas(ds)
>>> list(df.columns)[:3]
['time', 'I(R1)', 'V(out)']  # order may vary slightly
```
(Upcoming: coords will become index columns automatically.)

## Binary RAW Support
Both ASCII and binary LTspice/ngspice RAW formats are supported:

- UTF-8 or UTF-16 ("wide") headers auto-detected (LTspice sometimes emits UTF‑16 with NULL bytes).
- Binary payload layout is heuristically detected (point-major vs variable-major) and scalar size (float32 vs float64) inferred from file length.
- AC analyses with complex traces retain magnitude data by default; optional real/imag/phase channels via `complex_components`.
- Subcircuit currents `Ix(U1:...)` are normalized to canonical `I(U1:...)`.

Set `allow_binary=True` in `load_dataset` for clarity (default is False for explicit opt-in; lower-level `read_waveform` defaults to True).

## Parity Tests
Fixtures in `tests/fixtures` exercise transient & AC outputs (ASCII + binary) synthesized for ngspice, LTspice and Xyce. A relaxed tolerance (2–15%) validates expected numeric proximity while allowing formatting differences.

## Roadmap (Next Enhancements)
- Refine engine version extraction (LTspice, Xyce) with stricter regex and caching.
- Promote coords as index / multi-index in `to_pandas` / `to_polars`.
- Noise / distortion / spectral density ingestion.
- Additional export formats & compression knobs.
- Optional radians phase output (`_phase_rad`) toggle.

## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| `NotImplementedError: Binary RAW detected` | `allow_binary=False` guard triggered | Re-run with `allow_binary=True` or call `read_waveform` directly |
| Missing `time` coord | Alias only present as variable (`t`, etc.) | Use `load_dataset` (auto promotes) |
| Inconsistent variable names across engines | Raw headers differ | Use canonical names after normalization (`V(node)`, `I(element)`) |
| Garbled header characters | UTF‑16 LTspice header | Auto-detected; if persists report with sample file |
| Unexpected small/half data length | Scalar size/layout heuristic mismatch | File may be truncated; open an issue with the RAW file |

---
For deeper provenance link the dataset hash to your run cache using `netlist_hash` plus analysis args.

## API Reference
See the auto-generated API docs for details:

* [IO Reference](reference/io.md)
