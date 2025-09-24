# IO API

Unified I/O layer: parsing waveform outputs into `xarray.Dataset` objects.

## Low-level Parsers
::: spicelab.io.raw_reader
    options:
      show_root_heading: false

## Unified Readers Facade
::: spicelab.io.readers.load_dataset
::: spicelab.io.readers.read_waveform
::: spicelab.io.readers.read_ngspice_raw
::: spicelab.io.readers.read_ngspice_raw_multi
::: spicelab.io.readers.read_xyce_prn
::: spicelab.io.readers.read_ltspice_raw

## Normalization & Helpers
::: spicelab.io.readers.normalize_dataset
::: spicelab.io.readers.to_pandas
::: spicelab.io.readers.to_polars
::: spicelab.io.readers.dataset_to_long_polars
::: spicelab.io.readers.save_dataset
::: spicelab.io.readers.load_saved_dataset
::: spicelab.io.readers.list_signals

----
Binary RAW parsing, UTF‑16 header detection, complex AC component expansion (`complex_components`), device current normalization extensions (@R1[i] -> I(R1), @M1[id] -> Id(M1)), classification helpers & persistence utilities are integrated; upcoming work will focus on refined engine version heuristics, noise/distortion ingestion, and coordinate indexing improvements.
