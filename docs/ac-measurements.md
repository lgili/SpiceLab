# AC measurements

This page summarizes small-signal AC measurements available in `spicelab.analysis.measure` and how to consume results without optional DataFrame dependencies.

## Available specs

- PhaseMarginSpec — Phase margin at unity-gain crossover of H = numerator/denominator. Returns PM = 180° + angle(H) at |H| ≈ 1.
- GainBandwidthSpec — Unity-gain frequency (Hz) for H = numerator/denominator. For open-loop A, this is the GBW.
- GainMarginSpec — Gain margin (dB) at the phase crossing near −180°. If no phase sample is within a tolerance band of −180°, returns +∞.

## Usage

Evaluation returns a table. By default it uses Polars when available; for light environments (tests, scripts) you can request a plain Python result:

```python
from spicelab.analysis.measure import (
    measure,
    PhaseMarginSpec, GainBandwidthSpec, GainMarginSpec,
)

rows = measure(
    ds,
    [
        PhaseMarginSpec(name="pm", numerator="vout", denominator="vin"),
        GainBandwidthSpec(name="gbw", numerator="vout", denominator="vin"),
        GainMarginSpec(name="gm", numerator="vout", denominator="vin", tolerance_deg=15.0),
    ],
    return_as="python",
)
# rows is a list[dict]
rows_by_name = {r["measure"]: r for r in rows}
pm_deg = rows_by_name["pm"]["value"]
gbw_hz = rows_by_name["gbw"]["value"]
gm_db = rows_by_name["gm"]["value"]
```

## Notes

- The extractor preserves complex values when available (AC analysis), ensuring accurate magnitude and phase computations.
- Tolerance for `GainMarginSpec` defaults to ±15° around −180°; adjust via `tolerance_deg` to match your sweep resolution.
- If you prefer a DataFrame, omit `return_as` and ensure `polars` is installed; you’ll get a `polars.DataFrame` with the same columns.
