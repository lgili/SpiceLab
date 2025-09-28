"""Middlebrook-style loop stability using ngspice with series AC injection.

We build a simple Gm-C unity-feedback loop and break it with a series AC source
inserted between the error summing node and the plant control input. The input
source is set to AC=0 so the only small-signal excitation is the injection.
We then measure H(jw) ≈ return ratio as V(out)/V(verr) and derive PM/GBW/GM.

Run from repo root:

  uv run --active python examples/ngspice_middlebrook_loop_stability.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from spicelab.analysis import GainBandwidthSpec, GainMarginSpec, PhaseMarginSpec, measure
from spicelab.core.circuit import Circuit
from spicelab.core.components import VA, VCCS, VCVS, C
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines.factory import create_simulator


def build_loop_with_injection(gm_s: float = 1e-3, c_f: float = 1e-6) -> Circuit:
    n_vin = Net("vin")
    n_out = Net("out")
    n_err = Net("verr")
    n_ctrl = Net("vctrl")

    circ = Circuit("gm_c_middlebrook")

    # Vin with AC=0 (only DC bias if any); small-signal comes from injection source
    vin = VA(ac_mag=0.0)
    circ.add(vin)
    circ.connect(vin.ports[0], n_vin)
    circ.connect(vin.ports[1], GND)

    # Error generator: Verr = Vin - Vout (VCVS unity)
    verr = VCVS(ref="ERR", gain=1.0)
    circ.add(verr)
    circ.connect(verr.ports[0], n_err)  # output to error node
    circ.connect(verr.ports[1], GND)
    circ.connect(verr.ports[2], n_vin)
    circ.connect(verr.ports[3], n_out)

    # Series AC injection between error node and plant control node
    inj = VA(ac_mag=1.0)
    circ.add(inj)
    circ.connect(inj.ports[0], n_err)
    circ.connect(inj.ports[1], n_ctrl)

    # Plant: VCCS (gm) drives output from control (vctrl)
    gm = VCCS(ref="G1", gm=str(gm_s))
    circ.add(gm)
    circ.connect(gm.ports[0], n_out)
    circ.connect(gm.ports[1], GND)
    circ.connect(gm.ports[2], n_ctrl)  # control +: vctrl (after injection)
    circ.connect(gm.ports[3], GND)  # control -: ground (using Verr sign via VCVS)

    # Output capacitor (dominant pole)
    cap = C(c_f)
    circ.add(cap)
    circ.connect(cap.ports[0], n_out)
    circ.connect(cap.ports[1], GND)

    return circ


def maybe_plot(freq: np.ndarray, H: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    mag_db = 20 * np.log10(np.abs(H))
    phase_deg = np.unwrap(np.angle(H)) * (180.0 / np.pi)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    ax1.semilogx(freq, mag_db)
    ax1.axhline(0.0, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("|H| [dB]")
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax2.semilogx(freq, phase_deg)
    ax2.axhline(-180, color="r", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Phase [deg]")
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "middlebrook_loop_bode.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    gm = 1e-3
    c = 1e-6
    circ = build_loop_with_injection(gm, c)
    fc = gm / (2.0 * np.pi * c)
    fstart = max(1.0, fc / 100.0)
    fstop = fc * 100.0
    ac = AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": fstart, "fstop": fstop})

    sim = create_simulator("ngspice")
    handle = sim.run(circ, [ac], None, None)
    ds = handle.dataset()

    # Treat measured return ratio H ≈ V(out)/V(verr)
    rows = measure(
        ds,
        [
            PhaseMarginSpec(name="pm", numerator="V(out)", denominator="V(verr)"),
            GainBandwidthSpec(name="gbw", numerator="V(out)", denominator="V(verr)"),
            GainMarginSpec(
                name="gm", numerator="V(out)", denominator="V(verr)", tolerance_deg=20.0
            ),
        ],
        return_as="python",
    )
    rows_by = {r["measure"]: r for r in rows}
    pm = rows_by["pm"]["value"]
    gbw = rows_by["gbw"]["value"]
    gm_db = rows_by["gm"]["value"]
    print(f"Middlebrook (series AC) metrics: PM={pm:.1f} deg, GBW≈{gbw:.1f} Hz, GM={gm_db}")

    # Optional bode plot
    try:
        vout = np.asarray(ds["V(out)"].values)
        verr = np.asarray(ds["V(verr)"].values)
        H = vout / verr
        freq = np.asarray(ds["V(out)"].coords["freq"].values)
        maybe_plot(freq, H, Path("examples_output"))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
