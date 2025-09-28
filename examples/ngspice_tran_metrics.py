"""Transient metrics (rise time, THD, ENOB) using ngspice on simple setups.

Run from repo root:

  uv run --active python examples/ngspice_tran_metrics.py

Requires ngspice on PATH.
"""

from __future__ import annotations

from spicelab.analysis import ENOBSpec, RiseTimeSpec, THDSpec, measure
from spicelab.core.circuit import Circuit
from spicelab.core.components import VP, VSIN_T, Capacitor, Resistor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines.factory import create_simulator


def build_rc_step() -> Circuit:
    c = Circuit("rc_step")
    vin = VP(v1=0.0, v2=1.0, td=0.0, tr=1e-6, tf=1e-6, pw=1e-3, per=2e-3)
    r = Resistor("1", "1k")
    cap = Capacitor("1u")
    c.add(vin, r, cap)
    c.connect(vin.ports[0], r.ports[0])
    c.connect(r.ports[1], cap.ports[0])
    c.connect(cap.ports[1], GND)
    c.connect(vin.ports[1], GND)
    return c


def build_sine(fs: float = 1000.0, amp_v: float = 1.0) -> Circuit:
    """Single-tone voltage source into a resistive load.

    Uses a typed SIN source so ngspice transient can generate a steady sine.
    """
    c = Circuit("sine_tone")
    # VSIN_T(vo, va, freq) — zero DC offset, amplitude=amp_v (peak), freq in Hz
    vsin = VSIN_T(vo=0.0, va=amp_v, freq=fs)
    rload = Resistor("1", "1k")
    c.add(vsin, rload)
    # Node 'out' at source positive terminal; load to ground
    from spicelab.core.net import Net

    n_out = Net("out")
    c.connect(vsin.ports[0], n_out)
    c.connect(vsin.ports[1], GND)
    c.connect(rload.ports[0], n_out)
    c.connect(rload.ports[1], GND)
    return c


def main() -> int:
    sim = create_simulator("ngspice")
    # Rise time on RC charge
    circuit = build_rc_step()
    tran = AnalysisSpec("tran", {"tstep": 1e-6, "tstop": 2e-3})
    handle = sim.run(circuit, [tran], None, None)
    ds = handle.dataset()
    # Auto-discover a reasonable V(node) for rise-time
    try:
        import xarray as xr  # type: ignore

        def _candidate_voltage_signals(dataset: xr.Dataset) -> list[str]:
            keys = [str(k) for k in dataset.data_vars]
            return [k for k in keys if k.startswith("V(") and k.endswith(")")]

        vnames = _candidate_voltage_signals(ds)
        chosen: str | None = None
        # Prefer names containing common "out" aliases, avoid obvious inputs
        prefer = ("out", "vout", "vo")
        avoid = ("vin", "in", "source", "verr")
        scored: list[tuple[int, float, str]] = []
        for name in vnames:
            low = name.lower()
            score = 0
            if any(p in low for p in prefer):
                score += 2
            if any(a in low for a in avoid):
                score -= 2
            try:
                y = ds[name].values
                delta = float(abs(y[-1] - y[0])) if y.size >= 2 else 0.0
            except Exception:
                delta = 0.0
            scored.append((score, delta, name))
        if scored:
            # primary: score desc; secondary: delta desc
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            chosen = scored[0][2]
        if not chosen and vnames:
            chosen = vnames[0]
        if chosen:
            rows = measure(ds, [RiseTimeSpec(name="tr", signal=chosen)], return_as="python")
            tr = rows[0]["value"]
            print(f"RC rise time (10-90): {tr:.6e} s at {chosen}")
    except Exception:
        # Fall back silently if xarray missing or unexpected ds shape
        pass

    # THD/ENOB on a proper sine tone
    try:
        fs = 1000.0  # Hz
        sine_circ = build_sine(fs=fs, amp_v=1.0)
        # Sample fast enough and long enough (several cycles)
        tstep = 2e-6  # 500 kS/s
        tstop = 10e-3  # 10 ms ~ 10 cycles @ 1 kHz
        tran2 = AnalysisSpec("tran", {"tstep": tstep, "tstop": tstop})
        h2 = sim.run(sine_circ, [tran2], None, None)
        ds2 = h2.dataset()
        # Typical node naming from engine: V(out)
        rows = measure(
            ds2,
            [
                THDSpec(name="thd", signal="V(out)", harmonics=5, f0=fs),
                ENOBSpec(name="enob", signal="V(out)", harmonics=5, f0=fs),
            ],
            return_as="python",
        )
        rb = {r["measure"]: r for r in rows}
        thd = rb["thd"]["value"]
        enob = rb["enob"]["value"]
        print(f"Sine tone metrics @ {fs:.0f} Hz: THD={thd:.3f}%  ENOB={enob:.2f} bits")
    except KeyError:
        # If node names differ, ignore gracefully
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
