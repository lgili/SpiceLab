from __future__ import annotations

import pandas as pd
from _common import savefig
from spicelab.analysis.sweep_grid import run_param_grid
from spicelab.analysis.viz.plot import plot_sweep_df
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.spice import ngspice_cli
from spicelab.spice.base import RunArtifacts


def build_circuit() -> tuple[Circuit, Resistor, Capacitor]:
    # RC parametrizado via .param placeholders: {R}, {C}
    c = Circuit("ex_step_grid_tran")
    V1 = Vdc("VIN", 1.0)  # 1 V step source on R input
    R1 = Resistor("R", "1k")
    C1 = Capacitor("C", "100n")
    c.add(V1, R1, C1)
    vin = Net("vin")
    vout = Net("vout")
    c.connect(V1.ports[0], vin)  # vin
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)  # vout
    c.connect(C1.ports[0], vout)
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c, R1, C1


def main() -> None:
    c, R1, C1 = build_circuit()
    grid = {"R": ["1k", "2k", "5k"], "C": ["100n"]}
    result = run_param_grid(
        circuit=c,
        variables=[(R1, grid["R"]), (C1, grid["C"])],
        analyses=[AnalysisSpec("tran", {"tstep": "50us", "tstop": "5ms"})],
        engine="ngspice",
    )

    frames: list[pd.DataFrame] = []
    for run in result.runs:
        df = run.handle.dataset().to_dataframe().reset_index()
        for key, value in run.combo.items():
            df[key] = value
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Plot using helper
    fig = plot_sweep_df(df, x="time", y="v(vout)", hue="R", title="RC step (vary R)")
    savefig(fig, "step_grid.png")

    # cleanup artifacts
    for run in result.runs:
        attrs = run.handle.attrs()
        art = RunArtifacts(
            netlist_path=str(attrs.get("netlist_path")),
            log_path=str(attrs.get("log_path")),
            raw_path=str(attrs.get("raw_path")),
            workdir=str(attrs.get("workdir")),
        )
        ngspice_cli.cleanup_artifacts(art)


if __name__ == "__main__":
    main()
