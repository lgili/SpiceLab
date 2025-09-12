from __future__ import annotations

from _common import savefig

from cat.analysis import TRAN, ParamGrid, stack_step_to_df, step_grid
from cat.analysis.viz.plot import plot_sweep_df
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND, Net
from cat.spice import ngspice_cli


def build_circuit() -> Circuit:
    # RC parametrizado via .param placeholders: {R}, {C}
    c = Circuit("ex_step_grid_tran")
    V1 = Vdc("1", 1.0)  # 1 V step source on R input
    R1 = Resistor("1", "{R}")
    C1 = Capacitor("1", "{C}")
    c.add(V1, R1, C1)
    vin = Net("vin")
    vout = Net("vout")
    c.connect(V1.ports[0], vin)  # vin
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)  # vout
    c.connect(C1.ports[0], vout)
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


def main() -> None:
    c = build_circuit()
    grid: ParamGrid = {"R": ["1k", "2k", "5k"], "C": ["100n"]}
    res = step_grid(c, grid, analysis_factory=lambda: TRAN("50us", "5ms"), order=["R", "C"])
    df = stack_step_to_df(res, y=["v(vout)"], with_x=True)

    # Plot using helper
    fig = plot_sweep_df(df, x="time", y="v(vout)", hue="R", title="RC step (vary R)")
    savefig(fig, "step_grid.png")

    # cleanup artifacts
    for run in res.runs:
        ngspice_cli.cleanup_artifacts(run.run.artifacts)


if __name__ == "__main__":
    main()
