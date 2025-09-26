"""Closed-loop control example using the shared ngspice backend.

This script shows how to run a transient analysis where a Python controller
adjusts a voltage source at every time step based on the measured output. The
controller uses the "external_sources" hook exposed by ``NgSpiceSharedSimulator``
and a simple uniform ADC helper to quantise the measurement.

Prerequisites
-------------
Install ngspice with shared-library support (macOS example)::

    brew install ngspice

and export the path to the shared library before running the script::

    export SPICELAB_NGSPICE_SHARED="$(brew --prefix ngspice)/lib/libngspice.dylib"

Run the example with::

    python examples/closed_loop.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines.ngspice_shared import NgSpiceSharedSimulator
from spicelab.extensions import UniformADC

TARGET_VOLTAGE = 0.6  # volts
SUPPLY_MAX = 1.2  # volts
INTEGRAL_GAIN = 1_200.0  # simple integral controller gain (1/s)
SIM_STOP = 20e-3  # seconds


def build_loop() -> Circuit:
    circuit = Circuit("closed_loop_once")
    node_out = Net("out")

    ctrl = Vdc("DRV", 0.0)
    rout = Resistor("ROUT", "2k")
    cload = Capacitor("CLOAD", "1u")

    circuit.add(ctrl, rout, cload)

    circuit.connect(ctrl.ports[0], rout.ports[0])
    circuit.connect(rout.ports[1], node_out)
    circuit.connect(cload.ports[0], node_out)
    circuit.connect(ctrl.ports[1], GND)
    circuit.connect(cload.ports[1], GND)

    return circuit


@dataclass
class LoopController:
    adc: UniformADC
    target: float
    integral_gain: float
    limits: tuple[float, float]
    integral: float = 0.0
    last_time: float = 0.0
    last_drive: float = 0.0

    def __call__(self, request) -> float:
        vout = request.values.get("V(out)", request.values.get("V(1)", 0.0))
        if not math.isfinite(vout):
            vout = 0.0
        _code, measured = self.adc.sample(vout)

        dt = max(request.time - self.last_time, 0.0)
        error = self.target - measured
        self.integral += error * dt

        drive = self.integral_gain * self.integral
        drive = min(max(drive, self.limits[0]), self.limits[1])

        self.last_drive = drive
        self.last_time = request.time
        return drive


def main() -> None:
    circuit = build_loop()
    simulator = NgSpiceSharedSimulator()

    controller = LoopController(
        adc=UniformADC(bits=10, vref=SUPPLY_MAX),
        target=TARGET_VOLTAGE,
        integral_gain=INTEGRAL_GAIN,
        limits=(0.0, SUPPLY_MAX),
    )

    log: dict[str, list[float]] = {"time": [], "V(out)": [], "drive": []}

    def on_tran_point(point) -> None:
        last = log["time"][-1] if log["time"] else -float("inf")
        if point.time - last >= SIM_STOP / 10:
            log["time"].append(point.time)
            log["V(out)"].append(point.values.get("V(out)", np.nan))
            log["drive"].append(controller.last_drive)

    def control_fn(request, _ctrl=controller):
        return _ctrl(request)

    analysis = AnalysisSpec(
        "tran",
        {
            "tstep": 1e-5,
            "tstop": SIM_STOP,
            "callbacks": {"on_tran_point": on_tran_point},
            "external_sources": {"VDRV": control_fn},
        },
    )

    handle = simulator.run(circuit, [analysis])
    dataset = handle.dataset()

    print("Metadata:")
    print(handle.attrs())
    print("\nFinal samples:")
    print(dataset[["time", "V(out)"]].tail())

    if log["time"]:
        print("\nController snapshots:")
        for t, v, u in zip(log["time"], log["V(out)"], log["drive"], strict=False):
            print(f"t={t:.6e}s  V(out)={v:.3f} V  drive={u:.3f} V")


if __name__ == "__main__":
    main()
