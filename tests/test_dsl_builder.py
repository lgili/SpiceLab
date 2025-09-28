from __future__ import annotations

from spicelab.dsl import CircuitBuilder


def test_circuit_builder_basic_rc() -> None:
    builder = CircuitBuilder("rc_lowpass")
    builder.net("vin")
    builder.net("vout")
    builder.vdc("vin", "gnd", value="5")
    builder.resistor("vin", "vout", value="1k")
    builder.capacitor("vout", "0", value="100n")

    circuit = builder.build()
    df = circuit.connectivity_dataframe()

    assert set(df["component"]) == {"V1", "R1", "C1"}
    nets = df[df["component"] == "R1"]["net"].tolist()
    assert "vin" in nets and "vout" in nets
    assert "0" in df[df["component"] == "C1"]["net"].tolist()


def test_circuit_builder_bus_alias() -> None:
    builder = CircuitBuilder("pi_filter")
    _nodes = builder.bus("n", 3)  # n0, n1, n2
    builder.alias("n0", "vin")
    builder.alias("n2", "vout")
    builder.vdc("vin", "0", value="12")
    builder.inductor("n0", "n1", value="10u")
    builder.capacitor("n1", "0", value="1u")
    builder.inductor("n1", "n2", value="10u")
    builder.capacitor("n2", "0", value="1u")

    circuit = builder.build()
    df = circuit.connectivity_dataframe()
    assert {"n0", "n1", "n2"}.issubset(set(df["net"]))
    assert "vin" in df["net"].values
    assert "vout" in df["net"].values

    table = circuit.summary_table()
    assert "Component" in table
    assert "net" in table.lower()
