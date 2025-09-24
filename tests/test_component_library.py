from pathlib import Path

import pytest
from spicelab.core.components import Component, OpAmpIdeal, Resistor
from spicelab.library import (
    ComponentSpec,
    create_component,
    get_component_spec,
    list_components,
    register_component,
    search_components,
    unregister_component,
)
from spicelab.library.opamps import OpAmpSubckt
from spicelab.library.transistors import Bjt, Mosfet


def test_register_and_create_custom_component() -> None:
    created: list[Component] = []

    def factory(ref: str, value: float) -> Component:
        comp = Resistor(ref, value)
        created.append(comp)
        return comp

    register_component("custom.resistor", factory, overwrite=True)
    try:
        comp = create_component("custom.resistor", ref="R99", value=123.0)
        assert isinstance(comp, Resistor)
        assert comp.value == 123.0
        assert created[-1] is comp
    finally:
        unregister_component("custom.resistor")

    with pytest.raises(ValueError):
        register_component(
            "invalid.resistor",
            lambda ref: Resistor(ref, "1k"),
            category="resistor",
            metadata={"tolerance": "1%"},
            overwrite=True,
        )
    with pytest.raises(ValueError):
        register_component(
            "invalid.resistor.extra",
            lambda ref: Resistor(ref, "1k"),
            category="resistor",
            metadata={"value": "1k", "foo": "bar"},
            overwrite=True,
        )


def test_builtin_diode_entry_present() -> None:
    spec: ComponentSpec = get_component_spec("diode.1n4007")
    assert spec.category == "diode"
    diode = create_component("diode.1n4007", ref="D1")
    assert diode.value == "D1N4007"
    metadata = spec.metadata or {}
    assert "model_card" in metadata
    model_value = metadata.get("model")
    assert isinstance(model_value, str)
    assert model_value.startswith("D1N")


def test_list_components_by_category() -> None:
    diodes = list_components(category="diode")
    names = {spec.name for spec in diodes}
    assert "diode.1n4007" in names


def test_resistor_and_capacitor_entries() -> None:
    r_spec = get_component_spec("resistor.e24.1k_1pct_0.25w")
    assert r_spec.category == "resistor"
    r = create_component("resistor.e24.1k_1pct_0.25w", ref="R5")
    assert isinstance(r, Resistor)
    assert r.value == "1k"

    c_spec = get_component_spec("capacitor.ceramic_100n_50v_x7r")
    metadata_cap = c_spec.metadata or {}
    assert metadata_cap.get("dielectric") == "X7R"
    c = create_component("capacitor.ceramic_100n_50v_x7r", ref="C1")
    from spicelab.core.components import Capacitor as _Cap

    assert isinstance(c, _Cap)
    assert c.value == "100n"


def test_inductor_and_switch_entries() -> None:
    from spicelab.core.components import Inductor as _Ind

    ind = create_component("inductor.power_10u_3a", ref="L1")
    assert isinstance(ind, _Ind)

    vsw = create_component("switch.vsw_spst_fast", ref="S1")
    from spicelab.core.components import VSwitch as _VS

    assert isinstance(vsw, _VS)
    assert vsw.value == "SW_SPST_FAST"

    csw = create_component(
        "switch.isw_spst_sense",
        ref="W1",
        ctrl_vsrc="VCTRL",
    )
    from spicelab.core.components import ISwitch as _IS

    assert isinstance(csw, _IS)
    assert csw.value == "ISW_SPST_SLOW"


def test_transistor_entries() -> None:
    mos = create_component("mosfet.bss138", ref="M1")
    assert isinstance(mos, Mosfet)
    assert mos.value == "MBSS138"

    bjt = create_component("bjt.2n3904", ref="Q1")
    assert isinstance(bjt, Bjt)
    assert bjt.value == "Q2N3904"


def test_opamp_entries() -> None:
    from spicelab.core.circuit import Circuit
    from spicelab.library.utils import apply_metadata_to_circuit

    op_ideal = create_component("opamp.ideal", ref="U1")
    assert isinstance(op_ideal, OpAmpIdeal)

    op_lm741 = create_component("opamp.lm741", ref="U2")
    assert isinstance(op_lm741, OpAmpSubckt)
    assert op_lm741.subckt == "LM741"
    spec = get_component_spec("opamp.lm741")
    metadata = spec.metadata or {}
    assert "include" in metadata
    include_value = metadata["include"]
    if isinstance(include_value, str):
        path_obj = Path(include_value)
    elif (
        isinstance(include_value, list | tuple)
        and include_value
        and isinstance(include_value[0], str)
    ):
        path_obj = Path(include_value[0])
    else:
        raise TypeError("metadata['include'] must be a str or a non-empty list/tuple of str")
    assert path_obj.exists()
    circuit = Circuit("opamp-meta")
    added = apply_metadata_to_circuit(circuit, spec)
    assert any(".include" in line for line in added)


def test_search_components_filters() -> None:
    diodes = search_components(category="diode")
    assert any(spec.name == "diode.1n4007" for spec in diodes)

    n_mos = search_components(metadata={"polarity": "n-channel"})
    assert any(spec.name == "mosfet.bss138" for spec in n_mos)

    tl = search_components(name_contains="tl08")
    assert any(spec.name == "opamp.tl081" for spec in tl)
    custom = search_components(predicate=lambda spec: spec.name.endswith("2n3904"))
    assert len(custom) == 1 and custom[0].name == "bjt.2n3904"


def test_protocol_assignment_examples() -> None:
    from spicelab.library import MosfetFactory, ResistorFactory

    def resistor_factory(ref: str, *, value: str | float | None = None) -> Resistor:
        return Resistor(ref, value or "1k")

    def mosfet_factory(ref: str, *, model: str | None = None, params: str | None = None) -> Mosfet:
        return Mosfet(ref, model or "MBSS138", params)

    r_factory: ResistorFactory = resistor_factory
    m_factory: MosfetFactory = mosfet_factory

    assert isinstance(r_factory("Rtest"), Resistor)
    assert isinstance(m_factory("Mtest"), Mosfet)


def test_apply_metadata_to_circuit() -> None:
    from spicelab.core.circuit import Circuit
    from spicelab.library.utils import apply_metadata_to_circuit

    circuit = Circuit("meta")
    spec = get_component_spec("diode.1n4007")
    added = apply_metadata_to_circuit(circuit, spec)
    assert any("D1N4007" in line for line in added)
    # Second call should not duplicate
    added_again = apply_metadata_to_circuit(circuit, spec)
    assert not added_again

    register_component(
        "custom.with_include",
        lambda ref: Resistor(ref, "1k"),
        metadata={"include": "models.lib"},
        overwrite=True,
    )
    try:
        spec_include = get_component_spec("custom.with_include")
        added_include = apply_metadata_to_circuit(circuit, spec_include)
        assert any(line.strip().startswith(".include") for line in added_include)
    finally:
        unregister_component("custom.with_include")
