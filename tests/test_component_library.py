from cat.core.components import Component, OpAmpIdeal, Resistor
from cat.library import (
    ComponentSpec,
    create_component,
    get_component_spec,
    list_components,
    register_component,
    unregister_component,
)
from cat.library.opamps import OpAmpSubckt
from cat.library.transistors import Bjt, Mosfet


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
    from cat.core.components import Capacitor as _Cap

    assert isinstance(c, _Cap)
    assert c.value == "100n"


def test_inductor_and_switch_entries() -> None:
    from cat.core.components import Inductor as _Ind

    ind = create_component("inductor.power_10u_3a", ref="L1")
    assert isinstance(ind, _Ind)

    vsw = create_component("switch.vsw_spst_fast", ref="S1")
    from cat.core.components import VSwitch as _VS

    assert isinstance(vsw, _VS)
    assert vsw.value == "SW_SPST_FAST"

    csw = create_component(
        "switch.isw_spst_sense",
        ref="W1",
        ctrl_vsrc="VCTRL",
    )
    from cat.core.components import ISwitch as _IS

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
    op_ideal = create_component("opamp.ideal", ref="U1")
    assert isinstance(op_ideal, OpAmpIdeal)

    op_lm741 = create_component("opamp.lm741", ref="U2")
    assert isinstance(op_lm741, OpAmpSubckt)
    assert op_lm741.subckt == "LM741"
    spec = get_component_spec("opamp.lm741")
    metadata = spec.metadata or {}
    assert "model_card" in metadata
