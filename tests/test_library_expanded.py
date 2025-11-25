"""Tests for the expanded component library (M3).

Tests cover:
- Diode catalog (signal, rectifier, Schottky, Zener, LEDs)
- BJT catalog (NPN, PNP, Darlington)
- MOSFET catalog (N-channel, P-channel, small-signal, power)
- Behavioral models (ideal diodes, ideal switches, transformers)
"""

from __future__ import annotations

from spicelab.library import create_component, get_component_spec, list_components
from spicelab.library.behavioral import (
    _IDEAL_DIODES,
    _IDEAL_SWITCHES,
    _TRANSFORMERS,
)
from spicelab.library.diodes import _COMMON_DIODES
from spicelab.library.transistors import _BJTS, _MOSFETS


class TestLibraryOverview:
    """Test overall library structure."""

    def test_total_components_count(self) -> None:
        """Library should have at least 60 components."""
        comps = list_components()
        assert len(comps) >= 60

    def test_categories_exist(self) -> None:
        """Library should have expected categories."""
        comps = list_components()
        categories = {c.category for c in comps}
        expected = {"diode", "bjt", "mosfet", "capacitor", "resistor", "opamp", "behavioral"}
        assert expected.issubset(categories)

    def test_diode_count(self) -> None:
        """Should have at least 15 diodes."""
        comps = list_components()
        diodes = [c for c in comps if c.category == "diode"]
        assert len(diodes) >= 15

    def test_bjt_count(self) -> None:
        """Should have at least 10 BJTs."""
        comps = list_components()
        bjts = [c for c in comps if c.category == "bjt"]
        assert len(bjts) >= 10

    def test_mosfet_count(self) -> None:
        """Should have at least 10 MOSFETs."""
        comps = list_components()
        mosfets = [c for c in comps if c.category == "mosfet"]
        assert len(mosfets) >= 10


class TestDiodeCatalog:
    """Test diode catalog."""

    def test_diode_entries_count(self) -> None:
        """Should have expanded diode entries."""
        assert len(_COMMON_DIODES) >= 15

    def test_signal_diodes_exist(self) -> None:
        """Signal diodes should be registered."""
        spec = get_component_spec("diode.1n4148")
        assert spec is not None
        assert "switching" in spec.metadata.get("description", "").lower()

    def test_rectifier_diodes_exist(self) -> None:
        """Rectifier diodes should be registered."""
        spec = get_component_spec("diode.1n4007")
        assert spec is not None
        assert "rectifier" in spec.metadata.get("description", "").lower()

    def test_schottky_diodes_exist(self) -> None:
        """Schottky diodes should be registered."""
        spec = get_component_spec("diode.1n5819")
        assert spec is not None
        assert "schottky" in spec.metadata.get("description", "").lower()

    def test_zener_diodes_exist(self) -> None:
        """Zener diodes should be registered."""
        spec = get_component_spec("diode.1n4733a")
        assert spec is not None
        assert "zener" in spec.metadata.get("description", "").lower()

    def test_led_diodes_exist(self) -> None:
        """LED models should be registered."""
        spec = get_component_spec("diode.led_red")
        assert spec is not None
        assert "led" in spec.metadata.get("description", "").lower()

    def test_diode_model_cards(self) -> None:
        """All diodes should have model cards."""
        for entry in _COMMON_DIODES:
            assert entry.model_card is not None
            assert entry.model_card.startswith(".model")

    def test_create_diode_component(self) -> None:
        """Should be able to create diode from catalog."""
        diode = create_component("diode.1n4148", "1")
        assert diode is not None
        assert diode.value == "D1N4148"


class TestBjtCatalog:
    """Test BJT catalog."""

    def test_bjt_entries_count(self) -> None:
        """Should have expanded BJT entries."""
        assert len(_BJTS) >= 10

    def test_npn_transistors_exist(self) -> None:
        """NPN transistors should be registered."""
        npn_parts = ["2n2222", "2n3904", "bc547b"]
        for part in npn_parts:
            spec = get_component_spec(f"bjt.{part}")
            assert spec is not None, f"Missing: bjt.{part}"
            assert spec.metadata.get("type") == "npn"

    def test_pnp_transistors_exist(self) -> None:
        """PNP transistors should be registered."""
        pnp_parts = ["2n2907", "2n3906", "bc557"]
        for part in pnp_parts:
            spec = get_component_spec(f"bjt.{part}")
            assert spec is not None, f"Missing: bjt.{part}"
            assert spec.metadata.get("type") == "pnp"

    def test_darlington_transistors_exist(self) -> None:
        """Darlington transistors should be registered."""
        spec_npn = get_component_spec("bjt.tip120")
        spec_pnp = get_component_spec("bjt.tip125")
        assert spec_npn is not None
        assert spec_pnp is not None
        assert "darlington" in spec_npn.metadata.get("description", "").lower()

    def test_bjt_model_cards(self) -> None:
        """All BJTs should have model cards."""
        for entry in _BJTS:
            assert entry.model_card is not None
            assert entry.model_card.startswith(".model")
            assert "NPN" in entry.model_card or "PNP" in entry.model_card

    def test_create_bjt_component(self) -> None:
        """Should be able to create BJT from catalog."""
        bjt = create_component("bjt.2n2222", "1")
        assert bjt is not None
        assert bjt.value == "Q2N2222"


class TestMosfetCatalog:
    """Test MOSFET catalog."""

    def test_mosfet_entries_count(self) -> None:
        """Should have expanded MOSFET entries."""
        assert len(_MOSFETS) >= 10

    def test_n_channel_mosfets_exist(self) -> None:
        """N-channel MOSFETs should be registered."""
        n_parts = ["2n7000", "bss138", "irfz44n", "irf540n"]
        for part in n_parts:
            spec = get_component_spec(f"mosfet.{part}")
            assert spec is not None, f"Missing: mosfet.{part}"
            assert spec.metadata.get("polarity") == "n-channel"

    def test_p_channel_mosfets_exist(self) -> None:
        """P-channel MOSFETs should be registered."""
        p_parts = ["bs250", "irf9540n", "ao3401a"]
        for part in p_parts:
            spec = get_component_spec(f"mosfet.{part}")
            assert spec is not None, f"Missing: mosfet.{part}"
            assert spec.metadata.get("polarity") == "p-channel"

    def test_logic_level_mosfets_exist(self) -> None:
        """Logic-level gate MOSFETs should be registered."""
        spec = get_component_spec("mosfet.irlz44n")
        assert spec is not None
        assert "logic" in spec.metadata.get("description", "").lower()

    def test_mosfet_model_cards(self) -> None:
        """All MOSFETs should have model cards."""
        for entry in _MOSFETS:
            assert entry.model_card is not None
            assert entry.model_card.startswith(".model")
            assert "NMOS" in entry.model_card or "PMOS" in entry.model_card

    def test_create_mosfet_component(self) -> None:
        """Should be able to create MOSFET from catalog."""
        mosfet = create_component("mosfet.2n7000", "1")
        assert mosfet is not None
        assert mosfet.value == "M2N7000"


class TestBehavioralModels:
    """Test behavioral model catalog."""

    def test_ideal_diodes_entries(self) -> None:
        """Should have ideal diode entries."""
        assert len(_IDEAL_DIODES) >= 2

    def test_ideal_switches_entries(self) -> None:
        """Should have ideal switch entries."""
        assert len(_IDEAL_SWITCHES) >= 2

    def test_transformer_entries(self) -> None:
        """Should have transformer info entries."""
        assert len(_TRANSFORMERS) >= 2

    def test_ideal_diode_registered(self) -> None:
        """Ideal diode should be registered."""
        spec = get_component_spec("behavioral.diode.ideal")
        assert spec is not None
        assert "ideal" in spec.metadata.get("description", "").lower()

    def test_ideal_switch_registered(self) -> None:
        """Ideal switch should be registered."""
        spec = get_component_spec("behavioral.switch.ideal")
        assert spec is not None
        assert "ideal" in spec.metadata.get("description", "").lower()

    def test_transformer_info_registered(self) -> None:
        """Transformer info should be registered."""
        spec = get_component_spec("behavioral.transformer.ideal_1to1")
        assert spec is not None
        assert spec.metadata.get("turns_ratio") == 1.0

    def test_create_ideal_diode(self) -> None:
        """Should be able to create ideal diode."""
        diode = create_component("behavioral.diode.ideal", "1")
        assert diode is not None
        assert diode.value == "D_IDEAL"

    def test_create_ideal_switch(self) -> None:
        """Should be able to create ideal switch."""
        switch = create_component("behavioral.switch.ideal", "1")
        assert switch is not None
        assert switch.value == "SW_IDEAL"


class TestComponentMetadata:
    """Test that components have proper metadata."""

    def test_all_components_have_category(self) -> None:
        """All components should have a category."""
        for comp in list_components():
            assert comp.category is not None
            assert len(comp.category) > 0

    def test_diodes_have_model_card(self) -> None:
        """All diodes should have model_card in metadata."""
        for comp in list_components():
            if comp.category == "diode":
                assert "model_card" in comp.metadata or "model" in comp.metadata

    def test_bjts_have_type_metadata(self) -> None:
        """All BJTs should have type (npn/pnp) in metadata."""
        for comp in list_components():
            if comp.category == "bjt":
                assert comp.metadata.get("type") in ("npn", "pnp")

    def test_mosfets_have_polarity_metadata(self) -> None:
        """All MOSFETs should have polarity in metadata."""
        for comp in list_components():
            if comp.category == "mosfet":
                assert comp.metadata.get("polarity") in ("n-channel", "p-channel")
