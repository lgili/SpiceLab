"""Tests for comprehensive SPICE component classes (Phase 1 components)."""

import pytest
from spicelab.core.components import (
    BI,
    BV,
    DZ,
    IPROBE,
    JF,
    JFET,
    MK,
    OLINE,
    TLINE,
    ULINE,
    VPROBE,
    XFMR,
    XSUB,
    BCurrent,
    BVoltage,
    CurrentProbe,
    EExpr,
    GExpr,
    MutualInductance,
    SubcktInstance,
    TLine,
    TLineLossy,
    TLineRC,
    Transformer,
    VCCSExpr,
    VCVSExpr,
    VoltageProbe,
    ZenerDiode,
)


# Helper functions for node mapping (replacing lambdas for E731 compliance)
def _jfet_node_map(p):
    return {"d": "nd", "g": "ng", "s": "ns"}[p.name]


def _zener_node_map(p):
    return {"a": "anode", "c": "cathode"}[p.name]


def _tline_node_map(p):
    return {"p1p": "n1p", "p1n": "n1n", "p2p": "n2p", "p2n": "n2n"}[p.name]


def _tlinerc_node_map(p):
    return {"n1": "in", "n2": "out", "n3": "gnd"}[p.name]


def _bsource_node_map(p):
    return {"p": "out", "n": "gnd"}[p.name]


def _identity_node_map(p):
    return p.name


def _transformer_node_map(p):
    return {"p1": "pri_p", "p2": "pri_n", "s1": "sec_p", "s2": "sec_n"}[p.name]


def _probe_node_map(p):
    return {"p": "sense_p", "n": "sense_n"}[p.name]


class TestJFET:
    """Tests for JFET component."""

    def test_jfet_creation(self) -> None:
        j = JFET("1", model="2N5457")
        assert j.ref == "1"
        assert j.value == "2N5457"  # model stored in value
        assert len(j.ports) == 3
        assert {p.name for p in j.ports} == {"d", "g", "s"}

    def test_jfet_spice_card(self) -> None:
        j = JFET("1", model="2N5457")
        card = j.spice_card(_jfet_node_map)
        assert card == "J1 nd ng ns 2N5457"

    def test_jfet_factory(self) -> None:
        j1 = JF("2N5457")
        j2 = JF("2N5457")
        assert j1.ref != j2.ref  # Auto-increment refs


class TestZenerDiode:
    """Tests for ZenerDiode component."""

    def test_zener_creation(self) -> None:
        z = ZenerDiode("1", model="1N4733")
        assert z.ref == "1"
        assert z.value == "1N4733"  # model stored in value
        assert len(z.ports) == 2
        assert {p.name for p in z.ports} == {"a", "c"}

    def test_zener_spice_card(self) -> None:
        z = ZenerDiode("1", model="1N4733")
        card = z.spice_card(_zener_node_map)
        assert card == "D1 anode cathode 1N4733"

    def test_zener_factory(self) -> None:
        z1 = DZ("1N4733")
        z2 = DZ("1N4733")
        assert z1.ref != z2.ref


class TestMutualInductance:
    """Tests for MutualInductance (K) component."""

    def test_mutual_creation(self) -> None:
        k = MutualInductance("1", l1="L1", l2="L2", coupling=0.99)
        assert k.ref == "1"
        assert k.l1 == "L1"
        assert k.l2 == "L2"
        assert k.coupling == 0.99
        assert len(k.ports) == 0  # K has no physical ports

    def test_mutual_spice_card(self) -> None:
        k = MutualInductance("1", l1="L1", l2="L2", coupling=0.99)
        # K has no ports, so node_map is not used
        card = k.spice_card(_identity_node_map)
        assert card == "K1 L1 L2 0.99"

    def test_mutual_coupling_validation(self) -> None:
        with pytest.raises(ValueError, match="Coupling"):
            MutualInductance("1", l1="L1", l2="L2", coupling=1.5)
        with pytest.raises(ValueError, match="Coupling"):
            MutualInductance("1", l1="L1", l2="L2", coupling=-0.1)

    def test_mutual_factory(self) -> None:
        k1 = MK("L1", "L2", 0.95)
        k2 = MK("L3", "L4", 0.8)
        assert k1.ref != k2.ref


class TestTLine:
    """Tests for lossless transmission line component."""

    def test_tline_creation(self) -> None:
        t = TLine("1", z0=50, td="1n")
        assert t.ref == "1"
        assert t.z0 == 50
        assert t.td == "1n"
        assert len(t.ports) == 4
        assert {p.name for p in t.ports} == {"p1p", "p1n", "p2p", "p2n"}

    def test_tline_spice_card(self) -> None:
        t = TLine("1", z0=50, td="1n")
        card = t.spice_card(_tline_node_map)
        assert card == "T1 n1p n1n n2p n2n Z0=50 TD=1n"

    def test_tline_factory(self) -> None:
        t1 = TLINE(z0=50, td="1n")
        t2 = TLINE(z0=75, td="2n")
        assert t1.ref != t2.ref


class TestTLineLossy:
    """Tests for lossy transmission line (LTRA) component."""

    def test_tline_lossy_creation(self) -> None:
        o = TLineLossy("1", model="MYLINE")
        assert o.ref == "1"
        assert o.value == "MYLINE"  # model stored in value
        assert len(o.ports) == 4

    def test_tline_lossy_spice_card(self) -> None:
        o = TLineLossy("1", model="MYLINE")
        card = o.spice_card(_tline_node_map)
        assert card == "O1 n1p n1n n2p n2n MYLINE"

    def test_tline_lossy_factory(self) -> None:
        o1 = OLINE("LINE1")
        o2 = OLINE("LINE2")
        assert o1.ref != o2.ref


class TestTLineRC:
    """Tests for uniform RC line component."""

    def test_tline_rc_creation(self) -> None:
        u = TLineRC("1", model="RCLINE")
        assert u.ref == "1"
        assert u.value == "RCLINE"  # model stored in value
        assert len(u.ports) == 3
        assert {p.name for p in u.ports} == {"n1", "n2", "n3"}

    def test_tline_rc_spice_card(self) -> None:
        u = TLineRC("1", model="RCLINE")
        card = u.spice_card(_tlinerc_node_map)
        assert card == "U1 in out gnd RCLINE L=1"

    def test_tline_rc_factory(self) -> None:
        u1 = ULINE("RC1")
        u2 = ULINE("RC2")
        assert u1.ref != u2.ref


class TestBVoltage:
    """Tests for behavioral voltage source component."""

    def test_bvoltage_creation(self) -> None:
        b = BVoltage("1", expr="V(in)*2")
        assert b.ref == "1"
        assert b.expr == "V(in)*2"
        assert len(b.ports) == 2
        assert {p.name for p in b.ports} == {"p", "n"}

    def test_bvoltage_spice_card(self) -> None:
        b = BVoltage("1", expr="V(in)*2")
        card = b.spice_card(_bsource_node_map)
        assert card == "B1 out gnd V=V(in)*2"

    def test_bvoltage_conditional_expr(self) -> None:
        b = BVoltage("2", expr="IF(V(ctrl)>2.5, 5, 0)")
        card = b.spice_card(_bsource_node_map)
        assert card == "B2 out gnd V=IF(V(ctrl)>2.5, 5, 0)"

    def test_bvoltage_factory(self) -> None:
        b1 = BV("V(a)+V(b)")
        b2 = BV("V(c)*3")
        assert b1.ref != b2.ref


class TestBCurrent:
    """Tests for behavioral current source component."""

    def test_bcurrent_creation(self) -> None:
        b = BCurrent("1", expr="I(Vref)*10")
        assert b.ref == "1"
        assert b.expr == "I(Vref)*10"
        assert len(b.ports) == 2
        assert {p.name for p in b.ports} == {"p", "n"}

    def test_bcurrent_spice_card(self) -> None:
        b = BCurrent("1", expr="I(Vref)*10")
        card = b.spice_card(_bsource_node_map)
        assert card == "B1 out gnd I=I(Vref)*10"

    def test_bcurrent_factory(self) -> None:
        b1 = BI("I(V1)*2")
        b2 = BI("I(V2)*0.5")
        assert b1.ref != b2.ref


class TestSubcktInstance:
    """Tests for subcircuit instance component."""

    def test_subckt_creation(self) -> None:
        x = SubcktInstance("1", "LM741", ["inp", "inn", "vcc", "vee", "out"])
        assert x.ref == "1"
        assert x.subckt_name == "LM741"
        assert len(x.ports) == 5
        # Ports are named n0, n1, n2, ... dynamically
        assert [p.name for p in x.ports] == ["n0", "n1", "n2", "n3", "n4"]

    def test_subckt_spice_card(self) -> None:
        # SubcktInstance uses node names directly, not port mapping
        x = SubcktInstance("1", "LM741", ["inp", "inn", "vcc", "vee", "out"])
        card = x.spice_card(_identity_node_map)  # node_map not used for nodes
        assert card == "X1 inp inn vcc vee out LM741"

    def test_subckt_with_params(self) -> None:
        x = SubcktInstance("2", "RES", ["n1", "n2"], params={"R": "1k"})
        card = x.spice_card(_identity_node_map)
        assert card == "X2 n1 n2 RES R=1k"

    def test_subckt_with_multiple_params(self) -> None:
        x = SubcktInstance(
            "3",
            "NMOS",
            ["drain", "gate", "source", "bulk"],
            params={"W": "10u", "L": "1u"},
        )
        card = x.spice_card(_identity_node_map)
        assert "X3 drain gate source bulk NMOS" in card
        assert "W=10u" in card
        assert "L=1u" in card

    def test_subckt_factory(self) -> None:
        x1 = XSUB("OPAMP", ["inp", "inn", "out"])
        x2 = XSUB("OPAMP", ["inp", "inn", "out"])
        assert x1.ref != x2.ref


class TestTransformer:
    """Tests for Transformer component."""

    def test_transformer_creation(self) -> None:
        xfmr = Transformer("1", turns_ratio=10)
        assert xfmr.ref == "1"
        assert xfmr.turns_ratio == 10
        assert xfmr.l_primary == "1m"
        assert xfmr.coupling == 0.9999
        assert len(xfmr.ports) == 4
        assert {p.name for p in xfmr.ports} == {"p1", "p2", "s1", "s2"}

    def test_transformer_spice_card(self) -> None:
        xfmr = Transformer("1", turns_ratio=10)
        card = xfmr.spice_card(_transformer_node_map)
        lines = card.split("\n")
        assert len(lines) == 3
        assert lines[0] == "L1p pri_p pri_n 1m"
        assert lines[1] == "L1s sec_p sec_n 100.0m"  # 1m * 10^2
        assert lines[2] == "K1 L1p L1s 0.9999"

    def test_transformer_custom_inductance(self) -> None:
        xfmr = Transformer("2", turns_ratio=2, l_primary="10m", coupling=0.99)
        card = xfmr.spice_card(_transformer_node_map)
        lines = card.split("\n")
        assert "10m" in lines[0]
        assert "40.0m" in lines[1]  # 10m * 2^2
        assert "0.99" in lines[2]

    def test_transformer_validation(self) -> None:
        with pytest.raises(ValueError, match="Turns ratio"):
            Transformer("1", turns_ratio=-1)
        with pytest.raises(ValueError, match="Turns ratio"):
            Transformer("1", turns_ratio=0)
        with pytest.raises(ValueError, match="Coupling"):
            Transformer("1", turns_ratio=1, coupling=0)
        with pytest.raises(ValueError, match="Coupling"):
            Transformer("1", turns_ratio=1, coupling=1.5)

    def test_transformer_factory(self) -> None:
        x1 = XFMR(turns_ratio=5)
        x2 = XFMR(turns_ratio=10)
        assert x1.ref != x2.ref


class TestCurrentProbe:
    """Tests for CurrentProbe component."""

    def test_current_probe_creation(self) -> None:
        probe = CurrentProbe("sense")
        assert probe.ref == "sense"
        assert probe.value == "0"
        assert len(probe.ports) == 2
        assert {p.name for p in probe.ports} == {"p", "n"}

    def test_current_probe_spice_card(self) -> None:
        probe = CurrentProbe("sense")
        card = probe.spice_card(_probe_node_map)
        assert card == "Vsense sense_p sense_n DC 0"

    def test_current_probe_factory(self) -> None:
        p1 = IPROBE("mysense")
        assert p1.ref == "mysense"

        p2 = IPROBE()
        p3 = IPROBE()
        assert p2.ref != p3.ref  # Auto-generated refs should differ


class TestVoltageProbe:
    """Tests for VoltageProbe component."""

    def test_voltage_probe_creation(self) -> None:
        probe = VoltageProbe("vout")
        assert probe.ref == "vout"
        assert probe.differential is False
        assert len(probe.ports) == 2
        assert {p.name for p in probe.ports} == {"p", "n"}

    def test_voltage_probe_differential(self) -> None:
        probe = VoltageProbe("vdiff", differential=True)
        assert probe.differential is True

    def test_voltage_probe_spice_card(self) -> None:
        probe = VoltageProbe("out")
        card = probe.spice_card(_probe_node_map)
        # VoltageProbe generates a comment, not a real element
        assert card.startswith("*")
        assert "Voltage probe" in card

    def test_voltage_probe_factory(self) -> None:
        p1 = VPROBE("output")
        assert p1.ref == "output"
        assert p1.differential is False

        p2 = VPROBE("diff_out", differential=True)
        assert p2.differential is True


class TestExpressionControlledSources:
    """Tests for expression-based controlled sources (VCVSExpr, VCCSExpr)."""

    def test_vcvs_expr_is_bvoltage(self) -> None:
        # VCVSExpr is an alias for BVoltage
        assert VCVSExpr is BVoltage

    def test_vccs_expr_is_bcurrent(self) -> None:
        # VCCSExpr is an alias for BCurrent
        assert VCCSExpr is BCurrent

    def test_eexpr_factory(self) -> None:
        # EExpr creates a BVoltage (VCVS with expression)
        e = EExpr("V(inp, inn) * 1000")
        assert isinstance(e, BVoltage)
        assert e.expr == "V(inp, inn) * 1000"
        card = e.spice_card(_bsource_node_map)
        assert "V=V(inp, inn) * 1000" in card

    def test_gexpr_factory(self) -> None:
        # GExpr creates a BCurrent (VCCS with expression)
        g = GExpr("V(in) * 0.001")
        assert isinstance(g, BCurrent)
        assert g.expr == "V(in) * 0.001"
        card = g.spice_card(_bsource_node_map)
        assert "I=V(in) * 0.001" in card

    def test_vcvs_expr_opamp_like(self) -> None:
        # Simulate an ideal op-amp with very high gain
        opamp = VCVSExpr("opamp", expr="V(inp, inn) * 1e6")
        card = opamp.spice_card(_bsource_node_map)
        assert card == "Bopamp out gnd V=V(inp, inn) * 1e6"

    def test_vccs_expr_transconductance(self) -> None:
        # Simulate a transconductance amplifier (gm = 10mS)
        gm = VCCSExpr("gm", expr="V(gate, source) * 0.01")
        card = gm.spice_card(_bsource_node_map)
        assert card == "Bgm out gnd I=V(gate, source) * 0.01"
