"""Tests for netlist diff visualization (M4 DX improvement).

Tests the NetlistDiff class for comparing SPICE netlists.
"""

from io import StringIO

from rich.console import Console
from spicelab.templates import rc_highpass, rc_lowpass
from spicelab.viz.netlist_diff import DiffResult, NetlistDiff, diff_circuits


class TestDiffResult:
    """Tests for DiffResult dataclass."""

    def test_no_changes(self):
        """DiffResult with no changes."""
        result = DiffResult(
            added_lines=[],
            removed_lines=[],
            unchanged_lines=["line1", "line2"],
            similarity_ratio=1.0,
        )
        assert not result.has_changes
        assert result.summary == "No changes"

    def test_with_added_lines(self):
        """DiffResult with added lines."""
        result = DiffResult(
            added_lines=["new line"],
            removed_lines=[],
            unchanged_lines=["line1"],
            similarity_ratio=0.8,
        )
        assert result.has_changes
        assert "+1 added" in result.summary

    def test_with_removed_lines(self):
        """DiffResult with removed lines."""
        result = DiffResult(
            added_lines=[],
            removed_lines=["old line"],
            unchanged_lines=["line1"],
            similarity_ratio=0.8,
        )
        assert result.has_changes
        assert "-1 removed" in result.summary

    def test_with_both_changes(self):
        """DiffResult with added and removed lines."""
        result = DiffResult(
            added_lines=["new1", "new2"],
            removed_lines=["old1"],
            unchanged_lines=["line1"],
            similarity_ratio=0.5,
        )
        assert result.has_changes
        assert "+2 added" in result.summary
        assert "-1 removed" in result.summary


class TestNetlistDiff:
    """Tests for NetlistDiff class."""

    def test_identical_netlists(self):
        """Identical netlists should have similarity 1.0."""
        netlist = "* Test\nR1 1 2 1k\n.end"
        diff = NetlistDiff(netlist, netlist)
        assert diff.result.similarity_ratio == 1.0
        assert not diff.result.has_changes

    def test_different_netlists(self):
        """Different netlists should show changes."""
        netlist1 = "* Test\nR1 1 2 1k\n.end"
        netlist2 = "* Test\nR1 1 2 2k\n.end"
        diff = NetlistDiff(netlist1, netlist2)
        assert diff.result.similarity_ratio < 1.0
        assert diff.result.has_changes

    def test_unified_diff_output(self):
        """Unified diff should contain standard markers."""
        netlist1 = "* Test\nR1 1 2 1k\n.end"
        netlist2 = "* Test\nR1 1 2 2k\n.end"
        diff = NetlistDiff(netlist1, netlist2, "old", "new")
        unified = diff.unified_diff()
        assert "--- old" in unified
        assert "+++ new" in unified
        assert "-R1 1 2 1k" in unified
        assert "+R1 1 2 2k" in unified

    def test_unified_diff_identical(self):
        """Unified diff of identical netlists should be empty."""
        netlist = "* Test\nR1 1 2 1k\n.end"
        diff = NetlistDiff(netlist, netlist)
        unified = diff.unified_diff()
        assert unified == ""

    def test_print_unified_no_changes(self):
        """print_unified with no changes should indicate no differences."""
        netlist = "* Test\nR1 1 2 1k\n.end"
        diff = NetlistDiff(netlist, netlist)

        # Capture output
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        diff.print_unified(console=console)
        result = output.getvalue()

        assert "No differences" in result

    def test_print_summary_no_changes(self):
        """print_summary with no changes should indicate identical."""
        netlist = "* Test\nR1 1 2 1k\n.end"
        diff = NetlistDiff(netlist, netlist)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        diff.print_summary(console=console)
        result = output.getvalue()

        assert "identical" in result.lower()

    def test_print_summary_with_changes(self):
        """print_summary should show added/removed counts."""
        netlist1 = "* Test\nR1 1 2 1k\n.end"
        netlist2 = "* Test\nR1 1 2 2k\nC1 2 0 1u\n.end"
        diff = NetlistDiff(netlist1, netlist2)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        diff.print_summary(console=console)
        result = output.getvalue()

        assert "Similarity" in result
        assert "Added" in result or "added" in result.lower()

    def test_added_lines_detection(self):
        """Should detect lines added in second netlist."""
        netlist1 = "* Test\nR1 1 2 1k\n.end"
        netlist2 = "* Test\nR1 1 2 1k\nC1 2 0 1u\n.end"
        diff = NetlistDiff(netlist1, netlist2)

        assert len(diff.result.added_lines) == 1
        assert "C1" in diff.result.added_lines[0]

    def test_removed_lines_detection(self):
        """Should detect lines removed from first netlist."""
        netlist1 = "* Test\nR1 1 2 1k\nC1 2 0 1u\n.end"
        netlist2 = "* Test\nR1 1 2 1k\n.end"
        diff = NetlistDiff(netlist1, netlist2)

        assert len(diff.result.removed_lines) == 1
        assert "C1" in diff.result.removed_lines[0]


class TestDiffCircuits:
    """Tests for diff_circuits convenience function."""

    def test_diff_rc_filters(self):
        """Should diff two RC filter circuits."""
        c1 = rc_lowpass(fc=1000)
        c2 = rc_lowpass(fc=2000)

        # Capture output
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)

        # Need to patch the function to use our console
        from spicelab.viz import netlist_diff

        original_console = Console
        netlist_diff.Console = lambda: console

        try:
            diff = diff_circuits(c1, c2, "1kHz", "2kHz", mode="summary")
        finally:
            netlist_diff.Console = original_console

        assert diff.result.has_changes
        # Capacitance value changes for different fc
        assert len(diff.result.added_lines) >= 1
        assert len(diff.result.removed_lines) >= 1

    def test_diff_same_circuit(self):
        """Diffing same circuit should show no changes."""
        c1 = rc_lowpass(fc=1000)
        c2 = rc_lowpass(fc=1000)
        diff = NetlistDiff(c1.build_netlist(), c2.build_netlist())
        assert not diff.result.has_changes

    def test_diff_different_topologies(self):
        """Should diff circuits with different topologies."""
        c1 = rc_lowpass(fc=1000)
        c2 = rc_highpass(fc=1000)
        diff = NetlistDiff(c1.build_netlist(), c2.build_netlist())

        # Different circuit names
        assert diff.result.has_changes
        # Should have low similarity due to different topology
        assert diff.result.similarity_ratio < 1.0


class TestSideBySide:
    """Tests for side-by-side comparison."""

    def test_side_by_side_output(self):
        """Side-by-side should produce table output."""
        netlist1 = "* Test\nR1 1 2 1k\n.end"
        netlist2 = "* Test\nR1 1 2 2k\n.end"
        diff = NetlistDiff(netlist1, netlist2)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        diff.print_side_by_side(console=console)
        result = output.getvalue()

        # Should contain table-like output
        assert "original" in result.lower() or "modified" in result.lower()


class TestContextLines:
    """Tests for context line configuration."""

    def test_context_lines_in_unified(self):
        """Should respect context_lines parameter."""
        # Create netlists with many lines
        lines1 = ["* Test"] + [f"R{i} {i} {i+1} 1k" for i in range(20)] + [".end"]
        lines2 = lines1.copy()
        lines2[10] = "R10 10 11 2k"  # Change one line in the middle

        netlist1 = "\n".join(lines1)
        netlist2 = "\n".join(lines2)

        diff = NetlistDiff(netlist1, netlist2)

        # With 1 context line
        unified_1 = diff.unified_diff(context_lines=1)
        # With 5 context lines
        unified_5 = diff.unified_diff(context_lines=5)

        # More context should produce more output
        assert len(unified_5) > len(unified_1)
