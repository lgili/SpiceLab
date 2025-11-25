"""DX Demo: Netlist Diff

Demonstrates how to compare two netlists and visualize differences.

Features:
- Unified diff view (like git diff)
- Side-by-side comparison
- Summary of changes
- Similarity ratio

Run: python examples/dx_netlist_diff_demo.py
"""

from spicelab.templates import rc_highpass, rc_lowpass, sallen_key_lowpass
from spicelab.viz.netlist_diff import NetlistDiff, diff_circuits


def demo_simple_diff():
    """Demonstrate simple diff between two circuits."""
    print("=" * 60)
    print("1. Simple Diff: RC Lowpass at Different Frequencies")
    print("=" * 60)

    # Create two RC lowpass filters with different cutoff frequencies
    lpf_1k = rc_lowpass(fc=1000, impedance=10_000)
    lpf_2k = rc_lowpass(fc=2000, impedance=10_000)

    print("\nCircuit 1: 1kHz lowpass")
    print(lpf_1k.build_netlist())

    print("\nCircuit 2: 2kHz lowpass")
    print(lpf_2k.build_netlist())

    # Create diff
    diff = NetlistDiff(
        lpf_1k.build_netlist(),
        lpf_2k.build_netlist(),
        name1="1kHz",
        name2="2kHz",
    )

    print("\n--- Unified Diff ---")
    diff.print_unified()


def demo_topology_diff():
    """Demonstrate diff between different topologies."""
    print("\n" + "=" * 60)
    print("2. Topology Diff: Lowpass vs Highpass")
    print("=" * 60)

    lpf = rc_lowpass(fc=1000)
    hpf = rc_highpass(fc=1000)

    diff = NetlistDiff(
        lpf.build_netlist(),
        hpf.build_netlist(),
        name1="RC Lowpass",
        name2="RC Highpass",
    )

    print("\n--- Summary ---")
    diff.print_summary()


def demo_side_by_side():
    """Demonstrate side-by-side comparison."""
    print("\n" + "=" * 60)
    print("3. Side-by-Side: 1st Order vs 2nd Order Filter")
    print("=" * 60)

    first_order = rc_lowpass(fc=1000)
    second_order = sallen_key_lowpass(fc=1000).circuit

    diff = NetlistDiff(
        first_order.build_netlist(),
        second_order.build_netlist(),
        name1="1st Order RC",
        name2="2nd Order Sallen-Key",
    )

    print("\n--- Side-by-Side Comparison ---")
    diff.print_side_by_side()


def demo_identical_circuits():
    """Demonstrate diff of identical circuits."""
    print("\n" + "=" * 60)
    print("4. Identical Circuits")
    print("=" * 60)

    c1 = rc_lowpass(fc=1000)
    c2 = rc_lowpass(fc=1000)

    diff = NetlistDiff(
        c1.build_netlist(),
        c2.build_netlist(),
        name1="Circuit A",
        name2="Circuit B",
    )

    print(f"\nSimilarity: {diff.result.similarity_ratio:.1%}")
    print(f"Has changes: {diff.result.has_changes}")
    diff.print_unified()


def demo_diff_result():
    """Demonstrate accessing diff results programmatically."""
    print("\n" + "=" * 60)
    print("5. Programmatic Access to Diff Results")
    print("=" * 60)

    c1 = rc_lowpass(fc=1000)
    c2 = rc_lowpass(fc=2000)

    diff = NetlistDiff(
        c1.build_netlist(),
        c2.build_netlist(),
        name1="Original",
        name2="Modified",
    )

    result = diff.result

    print(f"\nSimilarity ratio: {result.similarity_ratio:.1%}")
    print(f"Added lines: {len(result.added_lines)}")
    print(f"Removed lines: {len(result.removed_lines)}")
    print(f"Unchanged lines: {len(result.unchanged_lines)}")
    print(f"Summary: {result.summary}")

    if result.removed_lines:
        print("\nRemoved:")
        for line in result.removed_lines:
            print(f"  - {line}")

    if result.added_lines:
        print("\nAdded:")
        for line in result.added_lines:
            print(f"  + {line}")


def demo_convenience_function():
    """Demonstrate the diff_circuits convenience function."""
    print("\n" + "=" * 60)
    print("6. diff_circuits() Convenience Function")
    print("=" * 60)

    c1 = rc_lowpass(fc=1000)
    c2 = rc_lowpass(fc=5000)

    print("\nUsing diff_circuits() for quick comparison:")
    diff_circuits(c1, c2, name1="1kHz", name2="5kHz", mode="summary")


def main():
    """Run all netlist diff demos."""
    print("\n" + "=" * 60)
    print("SpiceLab Netlist Diff Demo")
    print("=" * 60)
    print("\nNetlist diff helps you visualize changes between circuits,")
    print("useful for reviewing modifications and debugging.")

    demo_simple_diff()
    demo_topology_diff()
    demo_side_by_side()
    demo_identical_circuits()
    demo_diff_result()
    demo_convenience_function()

    print("\n" + "=" * 60)
    print("Use netlist diff to track circuit changes!")
    print("=" * 60)


if __name__ == "__main__":
    main()
