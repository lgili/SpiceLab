#!/usr/bin/env python3
"""Demonstration of SpiceLab UX features.

This example shows how to use the User Experience features:
- Progress bars with ETA
- Undo/redo for circuit modifications
- Clipboard for copy/paste
- Circuit diff visualization
- Bookmarks for saving configurations
"""

import tempfile
import time
from pathlib import Path

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net


def create_rc_filter(name: str = "RC_Filter", r: float = 10_000, c: float = 1e-6) -> Circuit:
    """Create a simple RC lowpass filter circuit."""
    circuit = Circuit(name)

    # Components
    V1 = Vac(ref="1", value="1", ac_mag="1")
    R1 = Resistor(ref="1", resistance=r)
    C1 = Capacitor(ref="1", capacitance=c)

    circuit.add(V1, R1, C1)

    # Nets
    vin = Net("vin")
    vout = Net("vout")

    # Connections
    circuit.connect(V1.ports[0], vin)
    circuit.connect(V1.ports[1], GND)
    circuit.connect(R1.ports[0], vin)
    circuit.connect(R1.ports[1], vout)
    circuit.connect(C1.ports[0], vout)
    circuit.connect(C1.ports[1], GND)

    return circuit


def demo_progress_bar():
    """Demonstrate progress bar functionality."""
    from spicelab.ux import ProgressBar, progress_context
    from spicelab.ux.progress import ProgressStyle

    print("\n" + "=" * 60)
    print("1. PROGRESS BAR DEMO")
    print("=" * 60)

    # Basic progress bar
    print("\n1.1 Basic progress bar:")
    with ProgressBar(total=20, desc="Processing") as pbar:
        for _ in range(20):
            time.sleep(0.05)  # Simulate work
            pbar.update(1)
    print()  # Newline after progress

    # Progress with iterator
    print("\n1.2 Progress with iterator:")
    items = list(range(15))
    for _ in ProgressBar.iter(items, desc="Iterating"):
        time.sleep(0.03)
    print()

    # Different styles
    print("\n1.3 Detailed style:")
    with ProgressBar(total=10, desc="Detailed", style=ProgressStyle.DETAILED) as pbar:
        for _ in range(10):
            time.sleep(0.05)
            pbar.update(1)
    print()

    # Context manager
    print("\n1.4 Using context manager:")
    with progress_context(desc="Simulating", total=25) as pbar:
        for _ in range(25):
            time.sleep(0.02)
            pbar.update(1)
    print()


def demo_undo_redo():
    """Demonstrate undo/redo functionality."""
    from spicelab.ux import CircuitHistory

    print("\n" + "=" * 60)
    print("2. UNDO/REDO DEMO")
    print("=" * 60)

    # Create circuit and history
    circuit = Circuit("MyFilter")
    history = CircuitHistory(circuit)

    print(f"\nInitial state: {len(circuit._components)} components")

    # Add R1
    R1 = Resistor(ref="1", resistance=1000)
    circuit.add(R1)
    history.save("Added R1 (1kΩ)")
    print(f"After adding R1: {len(circuit._components)} components")

    # Add R2
    R2 = Resistor(ref="2", resistance=2000)
    circuit.add(R2)
    history.save("Added R2 (2kΩ)")
    print(f"After adding R2: {len(circuit._components)} components")

    # Add C1
    C1 = Capacitor(ref="1", capacitance=100e-9)
    circuit.add(C1)
    history.save("Added C1 (100nF)")
    print(f"After adding C1: {len(circuit._components)} components")

    # Show history
    print(f"\nHistory ({len(history)} snapshots):")
    for snapshot in history.history():
        print(f"  - {snapshot}")

    # Undo
    print("\n--- Undo operations ---")
    history.undo()
    print(f"After undo: {len(circuit._components)} components (removed C1)")

    history.undo()
    print(f"After undo: {len(circuit._components)} components (removed R2)")

    print(f"\nCan undo: {history.can_undo()}, Can redo: {history.can_redo()}")

    # Redo
    print("\n--- Redo operations ---")
    history.redo()
    print(f"After redo: {len(circuit._components)} components (restored R2)")


def demo_clipboard():
    """Demonstrate clipboard functionality."""
    from spicelab.ux import (
        CircuitClipboard,
        copy_circuit,
        copy_component,
        paste_circuit,
        paste_component,
    )

    print("\n" + "=" * 60)
    print("3. CLIPBOARD DEMO")
    print("=" * 60)

    clipboard = CircuitClipboard()
    clipboard.clear()

    # Copy/paste component
    print("\n3.1 Copy/paste component:")
    R1 = Resistor(ref="original", resistance=4700)
    print(f"Original: R{R1.ref} = {R1.resistance}Ω")

    copy_component(R1, "4.7kΩ resistor")
    R1_copy = paste_component(new_ref="copy")
    print(f"Pasted:   R{R1_copy.ref} = {R1_copy.resistance}Ω")

    # Copy/paste circuit
    print("\n3.2 Copy/paste circuit:")
    original = create_rc_filter("Original_RC", r=10_000, c=1e-6)
    print(f"Original circuit: '{original.name}' with {len(original._components)} components")

    copy_circuit(original, "RC filter with fc=16Hz")
    duplicate = paste_circuit(name="Duplicate_RC")
    print(f"Pasted circuit:   '{duplicate.name}' with {len(duplicate._components)} components")

    # Clipboard history
    print("\n3.3 Clipboard history:")
    for item in clipboard.history():
        print(f"  - {item}")


def demo_diff():
    """Demonstrate circuit diff functionality."""
    from spicelab.ux import diff_circuits

    print("\n" + "=" * 60)
    print("4. CIRCUIT DIFF DEMO")
    print("=" * 60)

    # Create two versions of a circuit
    v1 = Circuit("Filter_v1")
    v1.add(Resistor(ref="1", resistance=1000))
    v1.add(Capacitor(ref="1", capacitance=100e-9))

    v2 = Circuit("Filter_v2")
    v2.add(Resistor(ref="1", resistance=2200))  # Modified value
    v2.add(Capacitor(ref="1", capacitance=100e-9))  # Same
    v2.add(Capacitor(ref="2", capacitance=47e-9))  # Added

    print("\nv1: R1=1kΩ, C1=100nF")
    print("v2: R1=2.2kΩ (modified), C1=100nF, C2=47nF (added)")

    # Generate diff
    diff = diff_circuits(v1, v2)

    print("\n--- Diff Summary ---")
    print(f"Summary: {diff.summary()}")
    print(f"Has changes: {diff.has_changes}")

    print(f"\nAdded ({len(diff.added)}):")
    for change in diff.added:
        print(f"  {change}")

    print(f"\nModified ({len(diff.modified)}):")
    for change in diff.modified:
        print(f"  {change}")

    # Full diff output
    print("\n--- Full Diff ---")
    print(diff)


def demo_bookmarks():
    """Demonstrate bookmark functionality."""
    from spicelab.core.types import AnalysisSpec
    from spicelab.ux import BookmarkManager

    print("\n" + "=" * 60)
    print("5. BOOKMARKS DEMO")
    print("=" * 60)

    # Use temporary file for bookmarks
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        bookmark_path = Path(f.name)

    try:
        manager = BookmarkManager(bookmark_path)

        # Save circuit bookmark
        print("\n5.1 Save circuit bookmark:")
        circuit = create_rc_filter("AudioFilter", r=15_000, c=10e-9)
        manager.save_circuit(
            "audio_lpf",
            circuit,
            "Low-pass filter for audio (fc≈1kHz)",
            tags=["audio", "filter", "lowpass"],
        )
        print("  Saved: 'audio_lpf'")

        # Save analysis preset
        print("\n5.2 Save analysis preset:")
        analyses = [
            AnalysisSpec("ac", {"sweep_type": "dec", "n": 100, "fstart": 1, "fstop": 1e6}),
        ]
        manager.save_analysis(
            "detailed_ac", analyses, "Detailed AC sweep 1Hz-1MHz", tags=["ac", "sweep"]
        )
        print("  Saved: 'detailed_ac'")

        # Save config
        print("\n5.3 Save configuration:")
        config = {
            "engine": "ngspice",
            "verbose": True,
            "timeout": 60,
        }
        manager.save_config(
            "default_sim_config",
            config,
            "Default simulation settings",
            tags=["config", "simulation"],
        )
        print("  Saved: 'default_sim_config'")

        # List all bookmarks
        print("\n5.4 List all bookmarks:")
        for bm in manager.list():
            print(f"  {bm}")

        # Filter by tag
        print("\n5.5 Filter by tag 'filter':")
        for bm in manager.list(tag_filter="filter"):
            print(f"  {bm}")

        # Load circuit back
        print("\n5.6 Load circuit from bookmark:")
        loaded_circuit = manager.load_circuit("audio_lpf")
        print(
            f"  Loaded: '{loaded_circuit.name}' with {len(loaded_circuit._components)} components"
        )

        # Search
        print("\n5.7 Search for 'audio':")
        results = manager.search("audio")
        for bm in results:
            print(f"  {bm}")

        # Show all tags
        print("\n5.8 All tags:")
        print(f"  {manager.tags()}")

    finally:
        # Cleanup
        bookmark_path.unlink(missing_ok=True)


def demo_integration():
    """Demonstrate integrated workflow using multiple UX features."""
    from spicelab.ux import (
        BookmarkManager,
        CircuitClipboard,
        CircuitHistory,
        ProgressBar,
        diff_circuits,
    )

    print("\n" + "=" * 60)
    print("6. INTEGRATED WORKFLOW DEMO")
    print("=" * 60)

    # Use temporary file for bookmarks
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        bookmark_path = Path(f.name)

    try:
        manager = BookmarkManager(bookmark_path)
        clipboard = CircuitClipboard()
        clipboard.clear()

        print("\nWorkflow: Design iteration with history tracking")
        print("-" * 50)

        # Step 1: Create initial circuit
        print("\n[Step 1] Create initial RC filter")
        circuit = create_rc_filter("DesignV1", r=10_000, c=1e-6)
        history = CircuitHistory(circuit)
        print(f"  Created: {circuit.name} (fc ≈ 16 Hz)")

        # Save initial version
        manager.save_circuit("design_v1", circuit, "Initial design", tags=["wip"])

        # Step 2: Modify with progress
        print("\n[Step 2] Adding components with progress tracking")
        new_components = [
            Resistor(ref="2", resistance=22_000),
            Capacitor(ref="2", capacitance=470e-9),
            Resistor(ref="3", resistance=4_700),
        ]

        for comp in ProgressBar.iter(new_components, desc="Adding"):
            circuit.add(comp)
            history.save(f"Added {type(comp).__name__} {comp.ref}")
            time.sleep(0.1)  # Simulate processing
        print()

        # Step 3: Show history
        print("\n[Step 3] Review history")
        print(f"  Total snapshots: {len(history)}")
        print(f"  Can undo: {history.can_undo()}")

        # Step 4: Create a copy for comparison
        print("\n[Step 4] Copy to clipboard and create variant")
        clipboard.copy_circuit(circuit, "Working version")
        _variant = clipboard.paste_circuit(name="DesignV2_Variant")  # noqa: F841

        # Step 5: Compare versions
        print("\n[Step 5] Compare original v1 with current")
        original = manager.load_circuit("design_v1")
        diff = diff_circuits(original, circuit)
        print(f"  Changes: {diff.summary()}")

        # Step 6: Undo some changes
        print("\n[Step 6] Undo last 2 changes")
        history.undo()
        history.undo()
        print(f"  Components after undo: {len(circuit._components)}")

        # Step 7: Save final version
        print("\n[Step 7] Save final version as bookmark")
        manager.save_circuit("design_final", circuit, "Final approved design", tags=["approved"])
        print("  Saved as 'design_final'")

        # Summary
        print("\n" + "-" * 50)
        print("Workflow complete!")
        print(f"  Bookmarks saved: {len(manager)}")
        print(f"  Clipboard items: {len(clipboard)}")
        print(f"  History snapshots: {len(history)}")

    finally:
        bookmark_path.unlink(missing_ok=True)


def main():
    """Run all demos."""
    print("=" * 60)
    print("SpiceLab UX Features Demo")
    print("=" * 60)

    demo_progress_bar()
    demo_undo_redo()
    demo_clipboard()
    demo_diff()
    demo_bookmarks()
    demo_integration()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
