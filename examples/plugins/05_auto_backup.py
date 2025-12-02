"""Example: Auto-Backup Plugin

This example demonstrates how to use the AutoBackupPlugin to
automatically backup circuits and maintain version history.

Features:
- Automatic backup before simulations
- Backup on simulation errors
- Compression support (gzip)
- Deduplication to save space
- Easy restore from any backup
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.plugins.examples import AutoBackupPlugin
from spicelab.plugins.hooks import HookManager, HookType


def create_circuit_v1() -> Circuit:
    """Create version 1 of a circuit."""
    circuit = Circuit("my_design")

    vin = Vdc("Vin", 5.0)
    r1 = Resistor("R1", "1k")

    circuit.add(vin, r1)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], GND)

    return circuit


def create_circuit_v2() -> Circuit:
    """Create version 2 with added capacitor."""
    circuit = Circuit("my_design")

    vin = Vdc("Vin", 5.0)
    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "100n")  # Added capacitor

    circuit.add(vin, r1, c1)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], Net("out"))
    circuit.connect(c1.ports[0], Net("out"))
    circuit.connect(c1.ports[1], GND)

    return circuit


def create_circuit_v3() -> Circuit:
    """Create version 3 with added inductor."""
    circuit = Circuit("my_design")

    vin = Vdc("Vin", 5.0)
    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "100n")
    l1 = Inductor("L1", "10m")  # Added inductor

    circuit.add(vin, r1, c1, l1)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], Net("mid"))
    circuit.connect(l1.ports[0], Net("mid"))
    circuit.connect(l1.ports[1], Net("out"))
    circuit.connect(c1.ports[0], Net("out"))
    circuit.connect(c1.ports[1], GND)

    return circuit


def example_basic_backup():
    """Basic backup and restore."""
    print("=" * 60)
    print("Example 1: Basic Backup and Restore")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": tmpdir,
            "compress": False,  # Plain text for easy viewing
        })
        plugin.activate()

        # Create and backup circuit
        circuit = create_circuit_v1()
        print(f"\nCreated circuit: {circuit.name}")
        print(f"Components: {len(circuit._components)}")

        # Manual backup
        info = plugin.backup(circuit, reason="initial_version")
        print(f"\nBackup created: {info.name}")
        print(f"Path: {info.path}")
        print(f"Size: {info.size_bytes} bytes")

        # View backup content
        print("\nBackup content:")
        content = plugin.restore_netlist(info.name)
        print(content)

        plugin.deactivate()


def example_version_history():
    """Maintain version history of a circuit."""
    print("\n" + "=" * 60)
    print("Example 2: Version History")
    print("=" * 60)

    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": tmpdir,
            "compress": True,
            "deduplicate": False,  # Keep all versions even if identical
        })
        plugin.activate()

        # Create multiple versions
        versions = [
            ("v1", create_circuit_v1, "Initial design"),
            ("v2", create_circuit_v2, "Added filter capacitor"),
            ("v3", create_circuit_v3, "Added inductor for LC filter"),
        ]

        print("\nCreating version history:")
        for version, create_fn, desc in versions:
            circuit = create_fn()
            info = plugin.backup(circuit, reason=desc)
            print(f"  {version}: {info.name} ({info.size_bytes} bytes)")
            time.sleep(0.1)  # Ensure different timestamps

        # List all backups
        print("\nBackup History (newest first):")
        for backup in plugin.list_backups():
            print(f"  {backup.name}")
            print(f"    Circuit: {backup.circuit_name}")
            print(f"    Time: {backup.timestamp}")
            print(f"    Reason: {backup.metadata.get('reason', 'N/A')}")

        plugin.deactivate()


def example_automatic_backup():
    """Automatic backup via hooks."""
    print("\n" + "=" * 60)
    print("Example 3: Automatic Backup via Hooks")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": tmpdir,
            "backup_on_error": True,
        })
        plugin.activate()

        hook_manager = HookManager.get_instance()

        # Simulate pre-simulation hook (auto-backup happens here)
        circuit = create_circuit_v2()
        print("\nSimulating PRE_SIMULATION hook...")
        hook_manager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=[])

        print(f"Backups after simulation start: {len(plugin.list_backups())}")

        # Simulate error (backup on error)
        print("\nSimulating SIMULATION_ERROR hook...")
        hook_manager.trigger(
            HookType.SIMULATION_ERROR,
            circuit=circuit,
            error="Convergence failed at t=0.001s",
        )

        backups = plugin.list_backups()
        print(f"Backups after error: {len(backups)}")

        # Show error backup
        error_backups = [b for b in backups if b.metadata.get("reason") == "error"]
        if error_backups:
            print(f"\nError backup: {error_backups[0].name}")
            print(f"Error message: {error_backups[0].metadata.get('error')}")

        plugin.deactivate()


def example_deduplication():
    """Demonstrate deduplication feature."""
    print("\n" + "=" * 60)
    print("Example 4: Deduplication")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": tmpdir,
            "deduplicate": True,  # Enable deduplication
        })
        plugin.activate()

        circuit = create_circuit_v1()

        # Try to backup same circuit multiple times
        print("\nBacking up same circuit 3 times with deduplicate=True:")
        results = []
        for i in range(3):
            info = plugin.backup(circuit, reason=f"backup_{i}")
            results.append(info)
            status = "Created" if info else "Skipped (duplicate)"
            print(f"  Attempt {i+1}: {status}")

        print(f"\nActual backups created: {len(plugin.list_backups())}")
        print("Deduplication saves space by not storing identical circuits.")

        plugin.deactivate()


def example_compression():
    """Compare compressed vs uncompressed backups."""
    print("\n" + "=" * 60)
    print("Example 5: Compression")
    print("=" * 60)

    import tempfile

    circuit = create_circuit_v3()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Uncompressed backup
        plugin_raw = AutoBackupPlugin()
        plugin_raw.configure({
            "backup_dir": f"{tmpdir}/raw",
            "compress": False,
            "deduplicate": False,
        })
        plugin_raw.activate()
        raw_info = plugin_raw.backup(circuit)

        # Compressed backup
        plugin_gz = AutoBackupPlugin()
        plugin_gz.configure({
            "backup_dir": f"{tmpdir}/compressed",
            "compress": True,
            "deduplicate": False,
        })
        plugin_gz.activate()
        gz_info = plugin_gz.backup(circuit)

        print("\nSize Comparison:")
        print(f"  Uncompressed: {raw_info.size_bytes} bytes ({raw_info.path.suffix})")
        print(f"  Compressed:   {gz_info.size_bytes} bytes ({gz_info.path.suffix})")

        if raw_info.size_bytes > 0:
            ratio = gz_info.size_bytes / raw_info.size_bytes
            print(f"  Compression ratio: {ratio:.1%}")

        plugin_raw.deactivate()
        plugin_gz.deactivate()


def example_restore_circuit():
    """Restore a circuit from backup."""
    print("\n" + "=" * 60)
    print("Example 6: Restore Circuit")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": tmpdir})
        plugin.activate()

        # Backup original circuit
        original = create_circuit_v2()
        info = plugin.backup(original, reason="original")

        print(f"Original circuit: {original.name}")
        print(f"Components: {len(original._components)}")
        print(f"Backup: {info.name}")

        # Restore from backup
        print("\nRestoring netlist from backup...")
        netlist = plugin.restore_netlist(info.name)
        print("Restored netlist:")
        print(netlist)

        # Note: Full circuit restore requires parser
        print("\nNote: Use plugin.restore(name) for full Circuit object")
        print("      (requires spicelab.io.parser)")

        plugin.deactivate()


def example_backup_management():
    """Manage backups - list, delete, clear."""
    print("\n" + "=" * 60)
    print("Example 7: Backup Management")
    print("=" * 60)

    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": tmpdir,
            "deduplicate": False,
        })
        plugin.activate()

        # Create several backups
        circuits = [
            ("design_a", create_circuit_v1),
            ("design_b", create_circuit_v2),
            ("design_a", create_circuit_v3),  # Another version of design_a
        ]

        for name, create_fn in circuits:
            circuit = create_fn()
            circuit._name = name  # Override name
            plugin.backup(circuit)
            time.sleep(0.1)

        # List all backups
        print("\nAll backups:")
        for b in plugin.list_backups():
            print(f"  {b.name} ({b.circuit_name})")

        # Filter by circuit name
        print("\nBackups for 'design_a' only:")
        for b in plugin.list_backups(circuit_name="design_a"):
            print(f"  {b.name}")

        # Get specific backup
        backups = plugin.list_backups()
        if backups:
            specific = plugin.get_backup(backups[0].name)
            print(f"\nSpecific backup details: {specific}")

        # Delete a backup
        if backups:
            deleted = plugin.delete_backup(backups[-1].name)
            print(f"\nDeleted oldest backup: {deleted}")
            print(f"Remaining backups: {len(plugin.list_backups())}")

        plugin.deactivate()


def example_max_backups():
    """Automatic cleanup of old backups."""
    print("\n" + "=" * 60)
    print("Example 8: Max Backups Limit")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": tmpdir,
            "max_backups": 3,  # Only keep 3 most recent
            "deduplicate": False,
        })
        plugin.activate()

        # Create 5 different circuits
        print("\nCreating 5 backups with max_backups=3:")
        for i in range(5):
            circuit = Circuit(f"circuit_{i}")
            r = Resistor(f"R{i}", f"{i+1}k")
            circuit.add(r)
            circuit.connect(r.ports[0], Net(f"n{i}"))
            circuit.connect(r.ports[1], GND)

            plugin.backup(circuit)
            current = len(plugin.list_backups())
            print(f"  After backup {i+1}: {current} backups stored")

        print("\nOldest backups are automatically removed.")

        plugin.deactivate()


def example_backup_stats():
    """View backup statistics."""
    print("\n" + "=" * 60)
    print("Example 9: Backup Statistics")
    print("=" * 60)

    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": tmpdir,
            "deduplicate": False,
        })
        plugin.activate()

        # Create various backups
        for create_fn in [create_circuit_v1, create_circuit_v2, create_circuit_v3]:
            circuit = create_fn()
            plugin.backup(circuit)
            time.sleep(0.1)

        # Get statistics
        stats = plugin.get_stats()

        print("\nBackup Statistics:")
        print(f"  Total backups: {stats['total_backups']}")
        print(f"  Total size: {stats['total_size_mb']:.3f} MB")
        print(f"  Unique circuits: {stats['unique_circuits']}")
        print(f"  Circuits: {stats['circuits']}")
        print(f"  Compressed: {stats['compressed']}")
        print(f"  Oldest: {stats['oldest']}")
        print(f"  Newest: {stats['newest']}")

        plugin.deactivate()


def example_export_backup():
    """Export backup to a specific location."""
    print("\n" + "=" * 60)
    print("Example 10: Export Backup")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": f"{tmpdir}/backups"})
        plugin.activate()

        # Create a backup
        circuit = create_circuit_v2()
        info = plugin.backup(circuit)

        # Export to different location
        export_dir = Path(tmpdir) / "exports"
        export_dir.mkdir()
        export_path = export_dir / "exported_circuit.cir.gz"

        result = plugin.export_backup(info.name, export_path)

        print(f"\nOriginal backup: {info.path}")
        print(f"Exported to: {result}")
        print(f"Export exists: {result.exists()}")

        plugin.deactivate()


if __name__ == "__main__":
    example_basic_backup()
    example_version_history()
    example_automatic_backup()
    example_deduplication()
    example_compression()
    example_restore_circuit()
    example_backup_management()
    example_max_backups()
    example_backup_stats()
    example_export_backup()

    print("\n" + "=" * 60)
    print("Auto-Backup Plugin Examples Complete!")
    print("=" * 60)
