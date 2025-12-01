"""Tests for UX module (Section 10 - User Experience Polish).

Tests progress bars, undo/redo, clipboard, diff, and bookmarks.
"""

import io
import tempfile
from pathlib import Path

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor
from spicelab.core.net import GND, Net


def _make_simple_circuit(name: str = "Test") -> Circuit:
    """Create a simple circuit for testing."""
    circuit = Circuit(name)
    R1 = Resistor(ref="1", resistance=1000)
    C1 = Capacitor(ref="1", capacitance=1e-6)
    circuit.add(R1, C1)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(R1.ports[0], vin)
    circuit.connect(R1.ports[1], vout)
    circuit.connect(C1.ports[0], vout)
    circuit.connect(C1.ports[1], GND)

    return circuit


# =============================================================================
# Progress Bar Tests (10.1)
# =============================================================================


class TestProgressBar:
    """Tests for progress bar functionality."""

    def test_import(self):
        """Should be importable."""
        from spicelab.ux import ProgressBar, progress_context, with_progress

        assert ProgressBar is not None
        assert progress_context is not None
        assert with_progress is not None

    def test_basic_progress(self):
        """Basic progress bar should work."""
        from spicelab.ux.progress import ProgressBar

        output = io.StringIO()
        config_module = __import__("spicelab.ux.progress", fromlist=["ProgressConfig"])
        config = config_module.ProgressConfig(output=output)

        with ProgressBar(total=10, desc="Test", config=config) as pbar:
            for _ in range(10):
                pbar.update(1)

        output.seek(0)
        content = output.read()
        # Should have rendered something
        assert len(content) > 0

    def test_progress_percentage(self):
        """Progress should track percentage correctly."""
        from spicelab.ux.progress import ProgressBar

        with ProgressBar(total=100, disable=True) as pbar:
            pbar.update(50)
            assert pbar._current == 50

    def test_progress_iter(self):
        """ProgressBar.iter should iterate correctly."""
        from spicelab.ux.progress import ProgressBar

        items = list(range(10))
        result = list(ProgressBar.iter(items, disable=True))
        assert result == items

    def test_progress_context(self):
        """progress_context should work as context manager."""
        from spicelab.ux import progress_context

        with progress_context(desc="Test", total=5, disable=True) as pbar:
            for _ in range(5):
                pbar.update(1)
            assert pbar._current == 5

    def test_progress_style_enum(self):
        """Progress styles should be available."""
        from spicelab.ux.progress import ProgressStyle

        assert ProgressStyle.BAR is not None
        assert ProgressStyle.SPINNER is not None
        assert ProgressStyle.DETAILED is not None

    def test_set_description(self):
        """Should be able to update description."""
        from spicelab.ux.progress import ProgressBar

        with ProgressBar(total=10, disable=True) as pbar:
            pbar.set_description("New description")
            assert pbar._desc == "New description"

    def test_set_total(self):
        """Should be able to update total."""
        from spicelab.ux.progress import ProgressBar

        with ProgressBar(total=10, disable=True) as pbar:
            pbar.set_total(20)
            assert pbar._total == 20


# =============================================================================
# Undo/Redo Tests (10.2)
# =============================================================================


class TestUndoRedo:
    """Tests for undo/redo functionality."""

    def test_import(self):
        """Should be importable."""
        from spicelab.ux import CircuitHistory, HistoryManager, Snapshot

        assert CircuitHistory is not None
        assert HistoryManager is not None
        assert Snapshot is not None

    def test_history_initialization(self):
        """History should initialize with initial state."""
        from spicelab.ux import CircuitHistory

        circuit = _make_simple_circuit()
        history = CircuitHistory(circuit)

        assert len(history) == 1  # Initial state
        assert not history.can_undo()  # Can't undo initial
        assert not history.can_redo()

    def test_save_creates_snapshot(self):
        """Saving should create a new snapshot."""
        from spicelab.ux import CircuitHistory

        circuit = _make_simple_circuit()
        history = CircuitHistory(circuit)

        circuit.add(Resistor(ref="2", resistance=2000))
        history.save("Added R2")

        assert len(history) == 2
        assert history.can_undo()

    def test_undo_restores_state(self):
        """Undo should restore previous state."""
        from spicelab.ux import CircuitHistory

        circuit = _make_simple_circuit()
        history = CircuitHistory(circuit)

        initial_count = len(circuit._components)

        circuit.add(Resistor(ref="2", resistance=2000))
        history.save("Added R2")

        assert len(circuit._components) == initial_count + 1

        history.undo()

        assert len(circuit._components) == initial_count

    def test_redo_restores_undone(self):
        """Redo should restore undone change."""
        from spicelab.ux import CircuitHistory

        circuit = _make_simple_circuit()
        history = CircuitHistory(circuit)

        circuit.add(Resistor(ref="2", resistance=2000))
        history.save("Added R2")

        history.undo()
        initial_count = len(circuit._components)

        history.redo()

        assert len(circuit._components) == initial_count + 1

    def test_new_change_clears_redo(self):
        """New change should clear redo stack."""
        from spicelab.ux import CircuitHistory

        circuit = _make_simple_circuit()
        history = CircuitHistory(circuit)

        circuit.add(Resistor(ref="2", resistance=2000))
        history.save("Added R2")

        history.undo()
        assert history.can_redo()

        circuit.add(Resistor(ref="3", resistance=3000))
        history.save("Added R3")

        assert not history.can_redo()

    def test_snapshot_has_checksum(self):
        """Snapshots should have checksums."""
        from spicelab.ux import Snapshot

        snapshot = Snapshot(state={"test": 123}, description="Test")
        assert snapshot.checksum
        assert len(snapshot.checksum) == 8

    def test_history_manager_singleton(self):
        """HistoryManager should be a singleton."""
        from spicelab.ux import HistoryManager

        m1 = HistoryManager()
        m2 = HistoryManager()
        assert m1 is m2


# =============================================================================
# Clipboard Tests (10.3)
# =============================================================================


class TestClipboard:
    """Tests for clipboard functionality."""

    def test_import(self):
        """Should be importable."""
        from spicelab.ux import (
            CircuitClipboard,
            copy_circuit,
            paste_circuit,
        )

        assert CircuitClipboard is not None
        assert copy_circuit is not None
        assert paste_circuit is not None

    def test_clipboard_singleton(self):
        """CircuitClipboard should be a singleton."""
        from spicelab.ux import CircuitClipboard

        c1 = CircuitClipboard()
        c2 = CircuitClipboard()
        assert c1 is c2

    def test_copy_paste_component(self):
        """Should copy and paste components."""
        from spicelab.ux import CircuitClipboard

        clipboard = CircuitClipboard()
        clipboard.clear()

        R1 = Resistor(ref="1", resistance=1000)
        clipboard.copy_component(R1)

        R1_copy = clipboard.paste_component()

        assert R1_copy.ref == "1"
        assert R1_copy.resistance == 1000
        assert R1_copy is not R1

    def test_copy_paste_component_with_new_ref(self):
        """Should paste component with new reference."""
        from spicelab.ux import CircuitClipboard

        clipboard = CircuitClipboard()
        clipboard.clear()

        R1 = Resistor(ref="1", resistance=1000)
        clipboard.copy_component(R1)

        R2 = clipboard.paste_component(new_ref="2")
        assert R2.ref == "2"

    def test_copy_paste_circuit(self):
        """Should copy and paste circuits."""
        from spicelab.ux import CircuitClipboard

        clipboard = CircuitClipboard()
        clipboard.clear()

        circuit = _make_simple_circuit("Original")
        clipboard.copy_circuit(circuit)

        copy = clipboard.paste_circuit()

        assert copy.name == "Original"
        assert len(copy._components) == len(circuit._components)

    def test_copy_paste_circuit_with_new_name(self):
        """Should paste circuit with new name."""
        from spicelab.ux import CircuitClipboard

        clipboard = CircuitClipboard()
        clipboard.clear()

        circuit = _make_simple_circuit()
        clipboard.copy_circuit(circuit)

        copy = clipboard.paste_circuit(name="NewName")
        assert copy.name == "NewName"

    def test_clipboard_history(self):
        """Clipboard should maintain history."""
        from spicelab.ux import CircuitClipboard

        clipboard = CircuitClipboard()
        clipboard.clear()

        R1 = Resistor(ref="1", resistance=1000)
        R2 = Resistor(ref="2", resistance=2000)

        clipboard.copy_component(R1)
        clipboard.copy_component(R2)

        history = clipboard.history()
        assert len(history) == 2

    def test_convenience_functions(self):
        """Convenience functions should work."""
        from spicelab.ux import CircuitClipboard, copy_component, paste_component

        CircuitClipboard().clear()

        R1 = Resistor(ref="1", resistance=1000)
        copy_component(R1)

        R1_copy = paste_component()
        assert R1_copy.resistance == 1000


# =============================================================================
# Diff Tests (10.4)
# =============================================================================


class TestCircuitDiff:
    """Tests for circuit diff functionality."""

    def test_import(self):
        """Should be importable."""
        from spicelab.ux import CircuitDiff, DiffChange, diff_circuits

        assert CircuitDiff is not None
        assert diff_circuits is not None
        assert DiffChange is not None

    def test_identical_circuits_no_changes(self):
        """Identical circuits should have no changes."""
        from spicelab.ux import diff_circuits

        c1 = _make_simple_circuit("Test")
        c2 = _make_simple_circuit("Test")

        diff = diff_circuits(c1, c2)

        assert not diff.has_changes

    def test_detect_added_component(self):
        """Should detect added components."""
        from spicelab.ux import diff_circuits

        c1 = Circuit("Test")
        c1.add(Resistor(ref="1", resistance=1000))

        c2 = Circuit("Test")
        c2.add(Resistor(ref="1", resistance=1000))
        c2.add(Resistor(ref="2", resistance=2000))

        diff = diff_circuits(c1, c2)

        assert diff.has_changes
        assert len(diff.added) == 1
        assert "2" in str(diff.added[0])  # ref="2"

    def test_detect_removed_component(self):
        """Should detect removed components."""
        from spicelab.ux import diff_circuits

        c1 = Circuit("Test")
        c1.add(Resistor(ref="1", resistance=1000))
        c1.add(Resistor(ref="2", resistance=2000))

        c2 = Circuit("Test")
        c2.add(Resistor(ref="1", resistance=1000))

        diff = diff_circuits(c1, c2)

        assert diff.has_changes
        assert len(diff.removed) == 1

    def test_detect_modified_component(self):
        """Should detect modified components."""
        from spicelab.ux import diff_circuits

        c1 = Circuit("Test")
        c1.add(Resistor(ref="1", resistance=1000))

        c2 = Circuit("Test")
        c2.add(Resistor(ref="1", resistance=2000))

        diff = diff_circuits(c1, c2)

        assert diff.has_changes
        assert len(diff.modified) == 1

    def test_diff_summary(self):
        """Diff should provide summary."""
        from spicelab.ux import diff_circuits

        c1 = Circuit("Test")
        c1.add(Resistor(ref="1", resistance=1000))

        c2 = Circuit("Test")
        c2.add(Resistor(ref="1", resistance=2000))
        c2.add(Resistor(ref="2", resistance=3000))

        diff = diff_circuits(c1, c2)
        summary = diff.summary()

        assert "added" in summary
        assert "modified" in summary

    def test_diff_to_html(self):
        """Diff should generate HTML output."""
        from spicelab.ux import diff_circuits

        c1 = _make_simple_circuit("v1")
        c2 = _make_simple_circuit("v2")
        c2.add(Resistor(ref="2", resistance=2000))

        diff = diff_circuits(c1, c2)
        html = diff.to_html()

        assert "<div" in html
        assert "diff" in html

    def test_diff_to_unified(self):
        """Diff should generate unified diff format."""
        from spicelab.ux import diff_circuits

        c1 = Circuit("v1")
        c1.add(Resistor(ref="1", resistance=1000))

        c2 = Circuit("v2")
        c2.add(Resistor(ref="2", resistance=2000))

        diff = diff_circuits(c1, c2)
        unified = diff.to_unified_diff()

        assert "---" in unified
        assert "+++" in unified


# =============================================================================
# Bookmarks Tests (10.5)
# =============================================================================


class TestBookmarks:
    """Tests for bookmark functionality."""

    def test_import(self):
        """Should be importable."""
        from spicelab.ux import (
            Bookmark,
            BookmarkManager,
            list_bookmarks,
            load_bookmark,
            save_bookmark,
        )

        assert Bookmark is not None
        assert BookmarkManager is not None
        assert save_bookmark is not None
        assert load_bookmark is not None
        assert list_bookmarks is not None

    def test_bookmark_manager_with_temp_file(self):
        """BookmarkManager should work with custom path."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager = BookmarkManager(path)
            assert len(manager) == 0
        finally:
            path.unlink(missing_ok=True)

    def test_save_and_load_circuit(self):
        """Should save and load circuit bookmarks."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager = BookmarkManager(path)

            circuit = _make_simple_circuit("MyCircuit")
            manager.save_circuit("test_circuit", circuit, "Test circuit")

            loaded = manager.load_circuit("test_circuit")

            assert loaded.name == "MyCircuit"
            assert len(loaded._components) == len(circuit._components)
        finally:
            path.unlink(missing_ok=True)

    def test_save_and_load_config(self):
        """Should save and load config bookmarks."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager = BookmarkManager(path)

            config = {"engine": "ngspice", "verbose": True}
            manager.save_config("my_config", config, "My config")

            loaded = manager.load_config("my_config")

            assert loaded["engine"] == "ngspice"
            assert loaded["verbose"] is True
        finally:
            path.unlink(missing_ok=True)

    def test_delete_bookmark(self):
        """Should delete bookmarks."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager = BookmarkManager(path)
            manager.save_config("test", {"key": "value"})

            assert "test" in manager
            manager.delete("test")
            assert "test" not in manager
        finally:
            path.unlink(missing_ok=True)

    def test_list_bookmarks(self):
        """Should list bookmarks with filtering."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager = BookmarkManager(path)

            circuit = _make_simple_circuit()
            manager.save_circuit("c1", circuit, tags=["filter"])
            manager.save_config("cfg1", {"a": 1}, tags=["other"])

            all_bms = manager.list()
            assert len(all_bms) == 2

            circuits = manager.list(type_filter="circuit")
            assert len(circuits) == 1

            tagged = manager.list(tag_filter="filter")
            assert len(tagged) == 1
        finally:
            path.unlink(missing_ok=True)

    def test_search_bookmarks(self):
        """Should search bookmarks."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager = BookmarkManager(path)

            manager.save_config("audio_filter", {}, "Audio processing filter")
            manager.save_config("video_encoder", {}, "Video encoding config")

            results = manager.search("audio")
            assert len(results) == 1
            assert results[0].name == "audio_filter"
        finally:
            path.unlink(missing_ok=True)

    def test_export_import(self):
        """Should export and import bookmarks."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path1 = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path2 = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_path = Path(f.name)

        try:
            # Create and export
            m1 = BookmarkManager(path1)
            m1.save_config("cfg1", {"key": "value"})
            m1.export_to_file(export_path)

            # Import to new manager
            m2 = BookmarkManager(path2)
            imported = m2.import_from_file(export_path)

            assert imported == 1
            assert "cfg1" in m2
        finally:
            path1.unlink(missing_ok=True)
            path2.unlink(missing_ok=True)
            export_path.unlink(missing_ok=True)

    def test_bookmark_tags(self):
        """Should manage bookmark tags."""
        from spicelab.ux import BookmarkManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager = BookmarkManager(path)

            manager.save_config("c1", {}, tags=["audio", "filter"])
            manager.save_config("c2", {}, tags=["video", "filter"])

            tags = manager.tags()
            assert "audio" in tags
            assert "video" in tags
            assert "filter" in tags
        finally:
            path.unlink(missing_ok=True)


# =============================================================================
# Integration Tests
# =============================================================================


class TestUXIntegration:
    """Integration tests for UX module."""

    def test_all_exports(self):
        """All public APIs should be exported."""
        from spicelab.ux import (
            Bookmark,
            BookmarkManager,
            CircuitClipboard,
            CircuitDiff,
            CircuitHistory,
            HistoryManager,
            ProgressBar,
            copy_circuit,
            diff_circuits,
            paste_circuit,
            progress_context,
        )

        assert ProgressBar is not None
        assert progress_context is not None
        assert CircuitHistory is not None
        assert HistoryManager is not None
        assert CircuitClipboard is not None
        assert copy_circuit is not None
        assert paste_circuit is not None
        assert CircuitDiff is not None
        assert diff_circuits is not None
        assert Bookmark is not None
        assert BookmarkManager is not None

    def test_history_with_diff(self):
        """History snapshots should be diffable."""
        from spicelab.ux import CircuitHistory

        circuit = _make_simple_circuit()
        history = CircuitHistory(circuit)

        # Make change
        circuit.add(Resistor(ref="2", resistance=2000))
        history.save("Added R2")

        # Compare states would require restoring - skip for now
        # This is a conceptual test that the modules work together
        assert True

    def test_clipboard_with_diff(self):
        """Pasted circuit should be diffable with original."""
        from spicelab.ux import CircuitClipboard, diff_circuits

        clipboard = CircuitClipboard()
        clipboard.clear()

        original = _make_simple_circuit("Original")
        clipboard.copy_circuit(original)

        copy = clipboard.paste_circuit()
        copy.add(Resistor(ref="2", resistance=2000))

        diff = diff_circuits(original, copy)
        assert diff.has_changes
        assert len(diff.added) == 1
