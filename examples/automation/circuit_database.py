"""Circuit Database Integration

Demonstrates how to store, query, and manage circuit designs
in a database-like structure for design reuse and tracking.

Run: python examples/automation/circuit_database.py
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc


@dataclass
class CircuitRecord:
    """Database record for a circuit design."""

    id: str
    name: str
    category: str
    description: str
    components: dict[str, dict]
    parameters: dict[str, float]
    specifications: dict[str, float]
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    status: str = "draft"  # draft, reviewed, approved, deprecated


class CircuitDatabase:
    """In-memory circuit database with JSON persistence."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else None
        self.records: dict[str, CircuitRecord] = {}
        self._next_id = 1

        if self.db_path and self.db_path.exists():
            self.load()

    def _generate_id(self) -> str:
        """Generate unique ID."""
        record_id = f"CKT-{self._next_id:04d}"
        self._next_id += 1
        return record_id

    def add(self, record: CircuitRecord) -> str:
        """Add a circuit record to the database."""
        if not record.id:
            record.id = self._generate_id()
        record.created_at = datetime.now().isoformat()
        record.updated_at = record.created_at
        self.records[record.id] = record
        return record.id

    def get(self, record_id: str) -> CircuitRecord | None:
        """Get a record by ID."""
        return self.records.get(record_id)

    def update(self, record_id: str, **kwargs) -> bool:
        """Update a record."""
        if record_id not in self.records:
            return False

        record = self.records[record_id]
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.updated_at = datetime.now().isoformat()
        return True

    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        if record_id in self.records:
            del self.records[record_id]
            return True
        return False

    def search(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
        name_contains: str | None = None,
    ) -> list[CircuitRecord]:
        """Search records with filters."""
        results = []

        for record in self.records.values():
            # Apply filters
            if category and record.category != category:
                continue
            if status and record.status != status:
                continue
            if name_contains and name_contains.lower() not in record.name.lower():
                continue
            if tags and not all(t in record.tags for t in tags):
                continue

            results.append(record)

        return results

    def list_categories(self) -> list[str]:
        """List all unique categories."""
        return list(set(r.category for r in self.records.values()))

    def list_tags(self) -> list[str]:
        """List all unique tags."""
        tags = set()
        for record in self.records.values():
            tags.update(record.tags)
        return list(tags)

    def save(self):
        """Save database to JSON file."""
        if not self.db_path:
            return

        data = {
            "next_id": self._next_id,
            "records": {k: asdict(v) for k, v in self.records.items()},
        }
        self.db_path.write_text(json.dumps(data, indent=2))

    def load(self):
        """Load database from JSON file."""
        if not self.db_path or not self.db_path.exists():
            return

        data = json.loads(self.db_path.read_text())
        self._next_id = data.get("next_id", 1)
        self.records = {k: CircuitRecord(**v) for k, v in data.get("records", {}).items()}

    def export_record(self, record_id: str) -> dict | None:
        """Export a record as dictionary."""
        record = self.get(record_id)
        return asdict(record) if record else None


def build_circuit_from_record(record: CircuitRecord) -> Circuit:
    """Build a SpiceLab circuit from a database record."""
    circuit = Circuit(record.name)

    # Build components from record
    for comp_name, comp_data in record.components.items():
        comp_type = comp_data["type"]

        if comp_type == "resistor":
            comp = Resistor(comp_name, resistance=comp_data["value"])
        elif comp_type == "capacitor":
            comp = Capacitor(comp_name, capacitance=comp_data["value"])
        elif comp_type == "vdc":
            comp = Vdc(comp_name, comp_data["value"])
        else:
            continue

        circuit.add(comp)

    return circuit


def main():
    """Demonstrate circuit database integration."""
    print("=" * 60)
    print("Automation: Circuit Database Integration")
    print("=" * 60)

    # Create in-memory database
    db = CircuitDatabase()

    # Add some circuit records
    print("""
   Creating Circuit Records
   ─────────────────────────────────
""")

    # Voltage divider
    record1 = CircuitRecord(
        id="",
        name="3.3V Voltage Divider",
        category="Power",
        description="Voltage divider for 3.3V MCU from 5V supply",
        components={
            "R1": {"type": "resistor", "value": 5100},
            "R2": {"type": "capacitor", "value": 10000},
            "Vin": {"type": "vdc", "value": 5.0},
        },
        parameters={
            "input_voltage": 5.0,
            "output_voltage": 3.3,
            "load_current_max": 0.001,
        },
        specifications={
            "vout_min": 3.2,
            "vout_max": 3.4,
            "power_max": 0.010,
        },
        tags=["divider", "mcu", "3v3"],
        status="approved",
    )

    # RC Filter
    record2 = CircuitRecord(
        id="",
        name="Audio Lowpass Filter",
        category="Filter",
        description="RC lowpass filter for audio applications",
        components={
            "R1": {"type": "resistor", "value": 10000},
            "C1": {"type": "capacitor", "value": 15.9e-9},
            "Vin": {"type": "vdc", "value": 1.0},
        },
        parameters={
            "cutoff_frequency": 1000,
            "order": 1,
        },
        specifications={
            "fc_min": 900,
            "fc_max": 1100,
        },
        tags=["filter", "audio", "lowpass"],
        status="reviewed",
    )

    # Decoupling network
    record3 = CircuitRecord(
        id="",
        name="IC Decoupling Network",
        category="Power",
        description="Standard decoupling for digital ICs",
        components={
            "C1": {"type": "capacitor", "value": 100e-9},
            "C2": {"type": "capacitor", "value": 10e-6},
        },
        parameters={
            "supply_voltage": 3.3,
        },
        specifications={
            "impedance_at_1mhz_max": 1.0,
        },
        tags=["decoupling", "digital", "ic"],
        status="draft",
    )

    # Add records to database
    id1 = db.add(record1)
    id2 = db.add(record2)
    id3 = db.add(record3)

    print(f"   Added: {id1} - {record1.name}")
    print(f"   Added: {id2} - {record2.name}")
    print(f"   Added: {id3} - {record3.name}")

    # Search examples
    print("""
   Database Queries
   ─────────────────────────────────
""")

    # Search by category
    power_circuits = db.search(category="Power")
    print(f"   Category 'Power': {len(power_circuits)} circuits")
    for r in power_circuits:
        print(f"      - {r.id}: {r.name} [{r.status}]")

    # Search by tags
    audio_circuits = db.search(tags=["audio"])
    print(f"\n   Tag 'audio': {len(audio_circuits)} circuits")
    for r in audio_circuits:
        print(f"      - {r.id}: {r.name}")

    # Search by status
    approved = db.search(status="approved")
    print(f"\n   Status 'approved': {len(approved)} circuits")

    # List categories and tags
    print(f"\n   Categories: {db.list_categories()}")
    print(f"   Tags: {db.list_tags()}")

    # Build circuit from record
    print("""
   Building Circuit from Record
   ─────────────────────────────────
""")

    record = db.get(id1)
    if record:
        circuit = build_circuit_from_record(record)
        result = circuit.validate()
        print(f"   Built: {record.name}")
        print(f"   Valid: {result.is_valid}")

    # Export record
    exported = db.export_record(id1)
    if exported:
        print("\n   Exported JSON preview:")
        json_str = json.dumps(exported, indent=2)
        for line in json_str.split("\n")[:10]:
            print(f"      {line}")
        print("      ...")

    print("""
   Database Features:
   ┌────────────────────────────────────────────────────────┐
   │ CRUD:     Create, Read, Update, Delete records        │
   │ Search:   Filter by category, tags, status, name      │
   │ Export:   JSON format for sharing/backup              │
   │ Persist:  Save/load to file system                    │
   │ Build:    Reconstruct circuits from records           │
   └────────────────────────────────────────────────────────┘

   Extension Ideas:
   - SQL backend for large libraries
   - Version history and diff tracking
   - Import from/export to other formats
   - Web API for team collaboration
   - Integration with component databases
""")


if __name__ == "__main__":
    main()
