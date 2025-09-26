"""Demonstrate reading LTspice/ngspice/Xyce raw/prn files.

Place a path to an existing .raw/.prn file as an argument to try reading it. If
no file is provided the script prints usage hints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spicelab.io import readers


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read a SPICE raw/prn file and show variables")
    p.add_argument("path", nargs="?", default=None, help="Path to .raw/.prn file to read")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.path is None:
        print(
            "Pass a path to a .raw / .prn file to inspect. "
            "Example: python examples/read_raw_demo.py mysim.raw"
        )
        return
    p = Path(args.path)
    if not p.exists():
        print(f"File {p} not found")
        return

    try:
        # try common readers in sequence
        if p.suffix.lower() in {".raw"}:
            ts = readers.read_ltspice_raw(str(p))
        elif p.suffix.lower() in {".prn", ".csv"}:
            ts = readers.read_xyce_prn(str(p))
        else:
            # fallback to ngspice parser which handles ASCII raw too
            ts = readers.read_ngspice_raw(str(p))
    except Exception as exc:
        print("Failed to read file:", exc)
        return

    print("Traces:", ts.names)
    print("First coord name:", ts.x.name)


if __name__ == "__main__":
    main()
