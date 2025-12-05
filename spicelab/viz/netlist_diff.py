"""Netlist diff visualization tool.

Provides side-by-side and unified diff views for comparing SPICE netlists,
with syntax highlighting and color-coded changes.

Example:
    >>> from spicelab.viz.netlist_diff import NetlistDiff
    >>> diff = NetlistDiff(netlist1, netlist2)
    >>> diff.print_unified()  # Print unified diff to terminal
    >>> diff.print_side_by_side()  # Print side-by-side comparison
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

__all__ = ["NetlistDiff", "DiffResult"]


@dataclass
class DiffResult:
    """Result of a netlist diff operation.

    Attributes:
        added_lines: Lines only in the second netlist.
        removed_lines: Lines only in the first netlist.
        unchanged_lines: Lines present in both netlists.
        similarity_ratio: Float 0.0-1.0 indicating how similar the netlists are.
    """

    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    unchanged_lines: list[str] = field(default_factory=list)
    similarity_ratio: float = 0.0

    @property
    def has_changes(self) -> bool:
        """Return True if there are any differences."""
        return bool(self.added_lines or self.removed_lines)

    @property
    def summary(self) -> str:
        """Return a brief summary of changes."""
        if not self.has_changes:
            return "No changes"
        parts = []
        if self.added_lines:
            parts.append(f"+{len(self.added_lines)} added")
        if self.removed_lines:
            parts.append(f"-{len(self.removed_lines)} removed")
        return ", ".join(parts)


class NetlistDiff:
    """Compare two SPICE netlists with visualization.

    Provides unified and side-by-side diff views with syntax highlighting.

    Args:
        netlist1: First netlist (original).
        netlist2: Second netlist (modified).
        name1: Label for first netlist (default: "original").
        name2: Label for second netlist (default: "modified").

    Example:
        >>> from spicelab.templates import rc_lowpass
        >>> c1 = rc_lowpass(fc=1000)
        >>> c2 = rc_lowpass(fc=2000)
        >>> diff = NetlistDiff(c1.build_netlist(), c2.build_netlist())
        >>> diff.print_unified()
    """

    def __init__(
        self,
        netlist1: str,
        netlist2: str,
        name1: str = "original",
        name2: str = "modified",
    ) -> None:
        self.netlist1 = netlist1
        self.netlist2 = netlist2
        self.name1 = name1
        self.name2 = name2
        self._lines1 = netlist1.strip().splitlines()
        self._lines2 = netlist2.strip().splitlines()
        self._result: DiffResult | None = None

    @property
    def result(self) -> DiffResult:
        """Get the diff result (computed lazily)."""
        if self._result is None:
            self._result = self._compute_diff()
        return self._result

    def _compute_diff(self) -> DiffResult:
        """Compute the diff between the two netlists."""
        matcher = difflib.SequenceMatcher(None, self._lines1, self._lines2)
        ratio = matcher.ratio()

        added: list[str] = []
        removed: list[str] = []
        unchanged: list[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                unchanged.extend(self._lines1[i1:i2])
            elif tag == "replace":
                removed.extend(self._lines1[i1:i2])
                added.extend(self._lines2[j1:j2])
            elif tag == "delete":
                removed.extend(self._lines1[i1:i2])
            elif tag == "insert":
                added.extend(self._lines2[j1:j2])

        return DiffResult(
            added_lines=added,
            removed_lines=removed,
            unchanged_lines=unchanged,
            similarity_ratio=ratio,
        )

    def unified_diff(self, context_lines: int = 3) -> str:
        """Generate a unified diff string.

        Args:
            context_lines: Number of context lines around changes.

        Returns:
            Unified diff as a string.
        """
        diff = difflib.unified_diff(
            self._lines1,
            self._lines2,
            fromfile=self.name1,
            tofile=self.name2,
            lineterm="",
            n=context_lines,
        )
        return "\n".join(diff)

    def print_unified(
        self,
        context_lines: int = 3,
        console: Console | None = None,
    ) -> None:
        """Print unified diff with syntax highlighting.

        Args:
            context_lines: Number of context lines around changes.
            console: Rich console to use (creates new one if None).
        """
        if console is None:
            console = Console()

        diff_text = self.unified_diff(context_lines)

        if not diff_text:
            console.print("[green]No differences found[/green]")
            return

        # Create colored output
        output = Text()
        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                output.append(line + "\n", style="bold")
            elif line.startswith("@@"):
                output.append(line + "\n", style="cyan")
            elif line.startswith("+"):
                output.append(line + "\n", style="green")
            elif line.startswith("-"):
                output.append(line + "\n", style="red")
            else:
                output.append(line + "\n")

        panel = Panel(
            output,
            title=f"Netlist Diff: {self.name1} → {self.name2}",
            subtitle=f"Similarity: {self.result.similarity_ratio:.1%}",
            border_style="blue",
        )
        console.print(panel)

    def print_side_by_side(
        self,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        """Print side-by-side comparison with highlighting.

        Args:
            console: Rich console to use (creates new one if None).
            width: Maximum width for each column (auto-detected if None).
        """
        if console is None:
            console = Console()

        table = Table(
            title=f"Netlist Comparison: {self.name1} vs {self.name2}",
            show_header=True,
            header_style="bold",
        )
        table.add_column(self.name1, style="dim", overflow="fold")
        table.add_column(self.name2, overflow="fold")
        table.add_column("Status", justify="center", width=8)

        matcher = difflib.SequenceMatcher(None, self._lines1, self._lines2)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for line in self._lines1[i1:i2]:
                    table.add_row(line, line, "=")
            elif tag == "replace":
                left_lines = self._lines1[i1:i2]
                right_lines = self._lines2[j1:j2]
                max_len = max(len(left_lines), len(right_lines))
                for i in range(max_len):
                    left = left_lines[i] if i < len(left_lines) else ""
                    right = right_lines[i] if i < len(right_lines) else ""
                    left_text = Text(left, style="red") if left else Text("")
                    right_text = Text(right, style="green") if right else Text("")
                    table.add_row(left_text, right_text, "~")
            elif tag == "delete":
                for line in self._lines1[i1:i2]:
                    table.add_row(Text(line, style="red"), "", "-")
            elif tag == "insert":
                for line in self._lines2[j1:j2]:
                    table.add_row("", Text(line, style="green"), "+")

        console.print(table)
        console.print(
            f"\n[dim]Similarity: {self.result.similarity_ratio:.1%} | "
            f"{self.result.summary}[/dim]"
        )

    def print_summary(self, console: Console | None = None) -> None:
        """Print a brief summary of differences.

        Args:
            console: Rich console to use (creates new one if None).
        """
        if console is None:
            console = Console()

        result = self.result

        if not result.has_changes:
            console.print("[green]✓ Netlists are identical[/green]")
            return

        console.print("[bold]Netlist Diff Summary[/bold]")
        console.print(f"  Similarity: {result.similarity_ratio:.1%}")
        console.print(f"  Added lines: [green]+{len(result.added_lines)}[/green]")
        console.print(f"  Removed lines: [red]-{len(result.removed_lines)}[/red]")
        console.print(f"  Unchanged lines: {len(result.unchanged_lines)}")

        if result.removed_lines:
            console.print("\n[red]Removed:[/red]")
            for line in result.removed_lines[:5]:  # Show first 5
                console.print(f"  [red]- {line}[/red]")
            if len(result.removed_lines) > 5:
                console.print(f"  [dim]... and {len(result.removed_lines) - 5} more[/dim]")

        if result.added_lines:
            console.print("\n[green]Added:[/green]")
            for line in result.added_lines[:5]:  # Show first 5
                console.print(f"  [green]+ {line}[/green]")
            if len(result.added_lines) > 5:
                console.print(f"  [dim]... and {len(result.added_lines) - 5} more[/dim]")


def diff_circuits(
    circuit1: Any,
    circuit2: Any,
    name1: str = "original",
    name2: str = "modified",
    mode: Literal["unified", "side_by_side", "summary"] = "unified",
) -> NetlistDiff:
    """Compare two circuits and display the diff.

    Convenience function that builds netlists and shows the comparison.

    Args:
        circuit1: First circuit object.
        circuit2: Second circuit object.
        name1: Label for first circuit.
        name2: Label for second circuit.
        mode: Display mode ("unified", "side_by_side", or "summary").

    Returns:
        NetlistDiff object for further analysis.

    Example:
        >>> from spicelab.templates import rc_lowpass
        >>> c1 = rc_lowpass(fc=1000)
        >>> c2 = rc_lowpass(fc=2000)
        >>> diff_circuits(c1, c2, name1="1kHz", name2="2kHz")
    """
    netlist1 = circuit1.build_netlist()
    netlist2 = circuit2.build_netlist()

    diff = NetlistDiff(netlist1, netlist2, name1, name2)

    if mode == "unified":
        diff.print_unified()
    elif mode == "side_by_side":
        diff.print_side_by_side()
    elif mode == "summary":
        diff.print_summary()

    return diff
