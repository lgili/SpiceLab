from __future__ import annotations

from typing import Any

from ..core.circuit import Circuit


def connectivity_table(circuit: Circuit, *, sort: bool = True, include_type: bool = True) -> Any:
    """Return a pandas DataFrame describing circuit connectivity.

    Parameters
    ----------
    circuit:
        The circuit to introspect.
    sort:
        Sort the resulting frame by component name and port order.
    include_type:
        Include the component type column.
    """

    return circuit.connectivity_dataframe(sort=sort, include_type=include_type)


def summary_table(circuit: Circuit, *, indent: int = 2) -> str:
    """Return a formatted ASCII table summarising connectivity."""

    return circuit.summary_table(indent=indent)


def components_for_net(circuit: Circuit, net_name: str) -> list[str]:
    """Return a list of component references connected to a given net label."""

    df = circuit.connectivity_dataframe()
    if df.empty:
        return []
    series = df["net"].astype(str).str.lower()
    mask = series == net_name.lower()
    return sorted(df.loc[mask, "component"].unique())


__all__ = ["connectivity_table", "summary_table", "components_for_net"]
