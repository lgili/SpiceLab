"""Example: construct an AnalogMux8, print its netlist card and a DOT diagram.

Run as module: python -m examples.analog_mux_demo
"""

from cat.core.components import AnalogMux8


def mux_to_dot(mux: AnalogMux8, net_of) -> str:
    """Return a DOT (Graphviz) string representing mux topology.

    Nodes are nets; edges are resistors or switches labeled by element.
    """
    lines = ["strict graph mux {", "  rankdir=LR;"]
    in_name = net_of(mux._ports[0])
    outs = [mux._ports[1 + i] for i in range(8)]
    # show connections
    for i, out in enumerate(outs):
        out_name = net_of(out)
        label = f"{mux.ref}_{i}"
        lines.append(f'  "{in_name}" -- "{out_name}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def main():
    mux = AnalogMux8(ref="MU1", r_series=100, sel=4)

    def net_of(p):
        # simple debug mapping
        return p.name

    print("--- SPICE card ---")
    print(mux.spice_card(net_of))

    dot = mux_to_dot(mux, net_of)
    print("--- DOT ---")
    print(dot)

    try:
        from graphviz import Source

        s = Source(dot)
        svg = s.pipe(format="svg")
        with open("site/assets/mux_demo.svg", "wb") as fh:
            fh.write(svg)
        print("Wrote site/assets/mux_demo.svg")
    except Exception:
        print("graphviz not available; skipping svg generation")


if __name__ == "__main__":
    main()
