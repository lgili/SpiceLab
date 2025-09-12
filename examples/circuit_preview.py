from __future__ import annotations

from cat import GND, C, Circuit, R, V


def build() -> Circuit:
    c = Circuit("preview_rc")
    V1, R1, C1 = V(5.0), R("1k"), C("100n")
    c.add(V1, R1, C1)
    # Name nets for readability in netlist and traces (explicit nets)
    from cat.core.net import Net

    vin = Net("vin")
    vout = Net("vout")
    c.connect(V1.ports[0], vin)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


def main() -> None:
    c = build()
    print("=== Connectivity Summary ===")
    print(c.summary())
    # Export DOT and try rendering to SVG (requires Graphviz 'dot')
    ok = c.render_svg("circuit_preview.svg")
    if ok:
        print("[saved] circuit_preview.svg")
    else:
        with open("circuit_preview.dot", "w", encoding="utf-8") as f:
            f.write(c.to_dot())
        print("[saved] circuit_preview.dot (install graphviz to render SVG)")


if __name__ == "__main__":
    main()
