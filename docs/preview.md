# Circuit Preview (Summary and Graphviz)

Use the new preview helpers to validate connectivity and generate a quick schematic-like view.

## Connectivity summary
```python
print(c.summary())
```
Outputs a compact mapping of `ref.port -> net`, highlights nets with degree 1 and unconnected ports.

## Graphviz rendering
```python
dot = c.to_dot()
ok = c.render_svg("circuit_preview.svg")  # requires 'dot' in PATH
```

If Graphviz isn't installed, you still get a `.dot` file:
```bash
dot -Tsvg circuit_preview.dot -o preview.svg
```

Example (RC low-pass):

![Circuit preview](assets/examples/circuit_preview.svg)
