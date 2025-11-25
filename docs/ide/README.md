# IDE Setup for SpiceLab

This directory contains IDE configuration files for enhanced SpiceLab development.

## VSCode Snippets

To use the SpiceLab VSCode snippets:

1. Copy `vscode-snippets.json` to your `.vscode/` directory
2. Rename it to `spicelab.code-snippets`

```bash
cp docs/ide/vscode-snippets.json .vscode/spicelab.code-snippets
```

### Available Snippets

| Prefix | Description |
|--------|-------------|
| `spicelab-circuit` | Basic circuit setup with imports |
| `spicelab-resistor` | Create a resistor component |
| `spicelab-capacitor` | Create a capacitor component |
| `spicelab-inductor` | Create an inductor component |
| `spicelab-vdc` | Create a DC voltage source |
| `spicelab-vac` | Create an AC voltage source |
| `spicelab-vpulse` | Create a pulse voltage source |
| `spicelab-net` | Create a net |
| `spicelab-connect` | Connect component port to net |
| `spicelab-voltage-divider` | Complete voltage divider circuit |
| `spicelab-rc-lowpass` | RC lowpass filter circuit |
| `spicelab-inverting-amp` | Inverting amplifier with op-amp |
| `spicelab-library` | Create component from library |
| `spicelab-template-filter` | Create filter using template |
| `spicelab-validate` | Validate circuit and print errors |
| `spicelab-monte-carlo` | Monte Carlo tolerance analysis setup |
| `spicelab-eseries` | Find nearest E-series resistor value |

## PyCharm Live Templates

For PyCharm users, import the snippets as Live Templates:

1. Go to Settings > Editor > Live Templates
2. Create a new Template Group called "SpiceLab"
3. Add templates based on the snippets in `vscode-snippets.json`

## Type Hints

SpiceLab uses modern Python type hints throughout the codebase. VSCode and PyCharm
will automatically provide autocomplete suggestions based on these hints.

For best results, ensure you have:
- Python extension (VSCode) or
- Python plugin (PyCharm)

The type hints in the source code provide:
- Parameter types and return types
- Docstrings with examples
- Module-level `__all__` exports
