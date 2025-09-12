# Troubleshooting

- ngspice not found: install it and ensure it's on PATH.
  - macOS: `brew install ngspice`
  - Ubuntu: `sudo apt install -y ngspice`
  - Windows: install Spice64 and add the `bin` to PATH
- Binary RAW: `Binary RAW not supported yet` → we force ASCII in our runner, but if you
  run NGSpice manually, set `set filetype=ascii` in a `.control` block or use `-r`.
- Unconnected port: connect every component port to a Net or another Port before building.
- Missing .model for switches: add directives with `circuit.add_directive(".model ...")`.
- AC analysis looks flat or phase missing: ensure you use small-signal sources (VA/Iac),
  or reconstruct complex traces with `ac_gain_phase` which handles re/im or mag/phase.
