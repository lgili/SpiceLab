# Environment Doctor

`python -m spicelab.doctor` inspects your system for common Circuit Toolkit
requirements and reports missing SPICE engines or shared libraries.

```bash
$ python -m spicelab.doctor
Circuit Toolkit environment check
 ✔ ngspice (/usr/local/bin/ngspice)
 ✖ ltspice
    hint: Download LTspice from Analog Devices
 ⚠ libngspice
    hint: Install libngspice (e.g. brew install libngspice or apt install libngspice0-dev)
```

Checks currently cover:

- CLI engines (`ngspice`, `ltspice`, `xyce`) via PATH discovery or the
  `SPICELAB_*` overrides
- The `libngspice` shared library used by the co-simulation backend

Results are colour-neutral (plain ASCII) so you can paste them into bug reports.
The command returns a non-zero exit code when required engines are missing,
making it suitable for CI sanity checks.
