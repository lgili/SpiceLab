# Environment Doctor

`python -m spicelab.doctor` inspects your system for common spicelab
requirements and reports missing SPICE engines or shared libraries.

```bash
$ python -m spicelab.doctor
spicelab environment check
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

## LLM-assisted Q&A (optional)

If you install the optional AI extra and configure an OpenAI-compatible API, the doctor can
answer questions and propose remediation steps. This is entirely optional and disabled by default.

Installation options:

```bash
pip install spicelab[ai]
# or
uv add 'spicelab[ai]'
```

Environment variables:

- `OPENAI_API_KEY`: your API key (required)
- `OPENAI_BASE_URL`: optional, for self-hosted or compatible services
- `OPENAI_MODEL`: default model name (can be overridden with `--model`)

Usage examples:

```bash
# Text output
spicelab-doctor --ask "How do I measure GBW from AC data?"

# JSON output (machine-readable)
spicelab-doctor --ask "Give me a minimal RC netlist" --format json

# Force a specific model for this request
spicelab-doctor --ask "Diagnose ngspice install issues on macOS" --model gpt-4o-mini
```

When `--format json` is used with `--ask`, the output includes either the structured fields
(`intent`, `summary`, `steps`, `commands`, `circuit_snippet`, `notes`) or a `raw` field when
the model returns non-JSON content.

### Troubleshooting LLM

If the LLM integration doesn’t behave as expected, try these tips:

- Non-JSON output: Use `--format json`. If the model returns non-JSON content, the doctor
   prints a `raw` field with the original text. Re-try the question with clearer wording or
   shorter prompts.
- Timeouts or slow responses: Networks and providers can be slow or rate-limited. Re-try the
   request, or reduce the prompt size. If you’re using a self-hosted endpoint, check its logs and
   capacity. The underlying SDK supports a request timeout; you can open an issue to request a
   CLI flag if you need it.
- Rate limits: Reduce frequency, stagger requests, or use a smaller/lower-cost model.
- Model errors: Verify `OPENAI_MODEL` exists for your provider, and that `OPENAI_BASE_URL` is
   correct for self-hosted deployments.
