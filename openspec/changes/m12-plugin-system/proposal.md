# M12 - Plugin System and API Stability

## Why
To build a sustainable ecosystem, SpiceLab needs a plugin system allowing third-party extensions (custom measurements, engines, readers). Additionally, API stability with SemVer guarantees and deprecation policies is critical for production use and library adoption.

## What Changes
- Implement plugin system using Python entry points for: measurements, readers, engines, optimizers
- Create `spicelab.plugins` registry with discovery and validation
- Establish SemVer versioning policy and deprecation guidelines (2 minor versions minimum)
- Create `spicelab._compat` module for backward compatibility shims
- Add API stability tests (import smoke tests, signature validation)
- Set up automated release process (tag → build → PyPI) with changelog generation
- Add optional opt-in telemetry for feature usage tracking (anonymized, no circuit data)

## Impact
- **Affected specs**: plugins, api-stability, versioning, release-process
- **Affected code**:
  - New: `spicelab/plugins/` with plugin registry and loaders
  - New: `spicelab/_compat.py` for deprecation shims
  - Modified: All public APIs with version decorators
  - New: `tests/test_api_stability.py`
  - Modified: CI/CD for automated releases
- **Dependencies**: setuptools (entry points), packaging (version parsing)
