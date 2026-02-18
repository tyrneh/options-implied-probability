
# Test Suite Guide

This directory contains the comprehensive test suite for the `oipd` library.

## CI Determinism Gate (SVI, Linux)

- The CI workflow includes a **hard determinism gate** for SVI calibration on Linux.
- This gate runs two seeded calibrations with identical inputs and compares all tracked fields using exact equality.
- If any field differs, CI fails.
- The tracked payload includes SVI params and key diagnostics:
  - Params: `a`, `b`, `rho`, `m`, `sigma`, `forward`, `maturity_years`
  - Diagnostics: `objective`, `rmse_weighted`, `rmse_unweighted`, `envelope_violations_pct`, `chosen_start_origin`, `chosen_start_index`

### Required Policy

- You **MUST** keep the seeded SVI determinism gate passing on Linux CI.
- You **MUST NOT** weaken this check by loosening tolerance without explicit, documented justification.
- Exact-equality policy applies to the deterministic payload, with paired `NaN` treated as equal.

### Troubleshooting Determinism Failures

If the determinism gate fails, check these in order before touching tolerances:
1. Confirm `random_seed` is explicitly set and actually propagated to SVI calibration.
2. Confirm dependency/version drift (`numpy`, `scipy`, `pandas`) from CI logs.
3. Confirm threaded numeric libraries are pinned for deterministic execution (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` set to `1`).
4. Investigate objective/diagnostic field diffs printed by CI to isolate the first diverging component.

## Interface Test Policy (Performance + Coverage)

### Fixture Scope Rules

- Expensive calibration setup (`VolCurve.fit`, `VolSurface.fit`, implied distribution build) **MUST** use module-scoped fixtures for smoke and contract checks.
- Function-scoped expensive fixtures are allowed only when a test intentionally mutates state and needs isolation.
- Shared canonical interface fixtures are defined in `tests/interface/conftest.py` and **SHOULD** be reused instead of duplicated per test module.

### Duplicate Test Rules

- Plotting smoke tests **MUST** live only in `tests/interface/test_visualization.py`.
- Class-specific interface files **MUST NOT** duplicate plotting smoke checks that are already covered there.
- Keep behavioral plotting assertions (for example interpolation-grid or fan-chart continuity behavior) in class-specific files when they test unique logic.

### Coverage Rules

- Public interface contract coverage remains one-test-per-method in `tests/interface/test_api_contract.py`.
- Refactors **MUST NOT** remove method-level contract coverage for `VolCurve`, `VolSurface`, `ProbCurve`, or `ProbSurface`.
- Interface tests **SHOULD** assert user-facing behavior; avoid private-attribute assertions unless no public signal exists.

## Directory Structure

- **`interface/`**: **Start here.** Tests the public API (`VolCurve`, `ProbCurve`, `VolSurface`, `ProbSurface`). These tests define the "contract" with the user. If you change the API, you must update these tests.
- **`core/`**: Unit tests for internal mathematical components (`svi`, `interpolation`, `numerical`). These verify that individual building blocks work correctly.
- **`regression/`**: Contains the **Golden Master** tests. These ensure that complex pipelines produce consistent numerical outputs over time.
- **`data_access/`**: Tests for data loading (CSV, DataFrame) and normalization.
- **`pipelines/`**: No dedicated suite currently; add explicit tests here when pipeline integration coverage is introduced.

## Running Tests

Run the full suite:
```bash
pytest tests/
```

Run specific categories:
```bash
pytest tests/interface/    # Verify public API
pytest tests/regression/   # Verify numerical consistency
```

## Maintenance Guide

### 1. Modifying Internal Logic (Refactoring)
If you optimize or refactor internal math (e.g., SVI calibration, interpolation) **without** changing the public API:
- Run `pytest tests/interface/`. These should ALL pass without changes.
- Run `pytest tests/regression/`. These should pass.
- If regression tests fail but you believe your new math is "better/correct":
    1. Verify the results manually.
    2. Regenerate the Golden Master data (see below).

### 2. Changing the API
If you add arguments or change method signatures in `VolCurve` or `VolSurface`:
- **Update `tests/interface/`**: Modify the tests to reflect the new API usage. These tests are your documentation.
- **Update `generate_golden_master.py`**: If the start-up sequence changes, you'll need to update the generator script to match.

### 3. Golden Master / Regression
The file `tests/data/golden_master.json` contains "known good" outputs for a specific AAPL option chain.
**If you intentionally change mathematical models** (e.g., switch to a better SVI solver, fix a bug in PDF integration):
1. Run the generator script:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python tests/data/generate_golden_master.py
   ```
2. Commit the updated `golden_master.json`.
3. Verify `tests/regression/test_golden_master.py` passes.

### 4. Adding New Features
- **New Core Math**: Add unit tests in `tests/core/`.
- **New Public Method**: Add a test in `tests/interface/` showing how a user calls it.
