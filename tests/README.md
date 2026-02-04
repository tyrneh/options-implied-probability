
# Test Suite Guide

This directory contains the comprehensive test suite for the `oipd` library.

## Directory Structure

- **`interface/`**: **Start here.** Tests the public API (`VolCurve`, `ProbCurve`, `VolSurface`, `ProbSurface`). These tests define the "contract" with the user. If you change the API, you must update these tests.
- **`core/`**: Unit tests for internal mathematical components (`svi`, `interpolation`, `numerical`). These verify that individual building blocks work correctly.
- **`regression/`**: Contains the **Golden Master** tests. These ensure that complex pipelines produce consistent numerical outputs over time.
- **`data_access/`**: Tests for data loading (CSV, DataFrame) and normalization.
- **`pipelines/`**: Integration tests for the underlying calculation pipelines.

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
