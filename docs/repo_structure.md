# Repository structure and refactor recommendations

## 1. Current layout
```
.
    CHANGELOG.md
    LICENSE
    MANIFEST.in
    README.md
    TECHNICAL_README.md
    pyproject.toml
    requirements-dev.txt
    requirements.txt
    setup.py
    examples/
        appl_example.py
        surface_example.py
        OIPD_colab_demo.ipynb
        example.ipynb
    data/
        WTIfutures_date250908exp251216_spot6169.csv
        bitcoin_date20250830_strike20251226_price108864.csv
        ussteel_date20250128_strike20251219_price3629.csv
    oipd/
        __init__.py
        pipelines/
            __init__.py
            estimator.py
            market_inputs.py
            rnd_slice.py
            rnd_surface.py
        data_access/
            __init__.py
            readers/
                __init__.py
                base.py
                csv_reader.py
                dataframe_reader.py
            vendors/
                __init__.py
                yfinance/
                    __init__.py
                    reader.py
        presentation/
            __init__.py
            iv_plotting.py
            iv_surface_3d.py
            matplot.py
            plot_rnd.py
            publication.py
        core/
            __init__.py
            errors.py
            iv.py
            parity.py
            data_processing/
                __init__.py
                dividends.py
                iv_inversion.py
                moneyness.py
                parity.py
                selection.py
                validation.py
            probability_density_conversion/
                __init__.py
                finite_diff.py
                price_curve.py
                rnd.py
            vol_surface_fitting/
                __init__.py
                api.py
                facade.py
                registry.py
                algorithms/
                    __init__.py
                    bspline/
                        __init__.py
                        fitter.py
                    svi/
                        __init__.py
                        fitter.py
            shared/
                    __init__.py
                    vol_model.py
                    svi.py
                    ssvi.py
                    svi_types.py
        pricing/
            __init__.py
            black76.py
            black_scholes.py
            utils.py
    tests/
        core/
            test_black76_integration.py
            test_parity.py
            test_prep.py
            test_rnd_surface_interface.py
            test_ssvi_calibration.py
            test_surface_diagnostics.py
            test_surface_fitting.py
            test_svi.py
            test_svi_calibration.py
        io/
            test_csv_reader.py
            resources/
                sample.csv
        pricing/test_pricing.py
        test_finite_diff.py
        test_iv_smile.py
        test_price_method.py
        test_rate_mode.py
```

## 2. Plain-English tour
- `CHANGELOG.md` – release notes for the package.
- `LICENSE` – legal terms for using the code.
- `MANIFEST.in` – packaging instructions for extra files when publishing to PyPI.
- `README.md` – high-level description, quick start, and marketing overview.
- `TECHNICAL_README.md` – deeper theory notes for the maths-minded reader.
- `example.png` – hero plot used in the README.
- `pyproject.toml` – build configuration so the package installs cleanly.
- `requirements-dev.txt` – additional libraries for contributors.
- `requirements.txt` – minimal runtime dependencies.
- `setup.py` – legacy installer hook (kept for compatibility).
- `svi_surface_fix_2.md` – project note about a prior surface-fitting fix.

- `examples/` – runnable notebooks and scripts that show end-to-end usage.
  - `OIPD_colab_demo.ipynb` – Google Colab notebook demonstrating the workflow.
  - `appl_example.py` – Apple single-expiry example script.
  - `example.ipynb` – richer notebook example with commentary.
  - `surface_example.py` – multi-expiry surface example.

- `data/` – sample option quote CSVs for offline experiments.
  - Each CSV file contains dated quotes for a different asset (WTI, bitcoin, US Steel).

- `.meta/images/` – marketing assets used in documentation.
  - PNGs and GIFs referenced by README and tutorials.

- `oipd/` – the main library.
  - `__init__.py` – re-exports the public API, now backed by the `pipelines/` package.
  - `pipelines/` – orchestration entry points. `rnd_slice.py` implements the single-expiry pipeline (and the `RND` façade). `rnd_surface.py` implements the multi-expiry pipeline and the `RNDSurface` façade.
  - `core/` – maths, data-prep, and calibration primitives.
    - `data_processing/` – parity application, price selection, IV inversion, normalization utilities, and staleness filtering.
    - `probability_density_conversion/` – price-curve generation, finite differences, and PDF/CDF utilities (formerly `core/density.py`).
    - `vol_surface_fitting/` – registry-based SVI/bspline implementations with shared constraints, objectives, and transforms, including the SSVI surface calibrator.
  - `data_access/` – ingestion and vendor access.
    - `readers/` – `AbstractReader` base class plus CSV/DataFrame implementations.
    - `vendors/` – registry of vendor readers (Yahoo Finance moved here).
  - `presentation/` – plotting utilities (`iv_plotting`, `plot_rnd`, `iv_surface_3d`, etc.).
  - `pipelines/estimator.py` – retains the low-level `_estimate` routine and data-source wrappers consumed by the pipeline.
  - `pricing/` – quantitative pricing engines (Black-76/Black-Scholes) and utilities.

- `.vscode/` – editor settings for contributors using VS Code.
  - `launch.json` – debugger presets.
  - `settings.json` – workspace defaults.

- `tests/` – automated checks covering each module.
  - Top-level tests verify finite differences, smiles, price methods, and rate modes.
  - `tests/core/` – unit tests for maths and prep utilities.
  - `tests/pricing/` – pricing engine tests.
  - `tests/io/` – reader tests with sample CSV fixtures.


## 5. Mapping your 3 steps to the code
Your pipeline naturally splits by responsibility. Keep the code grouped by domain responsibilities, then expose end‑to‑end “pipelines” that orchestrate the steps.

1) Preprocessing of data (before calibration)
  - Already in: `oipd/core/data_processing/` (parity, IV extraction).
   - Typical tasks (beyond the bullets you listed):
     - Schema normalization and column mapping (vendor → standard), option type unification.
     - Time to expiry resolution and discount factor computation; day‑count consistency.
     - Dividend/rate resolution (spot vs forward engine selection; `black76` vs `bs`).
     - Put‑call parity and forward inference; OTM‑only filtering and synthetic calls.
     - Price column selection (mid/last with fallbacks), staleness and illiquidity filters.
     - Moneyness/log‑moneyness conversion for model‑friendly parameterizations.
     - IV extraction from prices (robustly, with solver selection and tolerances).
     - Optional: outlier detection/reweighting (bid‑ask width, robust z‑scores), basic arbitrage diagnostics and minimal repairs.

2) Fitting the IV smile/surface
   - Single expiry (slice): SVI (preferred), b‑spline (legacy/backstop).
   - Surface (term structure): SSVI (arbitrage‑aware).
  - Already in: `oipd/core/vol_surface_fitting/` (shared SVI/SSVI maths and surface calibrators).
   - Practicalities: initialize on log‑moneyness grids; enforce bounds; optionally weight by bid‑ask/volume; add regularization; check butterfly/calendar conditions.

3) Converting the fitted IV smile into a PDF
   - Deterministic: evaluate prices on a strike grid, then apply Breeden‑Litzenberger via stable finite differences.
  - Already in: `oipd/core/probability_density_conversion/` (`price_curve_from_iv`, `pdf_from_price_curve`, `calculate_cdf_from_pdf`).
   - Practicalities: choose grid/spacing, smooth before differentiation (e.g., cubic splines), handle tails and boundary conditions, renormalize for numerical drift.

## 6. Should `interface` be a single file or a folder?
- If you only expose one or two façade classes (e.g., `RND`, `RNDSurface`) and they stay small, a single file can work.
- As soon as orchestration grows (multiple façades, CLI helpers, presets), prefer a small `interface/` package to keep responsibilities crisp:
  - `oipd/interface/__init__.py` — public exports.
  - (Optional) If the interface grows, introduce dedicated modules under `oipd/interface/`; currently the package simply re-exports from `pipelines`.

This mirrors how `pipelines/` would hold the orchestration (imperative flow), while `interface/` keeps user‑facing APIs tidy and stable.

## 7. Preprocessing checklist (actionable)
Use this as a guardrail when adding vendors or calibrators:
- Input normalization and validation (schema, units, day‑count).
- Price column selection and fallbacks.
- Parity/forward inference; synthetic‑call conversion; OTM filtering.
- Staleness/illiquidity filter; basic arbitrage diagnostics.
- Engine selection (`black76` vs `bs`) and dividend/rate resolution.
- Moneyness/log‑moneyness transform and strike grid selection.
- IV extraction with solver selection and tolerance.
- Optional: weighting by bid‑ask/volume; outlier down‑weighting.

## 8. Target structure to migrate to
This is the concrete layout we will move toward. The existing modules map cleanly and we can add thin re‑exports to keep API/tests stable during migration.

- Top level
  - `core/` — domain logic (pure computation and constraints). In this plan, the “domain” lives under `core/`.
  - `data_access/` — readers, vendor adapters, schema mapping, ingestion validation.
  - `pipelines/` — end‑to‑end orchestrators (`rnd_slice.py`, `rnd_surface.py`).
  - `presentation/` — plotting and reporting utilities.
  - `interface.py` (or `interface/`) — public API façade re‑exporting pipelines and types.

- `core/` subpackages
  - `core/data_processing/`
    - `parity.py` — forward inference, OTM/ITM split, synthetic calls.
    - `selection.py` — price column selection (mid/last), staleness/illiquidity filters.
    - `iv_inversion.py` — vectorized IV extraction (solver choice, tolerances).
    - `moneyness.py` — transforms (strike/forward ↔ log‑moneyness), grid helpers.
    - `dividends.py` — dividend/engine resolution; discount factors, time conventions.
    - `validation.py` — schema normalization, column mapping, unit/day‑count checks.

  - `core/vol_surface_fitting/`
    - See MECE breakdown in Section 9 below.

  - `core/probability_density_conversion/`
    - `price_curve.py` — call price evaluation from a vol curve (engine‑agnostic wrapper).
    - `finite_diff.py` — stable finite differences (uniform and non‑uniform fallback).
    - `rnd.py` — Breeden–Litzenberger PDF, CDF, normalization, tail handling.

- `data_access/`
  - `readers/` — `csv_reader.py`, `dataframe_reader.py`, `base.py` (schema normalization).
  - `vendors/` — vendor adapters (e.g., `yfinance/reader.py`), optional caching.
  - `validation.py` — ingestion‑time schema checks and coercions.

- `pipelines/`
  - `rnd_slice.py` — orchestrate: load/validate → data_processing → vol_surface_fitting → rnd → result object.
  - `rnd_surface.py` — multi‑expiry orchestration (term structure), same boundaries.

- `presentation/`
  - Move `oipd/graphics/*` here (re‑export under old path for compatibility during migration).

- `interface.py` or `interface/`
  - Currently re-exports the pipeline façades; expand only if additional presentation helpers are needed.

## 9. Vol surface fitting (final structure)
Organize by algorithm and shared math. Keep the public API stable through a façade and a registry. Avoid a generic “utils” module — use a focused `shared/` package instead.

- Core modules (stable API)
  - `api.py` — Protocols and core types (e.g., `VolCurve`, `IVSurface`, `SliceFitter`, `SurfaceFitter`).
  - `registry.py` — register/lookup of fitter implementations by ID (e.g., `"svi"`, `"bspline"`, `"ssvi"`, `"stitched_svi"`).
  - `facade.py` — stable entry points and compatibility shims: `AVAILABLE_SURFACE_FITS`, `fit_slice(...)`, `fit_surface(...)`.

- `shared/` (reusable math and primitives)
  - `parametrizations.py` — SVI/SSVI total variance forms w(k), derivatives; JW conversions; parameter dataclasses.
  - `transforms.py` — strike/forward ↔ log‑moneyness; IV ↔ total variance; time normalization.
  - `constraints.py` — butterfly positivity, calendar monotonicity, parameter bounds and projections.
  - `objectives.py` — residuals and regularizers (smoothness, wings, calendar); loss assembly.
  - `weighting.py` — bid‑ask/volume weights; robust (Huber/Tukey); outlier down‑weighting.
  - `grid.py` — log‑moneyness grids; tail anchoring; resampling strategies.
  - `diagnostics.py` — post‑fit checks: arbitrage flags, curvature/wing slopes; summaries.
  - `types.py` — typed containers for options, params, results, diagnostics; shared enums.
  - `errors.py` — calibration‑specific exceptions.

- `algorithms/` (each method in its own subpackage)
  - `svi/`
    - `fitter.py` — single‑expiry SVI calibration orchestration.
    - `init.py` — initial guesses and heuristics.
    - `params.py` — SVI‑specific options (bounds, restarts, tolerances).
    - `result.py` — slice result (params, `VolCurve`, diagnostics, metadata).
  - `ssvi/`
    - `fitter.py` — multi‑expiry SSVI calibration (arbitrage‑aware across expiries).
    - `init.py` — surface initialization and smoothing.
    - `params.py` — SSVI‑specific options (calendar penalties, smoothing).
    - `result.py` — surface result (per‑expiry params, `IVSurface`, diagnostics).
  - `bspline/`
    - `fitter.py` — b‑spline fit/evaluate (legacy fallback).
    - `basis.py` — basis and smoothing primitives.
    - `params.py` — bspline options (smoothing_factor, degree).
  - `stitched_svi/`
    - `fitter.py` — per‑expiry SVI fits stitched with cross‑expiry penalties.
    - `params.py` — stitching penalties/weights.
    - `result.py` — surface result (slice params + stitching diagnostics).

- Public API behavior
  - `fit_slice(method=...)` dispatches via `registry.py` to algorithm fitters and returns a `VolCurve` and typed result.
  - `fit_surface(method=...)` dispatches to SSVI or stitched SVI implementations and returns an `IVSurface` and typed result.
  - `AVAILABLE_SURFACE_FITS` is derived from the registry so adding a new algorithm is registration‑only.

- Compatibility during migration
  - Keep `oipd/core/surface_fitting.py` as a thin façade forwarding to `facade.py` until callers/tests are updated.

## 10. Migration mapping (minimal churn)
- `oipd/core/parity.py` → `core/data_processing/parity.py`
- `oipd/core/data_processing/iv.py` houses the IV solvers (former `core/iv.py`).
- `oipd/core/vol_surface_fitting/shared/svi.py` houses single-slice SVI maths and calibration helpers; see `algorithms/svi/` for orchestrators.
- `oipd/core/surface_fitting.py` → `vol_surface_fitting/slice_fit.py` + `bspline.py` (+ façade for `fit_surface`)
- `oipd/core/density.py` → `probability_density_conversion/{price_curve.py, finite_diff.py, rnd.py}`
- `oipd/io/*` → `data_access/readers/*`; `oipd/vendor/*` → `data_access/vendors/*`
- `oipd/graphics/*` → `presentation/*` (with re‑exports during transition)

Add temporary re‑exports at old paths to keep tests/imports working, then flip imports progressively.
