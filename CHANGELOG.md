
# Changelog
All notable changes to **oipd** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.4] - 2026-04-22
### Probability Domain and CDF Improvements
- Added full-domain probability grids for `ProbCurve` and `ProbSurface`,
  including smart `grid_points=None` sizing and explicit `full_domain=True`
  exports/plots.
- CDFs now use the direct call-price derivative formula instead of integrating
  the PDF.
- `density_results()` and `plot()` now default to compact, smooth view domains;
  `points` controls view resolution, while `grid_points` controls native
  numerical resolution.

### Lazy Probability Materialization
- Added lazy probability materialization for `ProbCurve` and `ProbSurface`.
  Plainly: probability objects store the fitted-volatility recipe first and
  build price/PDF/CDF arrays only when queried, exported, or plotted.
- `implied_distribution()` now accepts `grid_points=None` and
  `cdf_violation_policy="warn"`.

### Warning Diagnostics
- Added `warning_diagnostics` events and summaries on public curve and surface
  objects, with concise grouped warnings instead of repeated row/expiry noise.
- Added warn-and-repair CDF monotonicity handling via
  `cdf_violation_policy="warn"`; strict users can choose `"raise"`.
- Added diagnostics for stale quotes, price fallback, SVI butterfly risk,
  skipped expiries, CDF repairs, and fan-chart skips.

### Market Data, Forward Inference, and SVI Robustness
- Improved Black-76 put-call parity forward inference: nearest-ATM selection,
  selected-subset outlier filtering, richer diagnostics, and no influence from
  far-from-ATM pairs that merely pass validation.
- Temporarily disabled the default bid/ask relative-spread gate; explicit
  `max_bid_ask_relative_spread` values still apply.
- Added bid/ask-only and wide `call_price`/`put_price` ingestion, raw yfinance
  spot handling, and lowercase `volume` preservation.
- Fixed SVI weighting so reliable bid/ask spreads take precedence over volume,
  with diagnostics for the chosen auxiliary weight source and fallback reason.

### Developer Tooling and Release Hygiene
- Added non-mutating golden-master drift comparison tooling.
- Added optional `experiments` dependencies for reproducible experiment runners.
- Updated docs and quickstart notebooks for the new probability-grid and
  warning-diagnostics behavior.
- Bumped the package version to `2.0.4`.

### Removed
- Removed the old integrated-PDF CDF path and legacy normalized surface-CDF
  helpers.

## [2.0.3]
### Changed
- Finalized the maturity contract around three explicit fields:
    - `time_to_expiry_years` for pricing and calibration
    - `time_to_expiry_days` for continuous day-based reporting
    - `calendar_days_to_expiry` for integer calendar-bucket reporting

### Removed
- Removed the old `days_to_expiry` compatibility path from active APIs.

## [2.0.2] - 2026-03-06
### Added
- Added stable DataFrame export methods for fitted results:
    - `ProbCurve.density_results(domain=None, points=200)`
    - `ProbSurface.density_results(domain=None, points=200, start=None, end=None, step_days=1)`
    - extended `VolSurface.iv_results(domain=None, points=200, include_observed=True, start=None, end=None, step_days=1)`

### Changed
- Surface DataFrame exports now default to a daily expiry grid (`step_days=1`).
- When `start` and `end` are omitted, surface exports automatically span the first and last fitted pillar expiries.
- Fitted pillar expiries are always included in surface exports, even when they fall off the stepped calendar grid.

## [2.0.1] - 2025-02-17
### Code improvements
- Reworked `ProbSurface` to use fitted `VolSurface` as the canonical source of truth. Probability is now derived from interpolated volatility and option prices at query time, rather than interpolating probabilities directly. This is important because only the fitted vol smiles can be linearly interpolated in total-variance space; probabilities can't be interpolated directly. 
    - Added direct surface query methods for arbitrary maturities: `pdf(price, t)`, `cdf(price, t)`, `quantile(q, t)`, and callable alias `__call__(price, t)`.
    - Extended `ProbSurface.slice(expiry)` to support arbitrary (non-pillar) expiries/slicing
    - Added a dedicated surface probability pipeline (`oipd/pipelines/probability/rnd_surface.py`) and surface math kernels (`oipd/core/probability_density_conversion/surface_math.py`) with expanded interface/core tests.
### Bug fixes
- Made the `t` input consistent everywhere (same domain requirements and same type requirements)

## [2.0.0] - 2025-02-06
### Added
- overhauled user API interface yet again to accomodate full volatility surface fitting pipeline. This new API should be quite thoughtfully designed and futureproof, so I do not expect further breaking changes
    - seperate classes for fitting a single 'curve' (vol smile or prob distribution on a single future date) vs fitting a 'surface' (vol surface or implied-prob over time)
- implemented a full volatility fitting pipeline, compatible for fitting a single vol smile or fitting a vol surface over multiple expiries
    - uses SVI to fit vol smiles, with calendar interpolation in total variance with basic calendar arbitrage guard

## [1.0.0] - 2025-09-18
### Added
- overhauled user API interface. Now, users interface using the RND() class, and input arguments using MarketInputs and ModelParams
    - integrated plotting functionality
    - integrated convenient functions to access results
- handles dividend yield and schedule 
- integrated yfinance support for automated options data pulling, and retrieval of dividend info
- refactored folder structure - seperated RND from options-pricing functionality
- integrated put-call parity -> finds market-implied forward price, and implied dividend yield, replaced ITM call options with OTM puts converted to synthetic calls to reduce noise
- integrated Black-76 pricing model compatible with forwards
- removed KDE and fit_kde() argument
- seperated readme to quick start and technical

## [0.0.6] - 2025-09-29
### Added
- Initial skeleton for the CHANGELOG.
- Upgraded old dependencies; now supports Python 3.13.
- Removed Streamlit dashboard code and images (#42).

## [0.0.5] – 2025-03-03
### Added
- First public release on PyPI. 
