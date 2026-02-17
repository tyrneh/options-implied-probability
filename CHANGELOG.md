
# Changelog
All notable changes to **oipd** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2025-02-17
### Added
- Reworked `ProbSurface` to use fitted `VolSurface` as the canonical source of truth. Probability is now derived from interpolated volatility and option prices at query time, rather than interpolating probabilities directly. This is important because only the fitted vol smiles can be linearly interpolated in total-variance space; probabilities can't be interpolated directly. 
- Added direct surface query methods for arbitrary maturities: `pdf(price, t)`, `cdf(price, t)`, `quantile(q, t)`, and callable alias `__call__(price, t)`.
- Extended `ProbSurface.slice(expiry)` to support interior (non-pillar) expiries and return consistent `ProbCurve` slices with interpolated metadata.
- Added a dedicated surface probability pipeline (`oipd/pipelines/probability/rnd_surface.py`) and surface math kernels (`oipd/core/probability_density_conversion/surface_math.py`) with expanded interface/core tests.


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
