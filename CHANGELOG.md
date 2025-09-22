
# Changelog
All notable changes to **oipd** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **Cryptocurrency Options Support**: Added Bybit vendor integration for crypto options (BTC, ETH, etc.)
- New `crypto` installation option: `pip install oipd[crypto]`
- Bybit API integration via `pybit` library
- Crypto-specific examples and documentation
- overhauled user API interface. Now, users interface using the RND() class, and input arguments using MarketInputs and ModelParams
    - integrated plotting functionality
    - integrated convenient functions to access results
- handles dividend yield and schedule 
- integrated yfinance support for automated options data pulling, and retrieval of dividend info
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