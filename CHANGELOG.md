
# Changelog
All notable changes to **oipd** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-10
### Added
- overhauled user API interface. Now, users interface using the RND() class
    - integrated plotting functionality
    - integrated convenient function to return the probability future price will be at or above some value
- handles dividend yield and schedule 
- integrated yfinance support for automated options data pulling, and retrieval of dividend info
- refactored folder structure - seperated RND from options-pricing functionality
    ├─ core/                    # pure maths & density logic
    │   ├─ __init__.py
    │   ├─ pdf.py               # ∂²K → RND extractor
    │   └─ pricing/             # plug-in pricing engines
    │       ├─ __init__.py      # get_pricer registry
    │       ├─ european.py      # Black-Scholes (+ q, discrete-div helper)
    │       └─ utils.py         # PV-dividend, discount factors, etc.

## [0.0.6]
### Added
- Initial skeleton for the CHANGELOG.
- Upgraded old dependencies; now supports Python 3.13.
- Removed Streamlit dashboard code and images (#42).

## [0.0.5] – 2023-11-02
### Added
- First public release on PyPI. 