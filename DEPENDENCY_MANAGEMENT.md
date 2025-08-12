# Dependency Management Guide for OIPD

## Overview
This document explains the dependency management structure for the OIPD (Options Implied Probability Distribution) package.

## Dependency Structure

### Default Dependencies
The default installation includes all core functionality plus data vendor integrations:
- **numpy**: Numerical computations
- **pandas**: Data manipulation and DataFrame operations
- **scipy**: Scientific computing (interpolation, optimization, statistics)
- **matplotlib**: Basic plotting functionality
- **matplotlib-label-lines**: Enhanced line labeling for plots
- **traitlets**: Type checking for configuration
- **yfinance**: Yahoo Finance data fetching

### Optional Dependencies
#### Minimal (`[minimal]`)
For companies that require internal-only code without external data vendors:
- **numpy**: Numerical computations
- **pandas**: Data manipulation and DataFrame operations
- **scipy**: Scientific computing (interpolation, optimization, statistics)
- **matplotlib**: Basic plotting functionality
- **matplotlib-label-lines**: Enhanced line labeling for plots
- **traitlets**: Type checking for configuration



## Installation Options

### Default Installation (Includes Data Vendors)
```bash
pip install oipd
# or from source:
pip install -r requirements.txt
```

### Minimal Installation (No External Data Vendors)
For companies that require internal-only code:
```bash
pip install oipd[minimal]
```



## Version Management

We use minimum version requirements to ensure compatibility 

## Dependency Tree

### Default Installation
```
oipd
├── numpy >=1.26.0
├── pandas >=2.1.0
│   ├── numpy (shared)
│   ├── python-dateutil
│   └── pytz
├── scipy >=1.11.0
│   └── numpy (shared)
├── matplotlib >=3.8.0
│   ├── numpy (shared)
│   ├── contourpy
│   ├── cycler
│   ├── fonttools
│   ├── kiwisolver
│   ├── pillow
│   ├── pyparsing
│   └── python-dateutil (shared)
├── matplotlib-label-lines >=0.6.0
│   └── matplotlib (shared)
├── traitlets >=5.12.0
└── yfinance >=1.0.0
    ├── numpy (shared)
    ├── pandas (shared)
    ├── requests
    ├── multitasking
    └── appdirs
```

### Minimal Installation
```
oipd[minimal]
├── numpy >=1.26.0
├── pandas >=2.1.0
├── scipy >=1.11.0
├── matplotlib >=3.8.0
├── matplotlib-label-lines >=0.6.0
└── traitlets >=5.12.0
```

## Updating Dependencies

### For maintainers:
```bash
# Update all dependencies to latest compatible versions
pip install --upgrade -r requirements.txt

# Check for outdated packages
pip list --outdated

# Generate exact versions for reproducible builds
pip freeze > requirements-lock.txt
```

### For users:
```bash
# Update to latest compatible versions
pip install --upgrade oipd
```
