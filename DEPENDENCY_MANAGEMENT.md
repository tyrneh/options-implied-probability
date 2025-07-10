# Dependency Management Guide for OIPD

## Overview
This document explains the dependency management structure for the OIPD (Options Implied Probability Distribution) package.

## Dependency Structure

### Core Dependencies
The core dependencies are the minimal set of packages required for the basic functionality of OIPD:
- **numpy**: Numerical computations
- **pandas**: Data manipulation and DataFrame operations
- **scipy**: Scientific computing (interpolation, optimization, statistics)
- **matplotlib**: Basic plotting functionality
- **matplotlib-label-lines**: Enhanced line labeling for plots
- **traitlets**: Type checking for configuration

### Optional Dependencies
#### Development (`[dev]`)
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **mypy**: Static type checking
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting

## Installation Options

### Basic Installation (Core Only)
```bash
pip install oipd
# or from source:
pip install -r requirements.txt
```

### Development Installation
```bash
pip install oipd[dev]
# or from source:
pip install -r requirements.txt -r requirements-dev.txt
# or for everything:
pip install -e ".[all]"
```

## Version Management

We use flexible version ranges to ensure compatibility while allowing for updates:
- **Major version cap**: Prevents breaking changes (e.g., `numpy>=1.23.0,<2.0.0`)
- **Minor version flexibility**: Allows bug fixes and minor features
- **Security updates**: Automatically included within version ranges

## Dependency Tree

```
oipd
├── numpy >=1.23.0,<2.0.0
├── pandas >=1.5.0,<3.0.0
│   ├── numpy (shared)
│   ├── python-dateutil
│   └── pytz
├── scipy >=1.9.0,<2.0.0
│   └── numpy (shared)
├── matplotlib >=3.6.0,<4.0.0
│   ├── numpy (shared)
│   ├── contourpy
│   ├── cycler
│   ├── fonttools
│   ├── kiwisolver
│   ├── pillow
│   ├── pyparsing
│   └── python-dateutil (shared)
├── matplotlib-label-lines >=0.5.0,<1.0.0
│   └── matplotlib (shared)
└── traitlets >=5.0.0,<6.0.0
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
