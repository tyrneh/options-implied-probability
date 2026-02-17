---
title: Contributing
nav_order: 6
---

# Contributing

Contributions are welcome. Please open an issue first for non-trivial changes so design decisions are visible before implementation.

## Development Setup

```bash
git clone https://github.com/tyrneh/options-implied-probability.git
cd options-implied-probability
pip install -e '.[dev]'
```

## Testing Hierarchy (Project Policy)

1. `tests/interface/` (public contract): user-facing API behavior.
2. `tests/core/` (math units): numerical correctness and edge cases.
3. `tests/regression/` (golden master): numerical drift guard.

When changing math behavior, add targeted `tests/core/` coverage and then evaluate regression impact.

## Golden Master Workflow

Do **not** edit `tests/data/golden_master.json` manually.

If a mathematically justified change updates expected outputs:

```bash
export PYTHONPATH=$PYTHONPATH:.
python tests/data/generate_golden_master.py
```

Then run regression tests and review the new values critically.

## Code Quality Checklist

```bash
black .
isort .
mypy oipd
pytest -q
```



## 2. Interface vs pipeline architecture

OIPD uses a strict split:

- **Stateful interface objects** (`oipd/interface`): user-facing classes with fitted state, properties, and plotting methods.
- **Stateless pipelines** (`oipd/pipelines`): pure computational logic used by interfaces.
- **Core numerical modules** (`oipd/core`): fitting algorithms, interpolation methods, and finite-difference routines.

This design improves testability and keeps numerical logic independent from API ergonomics.
