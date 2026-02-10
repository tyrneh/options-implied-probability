---
title: Contributing
nav_order: 7
---

# 6. Contributing

## 6.1. How to Contribute

Contributions are welcome! If you have a bug fix, feature proposal, or improvement, please open an issue on GitHub to discuss it first. Then, you can submit a pull request.

## 6.2. Development Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/tyrneh/options-implied-probability.git
    cd options-implied-probability
    ```
2.  Install the library in editable mode with development dependencies:
    ```bash
    pip install -e .[dev]
    ```

## 6.3. Code Style and Conventions

We use `black` for code formatting and `isort` for organising imports. Before submitting a pull request, please run these tools:

```bash
black .
isort .
```

We also use `mypy` for static type checking. Please ensure your code passes the type checks.