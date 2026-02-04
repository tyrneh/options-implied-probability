---
title: Installation
parent: Introduction
nav_order: 1
---

## 1.3. Installation

You can install OIPD and its dependencies, including `yfinance` for data fetching, using `pip`:

```bash
pip install oipd
```

For a minimal installation without data vendor integrations (if you are providing your own data), you can use:

```bash
pip install oipd[minimal]
```