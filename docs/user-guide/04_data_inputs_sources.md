---
title: Data Inputs and Sources
parent: User Guide
nav_order: 4
---

# Data Inputs and Sources

OIPD keeps fitting strict and explicit:

- `fit(...)` always takes a DataFrame and `MarketInputs`.
- Vendor helpers live under `oipd.sources`.

## Required Chain Fields

At minimum, provide fields equivalent to:

- `expiry`
- `strike`
- `option_type`
- pricing columns used by your configuration (commonly `last_price`, optionally `bid`/`ask`)

You can remap custom columns with `column_mapping`.

## Local Input Helpers (Offline)

```python
import pandas as pd
from oipd import sources

mapping = {
    "K": "strike",
    "P": "last_price",
    "B": "bid",
    "A": "ask",
    "Type": "option_type",
    "Exp": "expiry",
}

raw_df = pd.DataFrame(
    {
        "K": [90, 95, 100, 105, 110, 90, 95, 100, 105, 110],
        "P": [12.5, 8.2, 5.1, 3.1, 1.6, 2.1, 3.0, 4.4, 6.5, 9.8],
        "B": [12.3, 8.0, 4.9, 2.9, 1.4, 1.9, 2.8, 4.2, 6.3, 9.6],
        "A": [12.7, 8.4, 5.3, 3.3, 1.8, 2.3, 3.2, 4.6, 6.7, 10.0],
        "Type": ["call", "call", "call", "call", "call", "put", "put", "put", "put", "put"],
        "Exp": ["2025-03-21"] * 10,
    }
)

chain_from_csv = sources.from_csv("/tmp/oipd_docs_chain.csv", column_mapping=mapping)
chain_from_df = sources.from_dataframe(raw_df, column_mapping=mapping)
```

## Vendor Helpers (Live)

```python
from oipd import sources

expiries = sources.list_expiry_dates("AAPL")
chain, snapshot = sources.fetch_chain("AAPL", expiries=expiries[0])
```

```python
from oipd import MarketInputs

market = MarketInputs(
    valuation_date=snapshot.date,
    risk_free_rate=0.04,
    underlying_price=snapshot.underlying_price,
)
```

`fetch_chain` enforces mutual exclusivity:

- provide `expiries=...` **or** `horizon=...`, but not both.

## Stage-2 Live Validation Checklist

Run this when network access is available:

1. `sources.list_expiry_dates("AAPL")` returns non-empty list.
2. `sources.fetch_chain("AAPL", expiries=...)` returns `(DataFrame, VendorSnapshot)`.
3. `MarketInputs` built from `snapshot` fits successfully via `VolCurve.fit(...)`.

## Validated Commands (Offline)

- [x] `sources.from_csv(...)` snippet executed in `.venv`.
- [x] `sources.from_dataframe(...)` snippet executed in `.venv`.
- [x] Live vendor snippets intentionally deferred to Stage-2 due network limits.
