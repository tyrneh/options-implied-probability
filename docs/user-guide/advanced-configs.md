---
title: Advanced Configuration
parent: User Guide
nav_order: 5
---

# 3.5 Advanced Configuration

## SVI Method Options
You can pass a `SVICalibrationOptions` object or a dictionary to `method_options` to control the SVI calibration:

```python
from oipd.core.vol_surface_fitting.shared.svi_types import SVICalibrationOptions

svi_options = SVICalibrationOptions(
    random_seed=42,
    max_iter=1000,
    tolerance=1e-6
)

vol_curve = oipd.VolCurve(method_options=svi_options).fit(options_df, market)
```

## Handling Dividends
When using the `'bs'` pricing engine, you must provide dividend information. You can provide either a continuous `dividend_yield` or a `dividend_schedule` (a list of (date, amount) tuples).