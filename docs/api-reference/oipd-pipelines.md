---
title: oipd.pipeline
parent: API Reference
nav_order: 2
---

## 4.2. `oipd.pipelines`
This module contains the high-level stateless pipelines that orchestrate the estimation process.
*   **`vol_estimation.fit_vol_curve_internal`**: The core function that takes cleaned data and market inputs and returns a fitted volatility curve.
*   **`prob_estimation.derive_distribution_from_curve`**: The function that takes a fitted curve and derives the probability distribution.