---
title: Compare Drift Only on Matching Grids
date: 2026-04-17
category: ce-docs/solutions/logic-errors
module: Golden master drift tooling
problem_type: logic_error
component: tooling
symptoms:
  - Drift output can report zero IV drift even when the candidate and reference strikes differ.
root_cause: logic_error
resolution_type: code_fix
severity: medium
tags: [golden-master, drift-reporting, grid-alignment, options]
---

# Compare Drift Only on Matching Grids

## Problem
The golden-master drift comparison script can produce misleading statistics if it compares values by array position before checking that both arrays use the same x-axis. For IV drift, the x-axis is strike.

## Symptoms
- The report can say `max_abs=0` and `rms=0` for IV drift even when the reference and candidate strikes are different.
- Example: IVs at strikes `100, 110` can be compared against IVs at strikes `200, 210` just because the arrays have the same length.

## What Didn't Work
- Comparing IV arrays directly by position was not enough. It answered "are these numbers equal?" but skipped the more important question: "are these numbers attached to the same strikes?"
- Distribution drift already needed price-grid handling, but the same principle also applied to IV test points.

## Solution
Validate the strike grid before printing aggregate IV drift statistics.

Before:

```python
print(_diff_stats(ref_ivs, cand_ivs))
```

After:

```python
same_strikes = ref_strikes.size == cand_strikes.size and np.allclose(
    ref_strikes[:n], cand_strikes[:n], rtol=0.0, atol=0.0
)

if same_strikes:
    print(_diff_stats(ref_ivs, cand_ivs))
else:
    print(
        "not comparable: strike grid mismatch "
        f"(reference={ref_strikes.size}, candidate={cand_strikes.size})"
    )
```

This keeps aggregate IV max/RMS drift only for same-strike comparisons. If strikes differ, the report still shows the per-row values for inspection, but marks the aggregate result as not comparable.

## Why This Works
An IV value is only economically comparable at the same strike. The fix makes the drift report compare `(strike, IV)` pairs instead of treating IV arrays as context-free numbers.

This prevents false reassurance. A "zero drift" report now means the values match at the same strikes, not merely that two arrays happened to contain the same numbers.

## Prevention
- Any drift comparison over sampled curves should check the x-axis first.
- For same-grid data, direct max/RMS is valid.
- For different-grid data, either interpolate onto a common grid and label the result clearly, or report the aggregate drift as not comparable.
- Add a small regression check using equal values on different grids:

```python
reference = {"strikes": [100, 110], "ivs": [0.20, 0.30]}
candidate = {"strikes": [200, 210], "ivs": [0.20, 0.30]}

# Expected: not comparable, not max_abs=0.
```

## Related Issues
- `tests/data/compare_golden_master.py` now validates strike grids before aggregate IV drift reporting.
- The same principle applies to distribution drift: compare PDF/CDF on matching price grids, or interpolate and label the comparison.
