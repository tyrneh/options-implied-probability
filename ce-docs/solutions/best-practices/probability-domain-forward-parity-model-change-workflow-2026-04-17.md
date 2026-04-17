---
title: Probability Domain and Forward Parity Model Change Workflow
date: 2026-04-17
category: ce-docs/solutions/best-practices
module: Probability, parity, and experiment pipelines
problem_type: best_practice
component: development_workflow
severity: high
applies_when:
  - Changing probability-domain construction, implied-forward inference, SVI weighting, or experiment comparability rules.
  - Regenerating golden masters after intentional numerical model changes.
  - Debugging event-study plots whose distribution, spot, volume, or skew panels must be comparable minute to minute.
tags: [probability-domain, put-call-parity, golden-master, svi, experiments, options]
---

# Probability Domain and Forward Parity Model Change Workflow

## Context
This conversation solved a cluster of related modeling issues in OIPD: probability plots were sometimes visually cut off, experiment animations were not always comparable across scenarios, SVI fitting had ambiguous bid/ask-versus-volume weighting behavior, and put-call parity forward inference was too sensitive to stale/thin far-from-ATM pairs.

The common theme is that these are not just plotting or test problems. They are model-semantics problems: the library must be clear about the price domain being materialized, the forward used by Black-76, the quote source used for calibration, and the exact data slice used for every plotted panel.

## Guidance
Treat probability-domain, forward-inference, and experiment-pipeline changes as one end-to-end numerical contract.

1. Keep the fitted probability domain independent from the plot viewport. The native probability grid should cover the economically relevant support; viewing/export helpers such as `density_results(domain=..., points=...)` can resample or zoom afterward, but should not define the upstream support.

2. Resolve the PDF domain before differentiation. The pipeline should choose a broad domain, evaluate fitted option prices from the vol curve, compute PDF/CDF, then widen if the right tail still contributes meaningful probability mass. The domain should not be capped merely because the observed option strikes stop.

3. Use direct CDF construction and diagnostics as the canonical probability check. The returned distribution should expose whether the CDF is monotone, bounded, and minimally cleaned, so drift or tail issues are visible instead of hidden inside plotting.

4. Infer Black-76 forward from coherent, liquid same-strike parity pairs. Validate all call/put candidates, select only the nearest-ATM valid subset by `abs(strike - spot)`, then run median/MAD filtering on that selected subset only. Far valid pairs should be diagnosed as `valid_not_selected`, not used as forward evidence.

5. Do not mix quote sources inside one parity pair. Use both-leg bid/ask mids only when both legs have coherent positive bid/ask and acceptable relative spread. Fall back to both-leg `last_price` only when mids are unavailable or rejected; require lowercase `volume >= 1` when lowercase volume exists, and ignore legacy uppercase `Volume`.

6. Prefer bid/ask spread quality over volume for SVI weighting. If reliable bid/ask spread coverage exists, it is the current quote-uncertainty signal and volume should not be an additional multiplier. Use volume only as a fallback liquidity proxy when bid/ask spread coverage is missing or unreliable.

7. Make experiment plots comparable by config, not convention. Every event/control scenario in a comparison group should share model parameters, risk-free rate policy, pricing engine, price method, panel list, axis ranges, and timestamp-aligned raw spot/options data. Fallbacks should be fail-fast unless a fallback is explicitly part of the experiment.

8. Update golden masters only after reviewing drift. First run a non-mutating comparison against a temporary candidate, report SVI/IV/PDF/CDF differences, confirm the direction is expected, then regenerate committed golden files only after explicit approval.

## Why This Matters
For options-implied distributions, small software ambiguities can look like market insights. A truncated PDF can be caused by an upstream domain policy, not market pricing. A mean-minus-spot panel can move because the parity-implied forward moved, not because the plot is forcing equality. A golden-master failure can be a valid model improvement or a hidden regression; the difference depends on whether the numerical contract is explicit.

Economically, each piece answers a different question:

- The PDF domain answers: "Where could probability mass plausibly live under the fitted model?"
- The implied forward answers: "What expiry price level is implied by put-call parity?"
- Bid/ask spreads answer: "How precise is the quoted price right now?"
- Volume answers: "How much did this contract trade?"
- Golden masters answer: "Did the model's numerical outputs change, and was that change intentional?"

Keeping these concepts separate prevents accidental double-counting, stale-data artifacts, and plots that compare unlike things.

## When to Apply
- Use this workflow before changing `oipd/pipelines/probability/`, `oipd/core/probability_density_conversion/`, `oipd/core/data_processing/parity.py`, SVI weighting, or event-study experiment infrastructure.
- Use it when a visual artifact appears in a generated GIF: first trace raw data -> parity/price selection -> IV survival -> vol fit -> probability materialization -> plotting.
- Use it when regenerated plots are meant to be comparable across events, controls, assets, or timestamps.
- Use it when golden-master regression drift appears after a math/modeling change.

## Examples
PDF domain change:

```python
# Good separation:
prob = vol_curve.implied_distribution(grid_points=None)
full = prob.density_results(full_domain=True)
view = prob.density_results(points=200)
zoom = prob.density_results(domain=(450, 550), points=200)

# The zoom changes the view, not the native fitted support.
```

Parity forward pair selection:

```text
collect same-strike call/put candidates
validate coherent quote source and liquidity
rank valid pairs by abs(strike - spot)
select up to max_forward_pairs nearest-ATM pairs
run median/MAD only on selected pairs
report valid_not_selected far pairs separately
```

SVI weighting policy:

```text
if reliable bid/ask spread coverage exists:
    use spread-based measurement weights
    do not multiply by volume
elif usable volume exists:
    use volume as fallback liquidity weighting
else:
    use the base calibration weights
```

Golden-master workflow:

```bash
python tests/data/compare_golden_master.py
pytest tests/regression/test_golden_master.py -q

# Only after explicit approval:
PYTHONPATH="$PYTHONPATH:." python tests/data/generate_golden_master.py
pytest tests/regression/test_golden_master.py -q
```

## Related
- [Compare Drift Only on Matching Grids](../logic-errors/compare-drift-on-matching-grids-2026-04-17.md)
- `CHANGELOG.md` v2.0.4 documents the shipped user-facing changes from this workflow.
- `tests/data/compare_golden_master.py` is the non-mutating drift-review tool.
- `tests/core/data_processing/test_parity.py` captures the nearest-ATM parity-pair and quote-source rules.
- `tests/core/numerical/test_probability_grid.py` and probability interface tests capture the native-domain and direct-CDF contract.

## Session History Notes
- Session history confirmed the original PDF cutoff investigation started from SOL event frames where plotting was not the root cause; calibration attrition and domain policy were separate issues. (session history)
- Session history confirmed a prior automation briefly attempted to make volume weighting opt-in after golden-master drift, but the later decision was more precise: bid/ask spread should dominate when reliable, with volume as fallback. (session history)
- Session history confirmed a stale drift-reporting issue where scripts could compare the wrong code or wrong grid; the prevention is to force repo-root imports and compare values only on matching axes. (session history)
