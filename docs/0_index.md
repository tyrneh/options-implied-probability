---
title: Introduction
nav_order: 1
has_children: true
---

# Introduction: What is OIPD?

[OIPD (Options Implied Probability Distribution)](https://github.com/tyrneh/options-implied-probability) provides two related capabilities:

**1. Compute market-implied probability distributions of future asset prices.**
- Market-implied probabilities come from option prices and reflect risk-neutral market expectations.
- In OIPD, probability estimation is built on top of fitted volatility structures for numerical stability.

<p align="center" style="margin-top: 20px;">
  <img src="https://github.com/tyrneh/options-implied-probability/blob/main/example.png" alt="example" style="width:100%; max-width:1200px; height:auto; display:block; margin-top:5px;" />
</p>

**2. Fit arbitrage-aware volatility smiles and surfaces for pricing and risk analysis.**
- OIPD exposes stateful estimators (`VolCurve`, `VolSurface`) and probability objects (`ProbCurve`, `ProbSurface`) in a workflow that is accessible for non-specialists and still usable for quant workflows.

<table align="center" cellspacing="12" style="margin-top:20px; width:100%; border-collapse:separate;">
  <tr>
    <td style="width:50%; border:5px solid #000;">
      <img src="images/vol_curve.png" alt="vol curve" style="width:100%; height:280px; object-fit:contain; display:block;" />
    </td>
    <td style="width:50%; border:5px solid #000;">
      <img src="images/vol_surface.png" alt="vol surface" style="width:100%; height:280px; object-fit:contain; display:block;" />
    </td>
  </tr>
</table>
