---
title: Introduction
nav_order: 1
has_children: true
---

# Introduction: What is OIPD?

[OIPD (Options Implied Probability Distribution)](https://github.com/tyrneh/options-implied-probability) provides two related capabilities:

**1. Compute market-implied probability distributions of future asset prices.**
- While markets don't predict the future with certainty, under the efficient market view, these market expectations represent the best available estimate of what might happen.
- The probability distribution is a transformation of the volatility surface. Thus, accurately modelling a volatility surface is crucial to computing the distribution, which leads to OIPD's second feature below.

<p align="center" style="margin-top: 20px;">
  <img src="https://github.com/tyrneh/options-implied-probability/blob/main/example.png" alt="example" style="width:100%; max-width:1200px; height:auto; display:block; margin-top:5px;" />
</p>

**2. Fit arbitrage-free volatility smiles and surfaces for pricing and risk analysis.**
- Fitting a vol surface well is a complex and expensive process, with the leading software provider costing $50k USD/month/seat. OIPD open-sources the entire pipeline fairly rigorously, with further improvements in the roadmap.

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
