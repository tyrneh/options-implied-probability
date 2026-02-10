---
title: Introduction
nav_order: 2
has_children: true
---

# 2. Introduction: What is OIPD?

[OIPD (Options Implied Probability Distribution)](https://github.com/tyrneh/options-implied-probability) provides 2 capabilities:

**1. It computes the market's expectations about the probable future prices of an asset, based on information contained in options data.**
   - While markets don't predict the future with certainty, under the efficient market view, these market expectations represent the best available estimate of what might happen.
   - Traditionally, extracting these market-implied distributions were limited to quants or academics. OIPD makes this capability accessible to everyone.
   - The probability distribution is a transformation of the volatility surface. Thus, accurately modelling a volatility surface is crucial to computing the distribution, which leads to OIPD's second feature below.

<p align="center" style="margin-top: 80px;">
  <img src="https://github.com/tyrneh/options-implied-probability/blob/main/example.png" alt="example" style="width:100%; max-width:1200px; height:auto; display:block; margin-top:50px;" />
</p>

**2. For options traders, it also offers a simple-to-use but rigorous pipeline to fit an arbitrage-free volatility  surface, which can be used to price options.**
   - Fitting a vol smile and surface well is a complex and expensive process, with the leading software provider costing $50k USD/month/seat. OIPD open-sources the entire pipeline fairly rigorously, with further improvements in the roadmap.

<table align="center" cellspacing="12" style="margin-top:120px; width:100%; border-collapse:separate;">
  <tr>
    <td style="width:50%; border:5px solid #000;">
      <img src=".meta/images/vol_curve.png" alt="vol curve" style="width:100%; height:280px; object-fit:contain; display:block;" />
    </td>
    <td style="width:50%; border:5px solid #000;">
      <img src=".meta/images/vol_surface.png" alt="vol surface" style="width:100%; height:280px; object-fit:contain; display:block;" />
    </td>
  </tr>
</table>
