---
title: Advanced Theory Overview
parent: Theory Overview
nav_order: 2
---

# Advanced Theory Overview

This page documents the full pipeline and implementation that OIPD uses. 

## 1. Conceptual decomposition

OIPD separates the workflow into two layers:

1. **Volatility fitting** (`VolCurve`, `VolSurface`)
2. **Probability derivation** (`ProbCurve`, `ProbSurface`)

Many users need only the volatility layer for pricing and risk, while others continue to probability analysis.

## 2. End-to-end pipeline

This represents a non-exhaustive, step-by-step pipeline to fit a volatility surface, and subsequently convert it to probability distribution. We've documentated the major steps below, as well as explaining OIPD's implementation. 

<img src="images/2_numerical_second_derivative.png" alt="Numerical second derivative" style="display:block; margin:10px auto 20px auto; width:70%; max-width:900px; height:auto;" />

## 3. OIPD vs open-source and commercial volatility fitting libraries

OIPD is an end-to-end opinionated volatility surface fitting pipeline, which handles the data plumbing and cleaning, smile/surface fitting, and probability conversion all in one interface. Other open-source packages, such as Quantlib, are very strong building-block libraries which provide certain components of the pipeline, but requires a sophisticated user to understand the complete end-to-end wiring. 

OIPD is (or at least aims to be) conceptually closer to commercial libraries like Vola Dynamics than to low-level libraries like QuantLib: it offers an integrated, configurable pipeline for fitting and probability extraction.