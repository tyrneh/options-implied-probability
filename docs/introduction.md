---
title: Introduction
nav_order: 2
has_children: true
---

# 1. Introduction

## 1.1. What is OIPD?

[OIPD (Options Implied Probability Distribution)](https://github.com/tyrneh/options-implied-probability) is a powerful Python library designed for financial quants, researchers, and traders. It provides a comprehensive toolkit for transforming raw options market data into meaningful, forward-looking probability distributions of a security's future price.

The core mission of OIPD is to democratise access to sophisticated options analysis. By handling the complex mathematics of volatility modeling and risk-neutral density estimation, OIPD allows you to focus on what matters: generating insights, managing risk, and developing novel trading strategies.

## 1.2. Core Concepts

## 1.2.1. Implied Volatility (IV)
Implied volatility is the market's expectation of future price movements of a security. It is the volatility value that, when plugged into an options pricing model (like Black-Scholes), yields the option's current market price. High IV suggests the market expects significant price swings, while low IV implies a period of relative calm.

## 1.2.2. The Volatility Smile & Skew
In theory, implied volatility should be the same for all options on the same underlying with the same expiration date. In practice, this is not the case. When you plot IV against strike prices, you often see a "smile" or "skew" pattern.

*   **Smile:** IV is lowest for at-the-money (ATM) options and increases for both in-the-money (ITM) and out-of-the-money (OTM) options.
*   **Skew:** More common in equity markets, where IV for OTM puts is higher than for OTM calls. This reflects the market's perception that there is a greater risk of a large downward move (a crash) than a large upward move.

OIPD fits mathematical models to these smiles to create a continuous volatility curve. The primary model used is the **SVI (Stochastic Volatility Inspired)** model, which is known for its robust and arbitrage-free representation of the volatility smile.

## 1.2.3. Risk-Neutral Density (RND)
The risk-neutral density is a probability distribution of the future price of an asset, derived from option prices. It represents the probabilities that a risk-neutral investor would assign to different future price levels.

A key insight from financial theory is that the RND can be derived from the second derivative of the call price function with respect to the strike price. OIPD uses the fitted volatility curve to create a smooth call price function, and then calculates its second derivative to obtain the RND. This RND is a powerful tool for:

*   **Risk Management:** Quantifying the probability of extreme price moves.
*   **Trade Idea Generation:** Identifying discrepancies between your own views and the market's implied probabilities.
*   **Product Pricing:** Valuing complex derivatives.

## 1.2.4 Fitting Methods
* **b-spline:**
* **SVI:**