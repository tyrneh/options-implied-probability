---
title: Working with Probability Distributions
parent: User Guide
nav_order: 3
---

## 3.3. Working with Probability Distributions

The `Distribution` object provides several ways to query the probability of future price movements:

*   `prob_below(price)`: P(S < price)
*   `prob_above(price)`: P(S >= price)
*   `prob_between(low, high)`: P(low <= S <= high)
*   `expected_value()`: The mean of the distribution.
*   `variance()`: The variance of the distribution.
