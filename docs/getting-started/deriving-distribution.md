---
title: Deriving a Probability Distribution
parent: Getting Started
nav_order: 3
---

## 2.3. Deriving a Probability Distribution

The `Distribution` object gives you access to the PDF (Probability Density Function) and CDF (Cumulative Distribution Function), which you can use for plotting.

```python
import matplotlib.pyplot as plt

# Get the PDF and CDF
prices = dist.prices
pdf = dist.pdf
cdf = dist.cdf

# Plot the PDF and CDF
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(prices, pdf, 'b-', label='PDF')
ax1.set_xlabel('Price')
ax1.set_ylabel('Probability Density', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(prices, cdf, 'r-', label='CDF')
ax2.set_ylabel('Cumulative Probability', color='r')
ax2.tick_params('y', colors='r')

plt.title('Risk-Neutral PDF and CDF for AAPL')
fig.tight_layout()
plt.show()
```