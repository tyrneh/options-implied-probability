![OIPD logo](https://github.com/tyrneh/options-implied-probability/blob/main/.meta/images/OIPD%20Logo.png)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oipd?logo=python&logoColor=white)](https://pypi.org/project/oipd/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tyrneh/options-implied-probability/blob/main/examples/OIPD_colab_demo.ipynb)
[![Chat on Discord](https://img.shields.io/badge/chat-on%20Discord-brightgreen?logo=discord&logoColor=white)](https://discord.gg/pWVrmQWk)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/oipd?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/oipd)

# Overview

OIPD computes the market's expectations about the probable future prices of an asset, based on information contained in options data. 

While markets don't predict the future with certainty, under the efficient market hypothesis, these collective expectations represent the best available estimate of what might happen.

Traditionally, extracting these “risk-neutral densities” required institutional knowledge and resources, limited to specialist quant-desks. OIPD makes this capability accessible to everyone — delivering an institutional-grade tool in a simple, production-ready Python package.

<p align="center" style="margin-top: 80px;">
  <img src="https://github.com/tyrneh/options-implied-probability/blob/main/example.png" alt="example" style="width:100%; max-width:1200px; height:auto; display:block; margin-top:50px;" />
</p>



# Quick start

#### Installation
```bash
pip install oipd
```

#### Usage

![OIPDwalkthrough](https://github.com/user-attachments/assets/2da5506d-a720-4f93-820b-23b368d074bb)

```python
from oipd import RND, MarketInputs
from datetime import date

# 1 ─ point to a ticker and provide market info
market = MarketInputs(
    valuation_date=date.today(),      # the "as-of" date for the analysis
    expiry_date=date(2025, 12, 19),   # option expiry date you care about
    risk_free_rate=0.04,              # annualized risk-free rate
)

# 2 - run estimator, auto fetching data from Yahoo Finance
est = RND.from_ticker("AAPL", market)   

# 3 ─ access results and plots
est.prob_at_or_above(120)               # P(price >= $120)
est.prob_below(100)                     # P(price < $100)
est.plot()                              # plot probability and cumulative distribution functions 
```

OIPD also supports manual CSV or DataFrame uploads. See [`TECHNICAL_README.md`](TECHNICAL_README.md) for more details.

See [more examples](examples/example.ipynb) with provided options data. 


# Use cases

**Event-driven strategies: assess market's belief about the likelihood of mergers**

- Nippon Steel offered to acquire US Steel for $55 per share; in early 2025, US Steel was trading at $30 per share. Using OIPD, you find that the market believed US Steel had a ~20% probability of acquisition (price >= $55 by end of year)
- If you believe that political backlash was overstated and the acquisition was likely to be approved, then you can quantify a trade's expected payoff. Compare your subjective belief with the market-priced probability to determine expected value of buying stock or calls

**Risk management: compute forward-looking Value-at-Risk**

- A 99% 12-month VaR of 3% is (i) backward-looking and (ii) assumes a parametric distribution, often unrealistic assumptions especially before catalysts
- Ahead of earnings season, pull option-implied distributions for holdings. The forward-looking, non-parametric distribution point to a 6% portfolio-blended VaR

**Treasury management: decide the next commodity hedge tranche**

- As an airline, a portion of next year’s jet fuel demand is hedged; the rest floats. Use OIPD to estimate the probability of breaching your budget and the expected overspend (earnings-at-risk) on the unhedged slice
- If OIPD shows higher price risk, add a small 5–10% hedged tranche using to pull P(breach)/EaR back within board guardrails

# Community

Pull requests welcome! Reach out on GitHub issues to discuss design choices.

Join the [Discord community](https://discord.gg/pWVrmQWk) to share ideas, discuss strategies, and get support. Message me with your feature requests, and let me know how you use this. 
