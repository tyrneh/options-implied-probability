# OIPD — Quick Start 🚀

> Turn raw option quotes into a market-implied distribution in **two minutes**.

```bash
# 1 ─ install core + yfinance connector
pip install oipd[yfinance]
```

```python
from oipd import RND, MarketParams
from datetime import date

# 2 ─ point to a ticker + expiry
market = MarketParams(
    current_price=None,               # auto-fetched spot
    expiry_date=date(2025, 12, 19),   # next option expiry you care about
    risk_free_rate=0.04,              # flat risk-free estimate
)

est = RND.from_ticker("AAPL", market)   # magic ✨

est.plot()                              # PDF + CDF in one line
print(est.prob_at_or_above(200))        # e.g. 0.075 → 7.5 %
```

That’s it—no CSVs, no maths, no extra setup.

What happened under the hood?
1. Chain downloaded from Yahoo Finance (cached for 15 min) 
2. Spot price & dividend yield filled automatically 
3. Implied volatilities solved, spline-smoothed, differentiated twice → risk-neutral density.

When you’re ready to go deeper (custom readers, model knobs, theory) open `TECHNICAL_README.md`. 