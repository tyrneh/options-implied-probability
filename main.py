from oipd import RND, MarketInputs
from datetime import date

# 1 ─ point to a ticker and provide market info
crypto_market = MarketInputs(
    valuation_date=date.today(),
    expiry_date=date(2025, 12, 26),
    risk_free_rate=0.04,
)
crypto_est = RND.from_ticker("ETH", crypto_market, vendor="bybit") 
# 2 - run estimator, auto fetching data from Yahoo Finance
# est = RND.from_ticker("AAPL", market)   

# 3 ─ access results and plots
est.prob_at_or_above(4500)               # P(price >= $120)
est.prob_below(4700)                     # P(price < $100)
est.plot()                              # plot probability and cumulative distribution functions
