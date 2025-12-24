from oipd import RND, MarketInputs, ModelParams , RNDResultSet
import matplotlib.pyplot as plt
from datetime import date


ticker_ = "AAPL"

# 3 INPUTS:
# 1. market parameters
# 2. model parameters
# 3. column mapping

market = MarketInputs(
    valuation_date=date.today(),
    expiry_date=date(2026, 3, 20),
    risk_free_rate=0.04199, # US 3-month nominal Treasury yield
)

# estimate RND Set
est_appl = RNDResultSet.set_from_ticker(ticker_,market)



est_appl.plot()

plt.show()