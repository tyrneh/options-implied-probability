![OIPD logo](.meta/images/OIPD%20Logo.png)

![Python version](https://img.shields.io/badge/python-3.10-blue.svg)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

<strong>OIPD generates a probability distribution–a single curve that describes the probabilities of future prices for a security–from options data. 

These probabilities reflect the consensus, risk-neutral expectations of all market participants. While markets don't predict the future with certainty, under the efficient market hypothesis, these collective expectations represent the best available estimate of what might happen.</strong>

# Quick start

```bash
# 1 ─ install core + yfinance connector
pip install oipd
```

```python
from oipd import RND, MarketParams
from datetime import date

# 2 ─ point to a ticker + expiry
market = MarketParams(
    current_price=None,               # auto-fetched spot price
    expiry_date=date(2025, 12, 19),   # next option expiry you care about
    risk_free_rate=0.04,              # risk-free rate
)

est = RND.from_ticker("AAPL", market)   # run estimator

# 3 ─ integrated plotting
est.plot()                              # PDF + CDF in one line
print(est.prob_at_or_above(200))        # e.g. 0.075 → 7.5 %
```

OIPD also supports manual CSV options data uploads. See `TECHNICAL_README.md` for more details. 

# Use cases

**Retail traders: Should I buy GME call options for $2 at $75 strike?**
- You discover the market assigns only a 6 % chance that GME finishes the week above $75
- A $2 premium on a 6% event means the expected gain is 0.06 × ($75 – $73 break-even) ≈ $0.12, while the expected loss (94 % of the time) is the full $2. The trade’s expected value is negative

**Professional managers: What is my portfolio's tail risk ahead of major events?**
- Historical view: A 99% one-month VaR of 3% is backward-looking and assumes a parametric distribution–often unrealistic before catalysts
- Market-implied view: Ahead of earnings and a central-bank meeting, pull option-implied distributions for holdings. The forward-looking, non-parametric distribution point to a 6% VaR–twice the historical figure–signalling the need to hedge, trim, or tighten limits

**Corporates: What proportion of my commodity exposure should I hedge next quarter?**
- Typically, my finance team hedges a fixed 70% of my jet fuel exposure, while leaving 30% unhedged to benefit if oil price falls
- I find OIPD shows a fatter upper tail–a higher probability of breaching the budget price. This points to lifting hedging coverage to 80% to manage price risk

# License

DISCLAIMER: This software is provided for informational and educational purposes only and does not constitute financial advice. Use it at your own risk. The author is not responsible for any financial losses or damages resulting from its use. Always consult with a qualified financial professional before making any financial decisions.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
