# OIPD — Technical Reference 📚

This document complements the high-level `README.md`. Here you’ll find:

1. Complete installation matrix & extras
2. Detailed API documentation
3. Theory & algorithmic notes

---

## 1. Installation flavours

| Use-case                          | Command                     |
| --------------------------------- | --------------------------- |
| Full installation (with yfinance) | `pip install oipd`          |
| Core maths only (no data vendors) | `pip install oipd[minimal]` |

Core requires at minimum **Python 3.10+**.

---

## 2. API Overview

OIPD provides a single `RND` class that extracts market-implied probability distributions from options data using the Breeden-Litzenberger formula and Black-Scholes pricing.

**Main Entry Point: `RND` Class**

The `RND` class is your high-level facade that users interact with. It has three main ways to load options data:

1. **CSV files**: `RND.from_csv(path, market_params)`
2. **DataFrames**: `RND.from_dataframe(df, market_params)`
3. **Live data**: `RND.from_ticker("AAPL", market_params)` (auto-fetches from vendors, only YFinance currently integrated)

**Configuration Objects**

- **`MarketParams`**: Market conditions (current price, risk-free rate, expiry date, dividends)
- **`ModelParams`**: Algorithm settings (solver type, KDE smoothing, pricing engine)

**Workflow Examples**

_From live data (auto-fetches price & dividends):_

```python
# First, discover available expiry dates
expiry_dates = RND.list_expiry_dates("AAPL")
print(expiry_dates[:3])  # ['2025-01-17', '2025-01-24', '2025-02-21']

# Then use one of the available dates (None values enable auto-fetch)
market = MarketParams(
    current_price=None,
    dividend_yield=None,
    expiry_date=date(2025, 1, 17),
    risk_free_rate=0.04
)
est = RND.from_ticker("AAPL", market)
```

_From CSV file:_

```python
market = MarketParams(current_price=150, expiry_date=date(2025, 12, 19), risk_free_rate=0.04)
est = RND.from_csv("options_data.csv", market, column_mapping={"Strike": "strike", "Last": "last_price"})
```

_From DataFrame:_

```python
market = MarketParams(current_price=150, expiry_date=date(2025, 12, 19), risk_free_rate=0.04)
est = RND.from_dataframe(df, market)
```

_Access results:_

```python
prob = est.prob_at_or_above(160)  # P(price >= $160)
est.plot()  # Publication-ready plots
df = est.to_frame()  # Export as DataFrame
```

**Smart Features**

- **Auto-fetching**: `from_ticker()` automatically gets current price and dividend data
- **Flexible data sources**: Pluggable vendor system (currently yfinance, extensible)
- **Built-in plotting**: Publication-ready PDF/CDF plots with dual y-axes
- **Probability calculations**: Easy P(price >= X) queries
- **Column mapping**: Handles different CSV formats automatically

The API follows a scikit-learn-like pattern with `.fit()` and result properties, making it familiar to ML practitioners while being finance-domain specific.

### 2.1 MarketParams

Configuration object that defines the market environment and time horizon for the RND estimation.

```
MarketParams(
    current_price: float,           # Required (or auto-fetched from ticker)
    risk_free_rate: float,          # Required

    # Time horizon - provide either:
    days_forward: int,              # Option 1: days until expiry
    # OR
    current_date: date,             # Option 2: valuation date
    expiry_date: date,              # Option 2: expiration date

    # Dividends - optional:
    dividend_yield: float,          # Annual dividend yield (e.g., 0.02 for 2%)
    dividend_schedule: DataFrame,   # Discrete dividend payments
)
```

_Provide either `days_forward` or the date pair._ During `from_ticker` unknown
fields are auto-populated (spot, dividend_yield, possibly a schedule).

### 2.2 ModelParams

Configuration object that controls the mathematical algorithms and smoothing techniques used in the RND calculation.

```
ModelParams(
    solver        = "brent" | "newton",   # IV root-finder
    fit_kde       = False,                # smooth tails
    pricing_engine= "bs",                # placeholder for alt models
)
```

### 2.3 RND – primary estimator

Main class that fits risk-neutral density models from options data and provides probability distribution results.

```
RND.from_csv(path, market, model=...)
RND.from_dataframe(df, market, ...)
RND.from_ticker("AAPL", market, vendor="yfinance", ...)
```

Returns a fitted instance exposing

```
est.pdf_                      # Probability density function values
est.cdf_                      # Cumulative distribution function values
est.plot(kind="both")         # Publication-ready plots ("pdf", "cdf", or "both" (default))
est.prob_at_or_above(120)     # P(future price >= $120)
est.to_frame()                # Export as pandas DataFrame
```

---

## 3. Theory Overview

An option is a financial derivative that gives the holder the right, but not the obligation, to buy or sell an asset at a specified price (strike price) on a certain date in the future. Intuitively, the value of an option depends on the probability that it will be profitable or "in-the-money" at expiration. If the probability of ending "in-the-money" (ITM) is high, the option is more valuable. If the probability is low, the option is worth less.

As an example, imagine Apple stock (AAPL) is currently $150, and you buy a call option with a strike price of $160 (meaning you can buy Apple at $160 at expiration).

- If Apple is likely to rise to $170, the option has a high probability of being ITM → more valuable
- If Apple is unlikely to go above $160, the option has little chance of being ITM → less valuable

This illustrates how option prices contain information about the probabilities of the future price of the stock (as determined by market expectations). By knowing the prices of options, we can reverse-engineer and extract information contained about probabilities.

For a simplified worked example, see this [excellent blog post](https://reasonabledeviations.com/2020/10/01/option-implied-pdfs/).
For a complete reading of the financial theory, see [this paper](https://www.bankofengland.co.uk/-/media/boe/files/quarterly-bulletin/2000/recent-developments-in-extracting-information-from-options-markets.pdf?la=en&hash=8D29F2572E08B9F2B541C04102DE181C791DB870).

---

## 4. Algorithm Overview

The process of generating the PDFs and CDFs is as follows:

1. For an underlying asset, options data along the full range of strike prices are read from a CSV file to create a DataFrame. This gives us a table of strike prices along with the last price[^1] each option sold for
2. Using the Black-Sholes formula, we convert strike prices into implied volatilities (IV)[^2]. IV are solved using either Newton's Method or Brent's root-finding algorithm, as specified by the `solver_method` argument.
3. Using B-spline, we fit a curve-of-best-fit onto the resulting IVs over the full range of strike prices[^3]. Thus, we have extracted a continuous model from discrete IV observations - this is called the volatility smile
4. From the volatility smile, we use Black-Scholes to convert IVs back to prices. Thus, we arrive at a continuous curve of options prices along the full range of strike prices
5. From the continuous price curve, we use numerical differentiation to get the first derivative of prices. Then we numerically differentiate again to get the second derivative of prices. The second derivative of prices multiplied by a discount factor $\exp^{r*\uptau}$, results in the probability density function [^4]
6. We can fit a KDE onto the resulting PDF, which in some cases will improve edge-behavior at very high or very low prices. This is specified by the argument `fit_kernal_pdf`
7. Once we have the PDF, we can calculate the CDF

[^1]: We chose to use last price instead of calculating the mid-price given the bid-ask spread. This is because Yahoo Finance, a common source for options chain data, often lacks bid-ask data. See for example [Apple options](https://finance.yahoo.com/quote/AAPL/options/)
[^2]: We convert from price-space to IV-space, and then back to price-space as described in step 4. See this [blog post](https://reasonabledeviations.com/2020/10/10/option-implied-pdfs-2/) for a breakdown of why we do this double conversion
[^3]: See [this paper](https://edoc.hu-berlin.de/bitstream/handle/18452/14708/zeng.pdf?sequence=1&isAllowed=y) for more details. In summary, options markets contains noise. Therefore, generating a volatility smile through simple interpolation will result in a noisy smile function. Then converting back to price-space will result in a noisy price curve. And finally when we numerically twice differentiate the price curve, noise will be amplified and the resulting PDF will be meaningless. Thus, we need either a parametric or non-parametric model to try to extract the true relationship between IV and strike price from the noisy observations. The paper suggests a 3rd order B-spline as a possible model choice
[^4]: For a proof of this derivation, see this [blog post](https://reasonabledeviations.com/2020/10/10/option-implied-pdfs-2/)

---

## 5. File layout

```
oipd/
 ├─ core/          # mathematical primitives (PDF/CDF calculation)
 ├─ pricing/       # pricing engines (Black-Scholes, future Heston…)
 ├─ vendor/        # data connectors (yfinance, alpaca, …)
 ├─ io/            # generic readers (CSV, DataFrame)
 ├─ graphics/      # plotting helpers (matplotlib, publication styles)
 ├─ estimator.py   # high-level façade (RND class)
 └─ generate_pdf.py # legacy CLI interface
```

---

## 6. Roadmap

- incorporate dividends - DONE
- integrate YFinance data fetching - DONE
- American-option de-Americanisation module
- integrate other data vendors (Alpaca, Deribit) for automatic stock and crypto options data fetching
- alternative volatility models (Heston, SABR)
- full term-structure surface (`RNDTermSurface`)

Pull requests welcome! Reach out on GitHub issues to discuss design choices.
