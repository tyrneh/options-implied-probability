# OIPD — Technical Reference 📚

This document complements the high-level `README.md`. Here you’ll find:

1. Complete installation notes
2. Detailed API documentation
3. Theory & algorithmic notes

---

## 1. Installation flavours

| Use-case                          | Command                     |
| --------------------------------- | --------------------------- |
| Full installation (with yfinance) | `pip install oipd`          |
| Core maths only (no data vendors) | `pip install oipd[minimal]` |

Requires at minimum **Python 3.10+**.

---

## 2. API Overview

OIPD provides a single `RND` class that extracts market-implied probability distributions from options data.
 
The default implementation uses Black-76 pricing in forward space with the Breeden-Litzenberger formula. Users may switch to Black-Scholes when only calls are available and dividend assumptions are provided.

**Main Entry Point: `RND` Class**

The `RND` class is the high-level facade that users interact with. It has three ways to load options data:

1. **CSV files**: `RND.from_csv(path, market)`
2. **DataFrames**: `RND.from_dataframe(df, market)`
3. **Live data**: `RND.from_ticker("AAPL", market)` 

`RND` takes the mandatory MarketInputs parameter, and an optional ModelParams parameter. 

MarketInputs loads information on market data, while ModelParams specifies algorithm settings.

### 2.1 MarketInputs

Configuration object that defines the market environment and time horizon for the RND estimation.

```python
MarketInputs(
    risk_free_rate: float,          # Required
    valuation_date: date,           # Required 
    risk_free_rate_mode: Literal["annualized", "continuous"] = "annualized",  # How r is quoted

    # Time horizon - provide ONE of these:
    days_to_expiry: int,            # Option 1: days until expiry  
    expiry_date: date,              # Option 2: expiration date (auto calculates days difference between expiry and valuation dates)

    # Market data - optional for from_ticker() mode, as they can be auto fetched:
    underlying_price: float,        # Current price (spot S or futures F) 

    # Dividends 
    #   Note: When both calls and puts data are available, OIPD infers forward-looking dividends information via put-call parity. 
    #   The fields below are only used when put data is missing (e.g., calls-only datasets).
    dividend_yield: float,          # Option 1: Annual dividend yield (e.g., 0.02 for 2%)
    dividend_schedule: DataFrame,   # Option 2: Discrete dividend payments, requires 'ex_date' & 'amount' columns
)
```

### 2.3 ModelParams

OPTIONAL configuration object that controls the algorithms used in the RND calculation.

```python
ModelParams(
    solver             = "newton" | "brent",   # IV root-finder; defaults to brent
    pricing_engine     = "black76" | "bs",     # default forward-based Black-76. Use Black-Scholes only when puts data are unavailable
    price_method       = "last" | "mid",       # defaults to 'mid'; mid-price calculated as `(bid + ask) / 2`
    max_staleness_days = 1,                   # filter options older than N calendar days from valuation_date. Set to None to disable filtering
)
```

### 2.4 RND estimator

Main class that fits risk-neutral density models from options data and provides probability distribution results.

```python
RND.from_csv(path, market, model, column_mapping={"YourHeader": "oipd_field"})  # Maps CSV headers to OIPD fields
RND.from_dataframe(df, market, model, column_mapping={"YourHeader": "oipd_field"})  # Maps DataFrame columns to OIPD fields
RND.from_ticker("AAPL", market, vendor="yfinance", ...)  # (auto-fetches from vendors, only YFinance currently integrated)
```

#### Expected Data Format

OIPD expects the following columns in your data:

**Required columns:**
- `strike` (float): Option strike price
- `last_price` (float): Option price (last traded price or mark)

**Optional columns for enhanced functionality:**
- `option_type` (str): "C" for calls, "P" for puts (enables put-call parity preprocessing)
- `bid` (float): Bid price
- `ask` (float): Ask price  
- `last_trade_date` (datetime): When the option was last traded (for staleness filtering)

**Example data format:**
```python
# CSV/DataFrame with put-call parity support
strike  | last_price | option_type | bid  | ask  | last_trade_date
95.0    | 2.50      | C          | 2.45 | 2.55 | 2025-01-15
95.0    | 1.20      | P          | 1.15 | 1.25 | 2025-01-15
100.0   | 1.85      | C          | 1.80 | 1.90 | 2025-01-15
100.0   | 2.10      | P          | 2.05 | 2.15 | 2025-01-15
```

**Column mapping examples:**
```python
# For CSV with different column names
column_mapping = {
    "Strike": "strike",
    "Last Price": "last_price", 
    "Type": "option_type",       
    "Bid": "bid",
    "Ask": "ask"
}
```

#### Put-Call Parity Enhancement

When both calls and puts are available (indicated by `option_type` column), OIPD automatically applies put-call parity preprocessing to improve data quality:

- **Infers forward price** from at-the-money call-put pairs
- **Uses OTM options only**: calls above forward, puts below forward  
- **Replaces ITM calls** with synthetic calls derived from OTM puts
- **Completely transparent** - no changes to your code required

This preprocessing happens automatically when `option_type` column is detected and improves the accuracy of the risk-neutral density estimation.

**Note:** For vendor data sources (like yfinance), the `option_type` column is created automatically when combining calls and puts - no column mapping needed for this field.

### Note on risk-free rate conventions

- Internally, OIPD uses continuous compounding (discounting with `exp(-r * T)`).
- Control how your input rate is interpreted via `MarketInputs.risk_free_rate_mode`:
  - `"annualized"` (default): treats `risk_free_rate` as a simple annualized yield on ACT/365 for the specific horizon T; converts to continuous using `r_cont = ln(1 + y*T)/T`.
  - `"continuous"`: takes `risk_free_rate` as already continuous and uses it directly.
- This aligns with common front-end Treasury quoting (annualized nominal). If your system already produces continuous rates, set `risk_free_rate_mode="continuous"`.

Returns a fitted instance exposing

```python
est.plot(kind="both")         # Publication-ready plots ("pdf", "cdf", or "both" (default))
est.prob_at_or_above(120)     # P(future price >= $120)
est.to_frame()                # Export as pandas DataFrame
est.to_csv(path)              # Export as a CSV
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

## 4. Algorithm Details

The process of generating the PDFs and CDFs is as follows:

1. For an underlying asset, options data along the full range of strike prices are read from a source file to create a DataFrame. This gives us a table of strike prices along with the last or mid price[^1] each option sold for
2. Apply put–call parity preprocessing: estimate the forward from near‑ATM call–put pairs
3. Based on the forward price, restrict to OTM options. Keep calls above the forward and replace in‑the‑money calls with synthetic calls constructed from OTM puts via parity, to reduce noise from illiquid ITM quotes[^5].
4. Using the chosen pricing model (Black‑76 as it works with forward prices) we convert strike prices into implied volatilities (IV)[^2]. IV are solved using either Newton's Method or Brent's root‑finding algorithm, as specified by the `solver_method` argument
5. Using B‑spline, we fit a curve‑of‑best‑fit onto the resulting IVs over the full range of strike prices[^3]. Thus, we have extracted a continuous model from discrete IV observations — this is called the volatility smile
6. From the volatility smile, we use the same pricing model to convert IVs back to prices. Thus, we arrive at a continuous curve of options prices along the full range of strike prices
7. From the continuous price curve, we use numerical differentiation to get the first derivative of prices. Then we numerically differentiate again to get the second derivative of prices. The second derivative of prices multiplied by a discount factor $\exp^{r*\uptau}$, results in the probability density function [^4]
8. Once we have the PDF, we can calculate the CDF

[^1]: Mid-price given the bid-ask spread is usually less noisy, as last price can reflect stale trades and do not reflect real-time information 
[^2]: We convert from price-space to IV-space, and then back to price-space as described in step 4. See this [blog post](https://reasonabledeviations.com/2020/10/10/option-implied-pdfs-2/) for a breakdown of why we do this double conversion
[^3]: See [this paper](https://edoc.hu-berlin.de/bitstream/handle/18452/14708/zeng.pdf?sequence=1&isAllowed=y) for more details. In summary, options markets contains noise. Therefore, generating a volatility smile through simple interpolation will result in a noisy smile function. Then converting back to price-space will result in a noisy price curve. And finally when we numerically twice differentiate the price curve, noise will be amplified and the resulting PDF will be meaningless. Thus, we need either a parametric or non-parametric model to try to extract the true relationship between IV and strike price from the noisy observations. The paper suggests a 3rd order B-spline as a possible model choice
[^4]: For a proof of this derivation, see this [blog post](https://reasonabledeviations.com/2020/10/10/option-implied-pdfs-2/)
[^5]: Parity-based OTM‑only preprocessing follows Aït‑Sahalia and Lo, ["Nonparametric Estimation of State‑Price Densities Implicit in Financial Asset Prices"](https://www.princeton.edu/~yacine/aslo.pdf).

---

## 5. Auto-fetching Architecture

OIPD uses an immutable data flow for vendor integration that prevents common mistakes and ensures data integrity.

### Data Flow

```
MarketInputs (user) + VendorSnapshot (fetched) → ResolvedMarket (merged)
     ↓                      ↓                           ↓
  frozen=True          frozen=True                 frozen=True
  (never modified)     (vendor data)              (final values)
```

### Key Principles

1. **MarketInputs is immutable** - It's a `@dataclass(frozen=True)` that is never modified
2. **Auto-fetched data lives in the result** - Access via `est.market.underlying_price`
3. **Provenance tracking** - The result knows where each value came from (user vs vendor)
4. **Resolution modes** - Different data sources use different resolution strategies

### Example: Accessing Auto-fetched Data

```python
# User provides minimal inputs
market = MarketInputs(
    valuation_date=date.today(),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04,
)

# Fetch and estimate
est = RND.from_ticker("SPY", market)

# Access fetched values through result
print(f"Current price: ${est.market.underlying_price:.2f}")
print(f"Source: {est.market.provenance.price}")  # "vendor"
print(est.summary())  # One-line summary of all sources
```

---

## 6. Roadmap

Convenience features:
- integrate other data vendors (Alpaca, Deribit) for automatic stock and crypto options data fetching


- American-option de-Americanisation module
- alternative volatility models (Heston, SABR)
- full term-structure surface (`RNDTermSurface`)

Pull requests welcome! Reach out on GitHub issues to discuss design choices.
