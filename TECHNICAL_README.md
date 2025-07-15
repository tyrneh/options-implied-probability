# OIPD — Technical Reference 📚

This document complements the high-level `README.md` and bite-sized
`QUICK_START.md`.  Here you’ll find:

1. Complete installation matrix & extras
2. Detailed API documentation
3. Theory & algorithmic notes

---

## 1. Installation flavours

| Use-case                        | Command                              |
|--------------------------------|---------------------------------------|
| Core maths only                | `pip install oipd`                    |
| Yahoo Finance connector        | `pip install oipd[yfinance]`          |
| All current connectors         | `pip install oipd[all]`               |
| Dev + test tooling             | `pip install oipd[dev]`               |

Core requires **Python 3.10+**.

---

## 2. API Overview

### MarketParams
```
MarketParams(
    current_price: float | None,
    risk_free_rate: float,
    # horizon (one of)
    days_forward: int | None = None,
    current_date : date | None = None,
    expiry_date  : date | None = None,
    # dividends (one of)
    dividend_yield    : float | None = None,
    dividend_schedule : pd.DataFrame | None = None,
)
```
*Provide either `days_forward` or the date pair.*  During `from_ticker` unknown
fields are auto-populated (spot, dividend_yield, possibly a schedule).

### ModelParams
```
ModelParams(
    solver        = "brent" | "newton",   # IV root-finder
    fit_kde       = False,                # smooth tails
    pricing_engine= "bs",                # placeholder for alt models
)
```

### RND – primary estimator
```
RND.from_csv(path, market, model=...)
RND.from_dataframe(df, market, ...)
RND.from_ticker("AAPL", market, vendor="yfinance", ...)
```
Returns a fitted instance exposing
```
est.pdf_              # numpy array
est.cdf_              # numpy array
est.plot(kind="both")
est.prob_at_or_above(120)
est.to_frame()
```

### Vendor plug-in system
```python
from oipd.vendor import register
register("alpaca", "oipd.vendor.alpaca.reader")
```
A **Reader** must subclass `oipd.io.reader.AbstractReader` and expose `Reader`
in its module. It should attach (optional) `current_price`, `dividend_yield`,
`dividend_schedule` into `DataFrame.attrs`.

---

## 3. Mathematics in brief

1. **Implied volatility inversion**  
   Solve σ for each strike via Black-Scholes (Brent/Newton).  Vectorised for speed.
2. **Smile smoothing**  
   Cubic B-spline reduces quote noise → stable derivatives.
3. **Re-pricing & differentiation**  
   Continuous smile → call price curve → first & second strike derivatives.
4. **Risk-neutral density**  
   Breeden-Litzenberger:  \( f(K) = e^{rT}\;\partial^2 C/\partial K^2 \).
5. **KDE (optional)**  
   Empirical kernel smooth; improves tail behaviour when strike grid is sparse.

For a full derivation see the links in the main README (footnotes 2–4).

---

## 4. Customisation cheat-sheet

| Want to…                              | Code snippet                                       |
|---------------------------------------|----------------------------------------------------|
| Use manual spot & yield               | `market = MarketParams(current_price=150, dividend_yield=0.02, …)` |
| Provide a discrete dividend schedule  | `market.dividend_schedule = pd.DataFrame({...})`   |
| Change IV solver                      | `model = ModelParams(solver="newton")`            |
| Disable KDE                           | `model = ModelParams(fit_kde=False)`               |
| Fetch via alternative vendor          | `est = RND.from_ticker("BTC", market, vendor="deribit")` |

---

## 5. File layout
```
oipd/
 ├─ core/          # mathematical primitives
 ├─ pricing/       # pricing engines (Black-Scholes, future Heston…)
 ├─ vendor/        # data connectors (yfinance, alpaca, …)
 ├─ io/            # generic readers (CSV, DataFrame)
 ├─ graphics/      # plotting helpers
 └─ estimator.py   # high-level façade (RND)
```

---

## 6. Roadmap

* dividend schedule auto-fetch for premium vendors  
* alternative volatility models (Heston, SABR)      
* American-option de-Americanisation module         
* full term-structure surface (`RNDTermSurface`)    

Pull requests welcome!  Reach out on GitHub issues to discuss design choices. 