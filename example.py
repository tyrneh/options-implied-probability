from oipd import RND, MarketInputs, ModelParams
import matplotlib.pyplot as plt
from datetime import date

# 1️⃣  what the library expects internally
#     strike , last_price , bid , ask
column_mapping_gamestop = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
    "OptionType": "option_type",
}

# 2️⃣  market parameters
market_gamestop = MarketInputs(
    spot_price=23.235,  # current price of the underlying asset
    valuation_date=date(2025, 9, 8),
    expiry_date=date(2026, 1, 16),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_gamestop = ModelParams(fit_kde=False, price_method="mid")

# 4️⃣  run using Gamestop options chain
est_gamestop = RND.from_csv(
    "data/GME_date250908exp260116_current23235.csv",
    market_gamestop,
    model=model_gamestop,
    column_mapping=column_mapping_gamestop,
)

# PDF only
fig = est_gamestop.plot(
    kind="pdf", title="Implied Gamestop price on Jan 16, 2026", figsize=(6, 4)
)
plt.show()

# ---- test prob at or above a price X ---- #
prob = est_gamestop.prob_at_or_above(120)
print(prob)

prob = est_gamestop.prob_below(150)
print(prob)

# ============================================
# Crude Oil
# ============================================

#     strike , last_price , bid , ask
column_mapping_wti = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
    "Type": "option_type",
}

# 2️⃣  market parameters
market_wti = MarketInputs(
    spot_price=61.69,  # current price of the underlying asset
    valuation_date=date(2025, 9, 8),
    expiry_date=date(2025, 12, 16),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_wti = ModelParams(
    solver="brent", fit_kde=False, price_method="mid", max_staleness_days=None
)

# 4️⃣  run using S&P500 e-mini futures options chain
est_wti = RND.from_csv(
    "data/WTIfutures_date250908exp251216_spot6169.csv",
    market_wti,
    model=model_wti,
    column_mapping=column_mapping_wti,
)

# PDF only
fig = est_wti.plot(
    kind="pdf",
    figsize=(8, 6),
    xlim=(0, 140),
    title="Implied crude oil price on Dec 16, 2025",
)
plt.show()

prob = est_wti.prob_below(200)
print(prob)


# ============================================
# Bitcoin
# ============================================

#     strike , last_price , bid , ask
column_mapping_bitcoin = {
    "Strike": "strike",
    "Last": "last_price",
    "Bid": "bid",
    "Ask": "ask",
    "OptionType": "option_type",
}

# 2️⃣  market parameters
market_bitcoin = MarketInputs(
    spot_price=108864,  # current price of the underlying asset
    valuation_date=date(2025, 8, 30),
    expiry_date=date(2025, 12, 26),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_bitcoin = ModelParams(fit_kde=False, price_method="mid", max_staleness_days=None)

# 4️⃣  run using S&P500 e-mini futures options chain
est_bitcoin = RND.from_csv(
    "data/bitcoin_date20250830_strike20251226_price108864.csv",
    market_bitcoin,
    model=model_bitcoin,
    column_mapping=column_mapping_bitcoin,
)

# PDF only
fig = est_bitcoin.plot(
    kind="pdf",
    figsize=(8, 6),
    xlim=(60000, 150000),
    title="Implied Bitcoin price on Dec 26, 2025",
)
plt.show()

# ============================================
# YFINANCE
# ============================================

# --- testing yfinance --- #
expiry_dates = RND.list_expiry_dates("NVDA")
print(expiry_dates[:])  # '2025-09-05', '2025-09-12', '2025-09-19',...]

# 2. Use ticker data with market parameters (current price fetched automatically)
market = MarketInputs(
    valuation_date=date(2025, 9, 1),
    expiry_date=date(2025, 12, 19),
    risk_free_rate=0.04199,
)

model = ModelParams(fit_kde=True, solver="brent")

# 3. Fetch and estimate - auto-fetched data is available in the result
result = RND.from_ticker("SPY", market, model=model)

# 4. Access auto-fetched data through the result (NOT through market!)
print(f"Fetched spot price: ${result.market.spot_price:.2f}")
print(f"Fetched dividend yield: {result.market.dividend_yield}")
print(f"Data sources: {result.summary()}")

# 5. Plot using the result object
result.plot()
plt.show()

result.plot(kind="pdf")
plt.show()

result.plot()
plt.show()

# Note: result.plot() uses the resolved market parameters from the calculation.
# The original MarketInputs object remains immutable and unchanged.
# Auto-fetched values are always accessed through result.market, not the original market object.

result.to_frame()

result.prob_below(177.82)

# ============================================
# Combined 1x3 panel for README (PDF only)
# ============================================

# Build a wide, horizontal figure with the first three results above
fig, axes = plt.subplots(1, 3, figsize=(15, 6))


def _plot_pdf(ax, est, title, xlim=None, ylim=None):
    ax.plot(est.prices, est.pdf, color="#1976D2", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Price at expiry", fontsize=10)
    # Intentionally omit y-axis label for cleaner 1x3 panel

    # Apply axis limits early so annotation positions respect them
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Spot marker + annotation
    spot = getattr(est.market, "spot_price", None)
    if spot is not None:
        ax.axvline(x=spot, color="black", linestyle="--", alpha=0.7, linewidth=1.0)
        # Add spot-price annotation similar to publication style
        try:
            val_date = est.market.valuation_date.strftime("%b %d, %Y")
        except Exception:
            val_date = None
        if val_date:
            price_text = f"Spot price on {val_date}\nis ${spot:,.2f}"
        else:
            price_text = f"Spot price\nis ${spot:,.2f}"
        # Position ~15% from bottom, slightly to the right of the line
        y_min, y_max = ax.get_ylim()
        y_text_pos = y_min + (y_max - y_min) * 0.15
        x_span = float(est.prices.max() - est.prices.min())
        ax.text(
            spot + 0.02 * x_span,
            y_text_pos,
            price_text,
            color="black",
            fontsize=14,
            fontstyle="italic",
            va="bottom",
            ha="left",
        )

    ax.grid(True, alpha=0.3)


_plot_pdf(
    axes[0],
    est_gamestop,
    title="Gamestop price on Jan 16, 2026",
    xlim=(0, 100),
    ylim=(0, 0.05),
)

_plot_pdf(
    axes[1],
    est_wti,
    title="WTI price on Dec 16, 2025",
    xlim=(20, 140),
    ylim=(0, 0.05),
)

_plot_pdf(
    axes[2],
    est_bitcoin,
    title="Bitcoin price on Dec 26, 2025",
    xlim=(60000, 150000),
    ylim=(0, 0.00008),
)

# Bitcoin axis: ticks every 20k (commas handled by integrated plotting style)
try:
    from matplotlib.ticker import MultipleLocator as _ML

    axes[2].xaxis.set_major_locator(_ML(20000))
except Exception:
    pass

# Overall title for the 1x3 panel
fig.suptitle(
    "Implied future prices of Gamestop, Crude Oil (WTI), and Bitcoin",
    fontsize=20,
    fontweight="bold",
    color="black",
)

# Tight layout with space for the suptitle
fig.tight_layout(rect=(0, 0, 1, 0.93))
plt.show()
