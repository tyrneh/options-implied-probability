from oipd import RND, MarketInputs, ModelParams
import matplotlib.pyplot as plt
from datetime import date

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
    underlying_price=61.69,  # current price of the underlying instrument
    valuation_date=date(2025, 9, 8),
    expiry_date=date(2025, 12, 16),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_wti = ModelParams(solver="brent", price_method="mid", max_staleness_days=1)

# 4️⃣  run using WTI options chain
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
    underlying_price=108864,  # current price of the underlying instrument
    valuation_date=date(2025, 8, 30),
    expiry_date=date(2025, 12, 26),
    risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
)

# 3️⃣  optional model knobs (could omit)
model_bitcoin = ModelParams(price_method="mid", max_staleness_days=1)

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

# --- using yfinance --- #
expiry_dates = RND.list_expiry_dates("GME")
print(expiry_dates[:])  # '2025-09-05', '2025-09-12', '2025-09-19',...]

# 2. Use ticker data with market parameters (current price fetched automatically)
market = MarketInputs(
    valuation_date=date.today(),
    expiry_date=date(2026, 1, 16),
    risk_free_rate=0.04199,
)

# 3. Fetch and estimate - auto-fetched data is available in the result
est_gamestop = RND.from_ticker("GME", market)

# 4. Access auto-fetched data through the result (NOT through market!)
print(f"Fetched current price: ${est_gamestop.market.current_price:.2f}")
print(f"Fetched dividend yield: {est_gamestop.market.dividend_yield}")
print(f"Data sources: {est_gamestop.summary()}")

# 5. Plot using the result object
est_gamestop.plot()
plt.show()


# Note: result.plot() uses the resolved market parameters from the calculation.
# The original MarketInputs object remains immutable and unchanged.
# Auto-fetched values are always accessed through result.market, not the original market object.


# ============================================
# Combined 1x3 panel for README
# ============================================

# Build a wide, horizontal figure with the first three results above
fig, axes = plt.subplots(1, 3, figsize=(15, 6))


def _plot_pdf(ax, est, title, xlim=None, ylim=None, title_size: int = 18):
    ax.plot(est.prices, est.pdf, color="#1976D2", linewidth=1.5)
    # Allow per-plot control of the title font size
    ax.set_title(title, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("Price at expiry", fontsize=14)
    # Intentionally omit y-axis label for cleaner 1x3 panel

    # Apply axis limits early so annotation positions respect them
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Spot marker + annotation
    # Use generic alias to avoid terminology confusion across asset classes
    spot = getattr(
        est.market, "current_price", getattr(est.market, "underlying_price", None)
    )
    if spot is not None:
        ax.axvline(x=spot, color="black", linestyle="--", alpha=0.7, linewidth=1.0)
        # Add spot-price annotation similar to publication style
        try:
            val_date = est.market.valuation_date.strftime("%b %d, %Y")
        except Exception:
            val_date = None
        if val_date:
            price_text = f"Price on {val_date}\nis ${spot:,.2f}"
        else:
            price_text = f"Price\nis ${spot:,.2f}"
        # Position ~15% from bottom, slightly to the right of the line
        y_min, y_max = ax.get_ylim()
        y_text_pos = y_min + (y_max - y_min) * 0.15
        x_span = float(est.prices.max() - est.prices.min())
        ax.text(
            spot + 0.02 * x_span,
            y_text_pos,
            price_text,
            color="black",
            fontsize=18,
            fontstyle="italic",
            va="bottom",
            ha="left",
        )

    ax.grid(True, alpha=0.3)


_plot_pdf(
    axes[0],
    est_gamestop,
    title="Gamestop on Jan 16, 2026",
    xlim=(0, 100),
    ylim=(0, 0.05),
)

_plot_pdf(
    axes[1],
    est_wti,
    title="WTI on Dec 16, 2025",
    xlim=(20, 140),
    ylim=(0, 0.05),
)

_plot_pdf(
    axes[2],
    est_bitcoin,
    title="Bitcoin on Dec 26, 2025",
    xlim=(80000, 150000),
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
    fontsize=22,
    fontweight="bold",
    color="black",
)

# Tight layout with space for the suptitle
fig.tight_layout(rect=(0, 0, 1, 0.93))
plt.show()
