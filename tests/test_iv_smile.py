import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from datetime import date

from oipd.estimator import RND, ModelParams
from oipd.market_inputs import MarketInputs


def _build_sample_chain() -> pd.DataFrame:
    """Create a representative options chain with last price quotes.

    Returns:
        pd.DataFrame: Options chain suitable for testing the smile helper.
    """
    return pd.DataFrame(
        {
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            "bid": [12.0, 7.8, 4.9, 2.9, 1.4],
            "ask": [13.0, 8.6, 5.3, 3.3, 1.8],
            "option_type": ["C", "C", "C", "C", "C"],
        }
    )


def _build_market_inputs() -> MarketInputs:
    """Construct market settings that align with the sample chain.

    Returns:
        MarketInputs: Market configuration aligned with the synthetic data.
    """
    return MarketInputs(
        valuation_date=date(2024, 1, 1),
        expiry_date=date(2024, 2, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
        risk_free_rate_mode="continuous",
    )


def test_iv_smile_default_grid():
    """Verify default grid evaluation for the fitted smile.

    Asserts that the helper produces a well-formed DataFrame with positive
    implied volatilities when no custom strike grid is supplied.
    """
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    smile = result.iv_smile()

    assert isinstance(smile, pd.DataFrame)
    assert list(smile.columns) == ["strike", "fitted_iv", "bid_iv", "ask_iv"]
    assert (smile["fitted_iv"] > 0).all()
    assert smile["bid_iv"].notna().all()
    assert smile["ask_iv"].notna().all()


def test_iv_smile_custom_strikes():
    """Ensure the smile helper respects a user-provided strike grid."""
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    strikes = [90.0, 100.0, 110.0]
    smile = result.iv_smile(strikes)

    assert list(smile["strike"]) == strikes
    assert list(smile.columns) == ["strike", "fitted_iv", "bid_iv", "ask_iv"]


def test_plot_iv_smile_includes_line_and_observed_points():
    """Ensure the publication plot renders the fitted line and scatter points."""
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    fig = result.plot(kind="iv_smile")
    ax = fig.axes[0]

    line_labels = [line.get_label() for line in ax.lines]
    assert "Fitted IV" in line_labels
    range_collections = [
        coll for coll in ax.collections if isinstance(coll, LineCollection)
    ]
    assert any(coll.get_label() == "Bid/Ask IV range" for coll in range_collections)

    plt.close(fig)


def test_plot_iv_smile_can_hide_observed_points():
    """Verify scatter points can be suppressed for the smile plot."""
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    fig = result.plot(kind="iv_smile", include_observed=False)
    ax = fig.axes[0]

    line_labels = [line.get_label() for line in ax.lines]
    assert "Fitted IV" in line_labels
    assert all(
        not isinstance(coll, LineCollection) for coll in ax.collections
    )

    plt.close(fig)
