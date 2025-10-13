import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pandas as pd
from datetime import date

from oipd.estimator import RND, ModelParams
from oipd.market_inputs import MarketInputs
from oipd.core.errors import CalculationError


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
    assert list(smile.columns) == ["strike", "fitted_iv", "bid_iv", "ask_iv", "last_iv"]
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
    assert list(smile.columns) == ["strike", "fitted_iv", "bid_iv", "ask_iv", "last_iv"]


def test_plot_iv_smile_includes_line_and_observed_points():
    """Ensure the publication plot renders the fitted line and scatter points."""
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    fig = result.plot_iv()
    ax = fig.axes[0]

    line_labels = [line.get_label() for line in ax.lines]
    assert "Fitted IV" in line_labels
    labels = [coll.get_label() for coll in ax.collections]
    assert any(label == "Bid/Ask IV range" for label in labels)

    plt.close(fig)


def test_plot_iv_smile_marker_style_shows_bid_ask_markers():
    """Marker style should render call/put bid/ask markers with distinct labels."""
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    rows = []
    for strike in strikes:
        intrinsic = max(0.0, strike - 100.0)
        call_price = max(1.5, 20.0 - 0.1 * (strike - 80.0))
        put_price = max(1.0, 0.1 * (strike - 80.0) + 2.0)
        rows.append(
            {
                "strike": strike,
                "last_price": call_price,
                "bid": call_price * 0.96,
                "ask": call_price * 1.04,
                "option_type": "C",
            }
        )
        rows.append(
            {
                "strike": strike,
                "last_price": put_price,
                "bid": put_price * 0.96,
                "ask": put_price * 1.04,
                "option_type": "P",
            }
        )
    chain = pd.DataFrame(rows)
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    bid_points = pd.DataFrame(
        {
            "strike": np.repeat(strikes, 2),
            "iv": np.linspace(0.25, 0.45, strikes.size * 2),
            "option_type": ["C", "P"] * strikes.size,
        }
    )
    ask_points = pd.DataFrame(
        {
            "strike": np.repeat(strikes, 2),
            "iv": np.linspace(0.26, 0.46, strikes.size * 2),
            "option_type": ["C", "P"] * strikes.size,
        }
    )
    result.meta["observed_iv_bid"] = bid_points
    result.meta["observed_iv_ask"] = ask_points

    fig = result.plot_iv(observed_style="markers")
    ax = fig.axes[0]

    labels = {coll.get_label() for coll in ax.collections if coll.get_label()}
    assert "Call bids" in labels
    assert "Call asks" in labels
    assert "Put bids" in labels
    assert "Put asks" in labels

    plt.close(fig)


def test_plot_iv_smile_can_hide_observed_points():
    """Verify scatter points can be suppressed for the smile plot."""
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    fig = result.plot_iv(include_observed=False)
    ax = fig.axes[0]

    line_labels = [line.get_label() for line in ax.lines]
    assert "Fitted IV" in line_labels
    assert not ax.collections

    plt.close(fig)


def test_plot_iv_smile_with_last_price_only():
    """Plot should fall back to last-price IV markers when bid/ask unavailable."""
    chain = _build_sample_chain().drop(columns=["bid", "ask"])
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    fig = result.plot_iv()
    ax = fig.axes[0]

    labels = [coll.get_label() for coll in ax.collections]
    assert "Observed IV" in labels

    plt.close(fig)


def test_svi_fallback_to_bspline(monkeypatch):
    """Ensure SVI failures emit a warning and fall back to B-spline."""
    chain = _build_sample_chain()
    market = _build_market_inputs()

    def fake_fit_surface(method, *, strikes, iv, options=None, **kwargs):
        if method == "svi":
            raise CalculationError("SVI failure")

        def curve(eval_strikes):
            eval_array = np.asarray(eval_strikes, dtype=float)
            return np.full_like(eval_array, 0.2, dtype=float)

        grid_x = np.asarray(strikes, dtype=float)
        grid_y = np.full_like(grid_x, 0.2, dtype=float)
        curve.grid = (grid_x, grid_y)  # type: ignore[attr-defined]
        return curve

    monkeypatch.setattr("oipd.estimator.fit_surface", fake_fit_surface)

    with pytest.warns(UserWarning, match="SVI calibration failed"):
        result = RND.from_dataframe(
            chain,
            market,
            model=ModelParams(
                price_method="last", pricing_engine="bs", surface_method="svi"
            ),
        )

    assert result.meta["surface_fit"] == "bspline"
