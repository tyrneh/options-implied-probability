from __future__ import annotations

from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import numpy as np
import pandas as pd
import pytest

from oipd import MarketInputs, RNDSurface, VolModel
from oipd.core.vol_surface_fitting.shared.ssvi import ssvi_total_variance
from oipd.pricing.black76 import black76_call_price


def make_market(expiry: date | None = None) -> MarketInputs:
    return MarketInputs(
        risk_free_rate=0.0,
        valuation_date=date(2024, 1, 2),
        risk_free_rate_mode="annualized",
        underlying_price=100.0,
        dividend_yield=0.0,
        dividend_schedule=None,
        expiry_date=expiry,
    )


def build_synthetic_dataframe() -> pd.DataFrame:
    valuation = date(2024, 1, 2)
    rho, eta, gamma = -0.35, 0.9, 0.3
    maturities = [0.25, 0.5]
    thetas = [0.05, 0.08]
    expiries = [
        pd.Timestamp(valuation + timedelta(days=int(T * 365))) for T in maturities
    ]
    rows = []
    F = 100.0

    for expiry, T, theta in zip(expiries, maturities, thetas):
        k_grid = np.array([-0.3, 0.0, 0.3])
        strikes = F * np.exp(k_grid)
        w = ssvi_total_variance(k_grid, theta, rho, eta, gamma)
        sigma = np.sqrt(w / T)
        call_prices = black76_call_price(F, strikes, sigma, T, 0.0)
        put_prices = call_prices - (F - strikes)
        for strike, call_price, put_price in zip(strikes, call_prices, put_prices):
            rows.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": "C",
                    "bid": call_price,
                    "ask": call_price,
                    "last_price": call_price,
                    "last_trade_date": pd.Timestamp(valuation),
                }
            )
            rows.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": "P",
                    "bid": put_price,
                    "ask": put_price,
                    "last_price": put_price,
                    "last_trade_date": pd.Timestamp(valuation),
                }
            )
    return pd.DataFrame(rows)


def test_surface_from_dataframe_defaults_to_ssvi():
    data = build_synthetic_dataframe()

    surface = RNDSurface.from_dataframe(data, make_market())

    assert surface.vol_model.method == "ssvi"
    assert len(surface.expiries) == 2

    first_maturity = 0.25
    tv = surface.total_variance(np.array([0.0]), first_maturity)
    assert np.allclose(tv, [0.05], atol=5e-3)

    iv = surface.iv(np.array([100.0]), first_maturity)
    expected_sigma = np.sqrt(0.05 / first_maturity)
    assert np.allclose(iv, expected_sigma, atol=5e-3)

    diagnostics = surface.check_no_arbitrage()
    assert diagnostics["min_calendar_margin"] >= -1e-3
    assert len(diagnostics["calendar_margins"]) == 1


def test_surface_from_dataframe_supports_column_mapping():
    data = build_synthetic_dataframe()
    renamed = data.rename(
        columns={
            "expiry": "expiration",
            "last_price": "settlement",
            "bid": "bid_px",
            "ask": "ask_px",
            "option_type": "call_put",
        }
    )

    mapping = {
        "expiration": "expiry",
        "settlement": "last_price",
        "bid_px": "bid",
        "ask_px": "ask",
        "call_put": "option_type",
    }

    surface = RNDSurface.from_dataframe(
        renamed,
        make_market(),
        column_mapping=mapping,
    )

    assert surface.vol_model.method == "ssvi"
    assert len(surface.expiries) == len(data["expiry"].unique())


def test_surface_rejects_slice_only_vol_methods():
    data = pd.DataFrame(
        {
            "expiry": [pd.Timestamp("2024-04-01")],
            "strike": [100],
            "option_type": ["C"],
            "bid": [2.0],
            "ask": [2.2],
            "last_price": [2.1],
            "last_trade_date": [pd.Timestamp("2024-01-10")],
        }
    )

    with pytest.raises(ValueError):
        RNDSurface.from_dataframe(data, make_market(), vol=VolModel(method="svi"))


def test_surface_from_csv_supports_column_mapping(tmp_path):
    data = build_synthetic_dataframe()
    renamed = data.rename(
        columns={
            "expiry": "expiration",
            "last_price": "settlement",
            "bid": "bid_px",
            "ask": "ask_px",
            "option_type": "call_put",
        }
    )
    csv_path = tmp_path / "options.csv"
    renamed.to_csv(csv_path, index=False)

    mapping = {
        "expiration": "expiry",
        "settlement": "last_price",
        "bid_px": "bid",
        "ask_px": "ask",
        "call_put": "option_type",
    }

    surface = RNDSurface.from_csv(
        str(csv_path),
        make_market(),
        column_mapping=mapping,
    )

    assert surface.vol_model.method == "ssvi"
    assert len(surface.expiries) == len(data["expiry"].unique())


def test_surface_from_ticker_respects_horizon(monkeypatch):
    class DummyReader:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def list_expiry_dates(_ticker: str) -> list[str]:
            return ["2024-02-01", "2024-04-01", "2024-12-01"]

        def read(self, ticker_expiry: str, column_mapping=None):
            expiry_dt = pd.Timestamp(ticker_expiry.split(":")[-1])
            valuation = date(2024, 1, 2)
            days = (expiry_dt.date() - valuation).days
            T = max(days / 365.0, 5 / 365.0)
            rho, eta, gamma = -0.35, 0.9, 0.3
            theta = 0.04 + 0.05 * min(T, 0.7)
            k_grid = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
            strikes = 100.0 * np.exp(k_grid)
            w = ssvi_total_variance(k_grid, theta, rho, eta, gamma)
            sigma = np.sqrt(w / T)
            call_prices = black76_call_price(100.0, strikes, sigma, T, 0.0)
            put_prices = call_prices - (100.0 - strikes)
            rows = []
            for strike, call_price, put_price in zip(strikes, call_prices, put_prices):
                rows.append(
                    {
                        "expiry": expiry_dt,
                        "strike": strike,
                        "option_type": "C",
                        "bid": call_price,
                        "ask": call_price,
                        "last_price": call_price,
                        "last_trade_date": pd.Timestamp(valuation),
                    }
                )
                rows.append(
                    {
                        "expiry": expiry_dt,
                        "strike": strike,
                        "option_type": "P",
                        "bid": put_price,
                        "ask": put_price,
                        "last_price": put_price,
                        "last_trade_date": pd.Timestamp(valuation),
                    }
                )
            df = pd.DataFrame(rows)
            df.attrs["underlying_price"] = 100.0
            df.attrs["dividend_yield"] = 0.0
            df.attrs["dividend_schedule"] = None
            return df

    monkeypatch.setattr(
        "oipd.pipelines.rnd_surface.get_reader", lambda vendor: DummyReader
    )

    market = make_market()
    surface = RNDSurface.from_ticker(
        "AAPL",
        market,
        horizon="90D",
        vendor="dummy",
    )

    assert surface.vol_model.method == "ssvi"
    assert surface.source == "ticker"
    assert len(surface.expiries) == 2  # only expiries within 90 days retained

    diagnostics = surface.check_no_arbitrage()
    assert "objective" in diagnostics


def test_ssvi_alpha_tilt_applied():
    data = build_synthetic_dataframe()
    surface = RNDSurface.from_dataframe(data, make_market())

    # Evaluate total variance on a k-grid at the first maturity
    t = 0.25
    k_grid = np.array([-0.3, 0.0, 0.2])
    w_base = surface.total_variance(k_grid, t)

    # Inject a positive alpha tilt into the calibrated params
    from dataclasses import replace

    alpha = 0.02
    assert surface.vol_model.method == "ssvi"
    # Replace the frozen fit and params with updated alpha
    surface._ssvi_fit = replace(
        surface._ssvi_fit,  # type: ignore[assignment]
        params=replace(surface._ssvi_fit.params, alpha=alpha),  # type: ignore[arg-type]
    )

    w_tilted = surface.total_variance(k_grid, t)
    expected = w_base + alpha * t
    assert np.allclose(w_tilted, expected, atol=1e-12)


def test_surface_plot_iv_overlay_legend_text_color():
    data = build_synthetic_dataframe()
    surface = RNDSurface.from_dataframe(data, make_market())

    fig = surface.plot_iv()
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_colors = {to_hex(text.get_color()) for text in legend.get_texts()}
    assert legend_colors == {"#333333"}
    plt.close(fig)


def test_surface_plot_iv_grid_layout_axes_count():
    data = build_synthetic_dataframe()
    surface = RNDSurface.from_dataframe(data, make_market())

    fig = surface.plot_iv(layout="grid", figsize=(4.0, 3.0))
    fitted_lines = sum(
        1 for ax in fig.axes for line in ax.lines if line.get_label() == "Fitted IV"
    )
    assert fitted_lines >= len(surface.expiries)
    plt.close(fig)


def test_surface_plot_iv_grid_includes_observed_quotes():
    data = build_synthetic_dataframe()
    surface = RNDSurface.from_dataframe(data, make_market())

    fig = surface.plot_iv(layout="grid")
    assert any(ax.collections for ax in fig.axes)
    plt.close(fig)


def test_surface_plot_iv_3d_returns_plotly_figure():
    plotly = pytest.importorskip("plotly.graph_objects")

    data = build_synthetic_dataframe()
    surface = RNDSurface.from_dataframe(data, make_market())

    fig = surface.plot_iv_3d()
    assert isinstance(fig, plotly.Figure)
    assert fig.data
