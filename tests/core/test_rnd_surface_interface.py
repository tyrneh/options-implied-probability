from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from oipd import MarketInputs, RNDSurface, VolModel
from oipd.core.ssvi import ssvi_total_variance
from oipd.core.svi import SVIParameters, svi_total_variance
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
    rho, eta, gamma = -0.35, 1.1, 0.6
    maturities = [0.25, 0.5]
    thetas = [0.05, 0.08]
    expiries = [pd.Timestamp(valuation + timedelta(days=int(T * 365))) for T in maturities]
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


def build_raw_svi_dataframe():
    valuation = date(2024, 1, 2)
    maturities = [0.25, 0.5]
    params_list = [
        SVIParameters(a=0.02, b=0.15, rho=-0.2, m=0.0, sigma=0.28),
        SVIParameters(a=0.045, b=0.25, rho=-0.25, m=0.03, sigma=0.32),
    ]
    expiries = [pd.Timestamp(valuation + timedelta(days=int(T * 365))) for T in maturities]
    rows = []
    F = 100.0

    for expiry, T, params in zip(expiries, maturities, params_list):
        k_grid = np.linspace(-0.4, 0.4, 5)
        strikes = F * np.exp(k_grid)
        w = svi_total_variance(k_grid, params)
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
    return pd.DataFrame(rows), params_list, maturities


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
            rho, eta, gamma = -0.35, 1.1, 0.6
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

    monkeypatch.setattr("oipd.surface.get_reader", lambda vendor: DummyReader)

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


def test_raw_svi_surface_calibration():
    data, params_list, maturities = build_raw_svi_dataframe()
    surface = RNDSurface.from_dataframe(
        data,
        make_market(),
        vol=VolModel(method="raw_svi"),
    )

    assert surface.vol_model.method == "raw_svi"
    first_maturity = 0.25
    tv = surface.total_variance(np.array([0.0]), first_maturity)
    expected = svi_total_variance(np.array([0.0]), params_list[0])
    assert np.allclose(tv, expected, atol=5e-3)
    diagnostics = surface.check_no_arbitrage()
    assert diagnostics["min_calendar_margin"] >= -1e-2


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
