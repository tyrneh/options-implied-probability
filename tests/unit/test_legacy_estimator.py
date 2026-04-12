"""Focused smoke tests for the legacy estimator maturity migration."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from oipd.market_inputs import MarketInputs, resolve_market
from oipd.pipelines._legacy.estimator import ModelParams, RNDResult, _estimate
from oipd.pricing.utils import implied_dividend_yield_from_forward


def _build_single_expiry_chain(
    *,
    expiry: pd.Timestamp,
    valuation_timestamp: pd.Timestamp,
    spot: float = 100.0,
    rate: float = 0.05,
    intraday_dense: bool = False,
) -> pd.DataFrame:
    """Build a simple parity-consistent single-expiry chain."""
    if intraday_dense:
        strikes = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
        call_prices = np.array(
            [20.75, 16.0, 11.25, 6.75, 3.0, 1.05, 0.4, 0.15, 0.08],
            dtype=float,
        )
        bid_prices = np.array(
            [20.5, 15.8, 11.0, 6.5, 2.8, 0.9, 0.3, 0.1, 0.05],
            dtype=float,
        )
        ask_prices = np.array(
            [21.0, 16.2, 11.5, 7.0, 3.2, 1.2, 0.5, 0.2, 0.1],
            dtype=float,
        )
    else:
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0], dtype=float)
        call_prices = np.array([12.5, 8.2, 5.1, 3.1, 1.6], dtype=float)
        bid_prices = call_prices - 0.2
        ask_prices = call_prices + 0.2

    time_to_expiry_years = float(
        (expiry - valuation_timestamp).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)
    )
    discount_factor = np.exp(-rate * time_to_expiry_years)

    calls = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": call_prices,
            "bid": bid_prices,
            "ask": ask_prices,
            "option_type": ["C"] * len(strikes),
            "expiry": [expiry] * len(strikes),
        }
    )
    puts = calls.copy()
    puts["option_type"] = "P"
    puts["last_price"] = np.abs(call_prices - spot + strikes * discount_factor)
    puts["bid"] = np.abs(calls["bid"].to_numpy() - spot + strikes * discount_factor)
    puts["ask"] = np.abs(calls["ask"].to_numpy() - spot + strikes * discount_factor)
    return pd.concat([calls, puts], ignore_index=True)


def test_legacy_estimate_date_only_single_expiry_smoke():
    """Legacy estimator should resolve date-only maturity locally."""
    expiry = pd.Timestamp("2025-03-21 00:00:00")
    market = resolve_market(
        MarketInputs(
            valuation_date=date(2025, 1, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
        )
    )
    chain = _build_single_expiry_chain(
        expiry=expiry,
        valuation_timestamp=market.valuation_timestamp,
        intraday_dense=True,
    )

    prices, pdf, cdf, meta = _estimate(
        chain,
        market,
        ModelParams(pricing_engine="black76", surface_method="bspline"),
    )

    resolved_maturity = meta["resolved_maturity"]
    expected_years = 79.0 / 365.0

    assert prices.shape == pdf.shape == cdf.shape
    assert resolved_maturity.expiry == expiry
    assert resolved_maturity.time_to_expiry_years == pytest.approx(expected_years)
    assert resolved_maturity.calendar_days_to_expiry == 79


def test_legacy_estimate_same_day_intraday_smoke():
    """Legacy estimator should preserve same-day intraday maturity precision."""
    expiry = pd.Timestamp("2025-01-01 16:00:00")
    market = resolve_market(
        MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
        )
    )
    chain = _build_single_expiry_chain(
        expiry=expiry,
        valuation_timestamp=market.valuation_timestamp,
        intraday_dense=True,
    )

    _, _, _, meta = _estimate(
        chain,
        market,
        ModelParams(pricing_engine="black76", surface_method="bspline"),
    )

    resolved_maturity = meta["resolved_maturity"]
    assert resolved_maturity.time_to_expiry_days > 0.0
    assert resolved_maturity.calendar_days_to_expiry == 0
    assert resolved_maturity.time_to_expiry_years == pytest.approx(6.5 / (24.0 * 365.0))


def test_legacy_rndresult_summary_uses_calendar_days():
    """Legacy summary should use resolved maturity instead of market day aliases."""
    market = resolve_market(
        MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
        )
    )
    expiry = pd.Timestamp("2025-01-01 16:00:00")
    chain = _build_single_expiry_chain(
        expiry=expiry,
        valuation_timestamp=market.valuation_timestamp,
        intraday_dense=True,
    )
    prices, pdf, cdf, meta = _estimate(
        chain,
        market,
        ModelParams(pricing_engine="black76", surface_method="bspline"),
    )
    result = RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=market, meta=meta)

    summary = result.summary()

    assert "calendar_days_to_expiry=0" in summary


def test_legacy_rndresult_implied_dividend_yield_uses_time_to_expiry_years():
    """Legacy implied yield should use precise year fraction, not integer days."""
    market = resolve_market(
        MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
        )
    )
    expiry = pd.Timestamp("2025-01-01 16:00:00")
    chain = _build_single_expiry_chain(
        expiry=expiry,
        valuation_timestamp=market.valuation_timestamp,
        intraday_dense=True,
    )
    prices, pdf, cdf, meta = _estimate(
        chain,
        market,
        ModelParams(pricing_engine="black76", surface_method="bspline"),
    )
    meta["forward_price"] = 101.0
    result = RNDResult(prices=prices, pdf=pdf, cdf=cdf, market=market, meta=meta)

    expected = implied_dividend_yield_from_forward(
        100.0,
        101.0,
        0.05,
        meta["resolved_maturity"].time_to_expiry_years,
    )

    assert result.implied_dividend_yield() == pytest.approx(expected)
