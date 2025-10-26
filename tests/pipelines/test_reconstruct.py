from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from oipd import (
    MarketInputs,
    ModelParams,
    RNDSurface,
    RND,
    VolModel,
    rebuild_slice_from_svi,
    rebuild_surface_from_ssvi,
)
from oipd.pricing.black76 import black76_call_price


def _sample_option_dataset(forward: float, maturity_years: float, rate: float) -> pd.DataFrame:
    """Build a small option chain with both calls and puts."""

    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)
    sigma = 0.25
    calls = black76_call_price(forward, strikes, sigma, maturity_years, rate)
    discount = np.exp(-rate * maturity_years)

    rows = []
    for strike, call_price in zip(strikes, calls):
        put_price = call_price - discount * (forward - strike)
        rows.append(
            {
                "strike": strike,
                "last_price": call_price,
                "bid": call_price * 0.98,
                "ask": call_price * 1.02,
                "option_type": "call",
            }
        )
        rows.append(
            {
                "strike": strike,
                "last_price": put_price,
                "bid": put_price * 0.98,
                "ask": put_price * 1.02,
                "option_type": "put",
            }
        )
    return pd.DataFrame(rows)


def test_rebuild_slice_from_svi_matches_rnd_result() -> None:
    """Ensure reconstruction reproduces the calibrated smile and density."""

    forward = 100.0
    rate = 0.02
    days = 30
    maturity_years = days / 365.0

    options_df = _sample_option_dataset(forward, maturity_years, rate)
    market = MarketInputs(
        valuation_date=date(2024, 1, 1),
        expiry_date=date(2024, 1, 31),
        underlying_price=forward,
        risk_free_rate=rate,
    )

    model = ModelParams(price_method="last", pricing_engine="black76")
    result = RND.from_dataframe(options_df, market, column_mapping=None, model=model)

    svi_params = result.svi_params()
    forward_price = float(result.meta["forward_price"])
    strike_grid = result.meta["vol_curve"].grid[0]

    rebuilt = rebuild_slice_from_svi(
        svi_params,
        forward_price=forward_price,
        days_to_expiry=result.market.days_to_expiry,
        risk_free_rate=float(result.market.risk_free_rate),
        strike_grid=strike_grid,
    )

    original_vol = result.meta["vol_curve"](strike_grid)
    np.testing.assert_allclose(rebuilt.vol_curve(strike_grid), original_vol)
    np.testing.assert_allclose(rebuilt.data["pdf"].to_numpy(), result.pdf)
    np.testing.assert_allclose(rebuilt.data["cdf"].to_numpy(), result.cdf)


def _surface_dataset() -> tuple[pd.DataFrame, MarketInputs, ModelParams, VolModel]:
    valuation = date(2024, 1, 1)
    risk_free_rate = 0.02
    underlying = 100.0

    expiries = [date(2024, 1, 31), date(2024, 3, 1)]
    sigma_levels = [0.22, 0.24]
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)

    rows = []
    for expiry, sigma in zip(expiries, sigma_levels):
        days = (expiry - valuation).days
        T = days / 365.0
        forward = underlying * np.exp(risk_free_rate * T)
        df = np.exp(-risk_free_rate * T)
        calls = black76_call_price(forward, strikes, sigma, T, risk_free_rate)
        for strike, call_price in zip(strikes, calls):
            put_price = call_price - df * (forward - strike)
            rows.append(
                {
                    "strike": strike,
                    "last_price": call_price,
                    "bid": call_price * 0.98,
                    "ask": call_price * 1.02,
                    "option_type": "call",
                    "expiration": expiry.strftime("%Y-%m-%d"),
                }
            )
            rows.append(
                {
                    "strike": strike,
                    "last_price": put_price,
                    "bid": put_price * 0.98,
                    "ask": put_price * 1.02,
                    "option_type": "put",
                    "expiration": expiry.strftime("%Y-%m-%d"),
                }
            )

    df = pd.DataFrame(rows)
    market = MarketInputs(
        valuation_date=valuation,
        risk_free_rate=risk_free_rate,
        underlying_price=underlying,
    )
    model = ModelParams(price_method="last", pricing_engine="black76")
    vol_model = VolModel(method="ssvi")
    return df, market, model, vol_model


def test_rebuild_surface_from_ssvi_matches_surface_result() -> None:
    options_df, market, model, vol_model = _surface_dataset()
    surface = RNDSurface.from_dataframe(
        options_df,
        market,
        model=model,
        vol=vol_model,
        column_mapping={"expiration": "expiry"},
    )

    ssvi_df = surface.ssvi_params()
    forward_map = surface.forward_levels()
    strike_grids = {
        maturity: tuple(np.linspace(0.5, 1.5, 201) * forward)
        for maturity, forward in forward_map.items()
    }

    rebuilt_surface = rebuild_surface_from_ssvi(
        ssvi_df,
        forwards=forward_map,
        risk_free_rate=float(market.risk_free_rate),
        strike_grids=strike_grids,
    )

    for maturity in rebuilt_surface.available_maturities():
        rebuilt_slice = rebuilt_surface.slice(maturity)
        forward = forward_map[maturity]
        strike_grid = rebuilt_slice.data["strike"].to_numpy()
        original_vol = surface.iv(strike_grid, maturity, forward)
        np.testing.assert_allclose(rebuilt_slice.vol_curve(strike_grid), original_vol)

        original_slice = surface.slice(maturity)
        np.testing.assert_allclose(rebuilt_slice.data["strike"].to_numpy(), original_slice.prices)
        np.testing.assert_allclose(
            rebuilt_slice.data["pdf"].to_numpy(),
            original_slice.pdf,
            rtol=1e-5,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rebuilt_slice.data["cdf"].to_numpy(),
            original_slice.cdf,
            rtol=1e-5,
            atol=1e-5,
        )

    # Arbitrary maturity interpolation
    maturities = sorted(rebuilt_surface.available_maturities())
    mid_maturity = float(np.mean(maturities))
    original_mid_slice = surface.slice(mid_maturity)
    rebuilt_mid_slice = rebuilt_surface.slice(
        mid_maturity,
        strike_grid=original_mid_slice.prices,
    )
    np.testing.assert_allclose(
        rebuilt_mid_slice.vol_curve(original_mid_slice.prices),
        surface.iv(original_mid_slice.prices, mid_maturity),
    )
    np.testing.assert_allclose(
        rebuilt_mid_slice.data["pdf"].to_numpy(),
        original_mid_slice.pdf,
        rtol=1e-5,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rebuilt_mid_slice.data["cdf"].to_numpy(),
        original_mid_slice.cdf,
        rtol=1e-5,
        atol=1e-5,
    )
