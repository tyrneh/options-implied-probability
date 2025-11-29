from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from oipd.core.vol_surface_fitting.shared.ssvi import (
    check_ssvi_calendar,
    check_ssvi_constraints,
)
from oipd.core.vol_surface_fitting.shared.svi import SVIParameters, check_butterfly
from oipd import ModelParams, RND
from oipd.market_inputs import MarketInputs
from oipd.pricing.black_scholes import black_scholes_call_price


def _simple_chain():
    S = 100.0
    r = 0.01
    q = 0.0
    T = 0.25
    sigma = 0.2
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    rows = []
    for K in strikes:
        call = black_scholes_call_price(S, K, sigma, T, r, q)
        put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
        rows.append({"strike": K, "last_price": call, "option_type": "C"})
        rows.append({"strike": K, "last_price": put, "option_type": "P"})
    df = pd.DataFrame(rows)
    market = MarketInputs(
        valuation_date=date(2024, 1, 1),
        expiry_date=date(2024, 1, 1) + timedelta(days=int(T * 365)),
        underlying_price=S,
        risk_free_rate=r,
        risk_free_rate_mode="continuous",
    )
    return df, market


def test_rnd_surface_diagnostics_default_svi():
    df, market = _simple_chain()
    result = RND.from_dataframe(df, market, model=ModelParams(pricing_engine="black76"))
    assert result.meta["surface_fit"] == "svi"
    vol_curve = result.meta["vol_curve"]  # type: ignore[index]
    assert callable(vol_curve)
    assert hasattr(vol_curve, "params")


def test_rnd_surface_diagnostics_bspline():
    df, market = _simple_chain()
    result = RND.from_dataframe(
        df,
        market,
        model=ModelParams(surface_method="bspline", pricing_engine="black76"),
    )
    assert result.meta["surface_fit"] == "bspline"
    vol_curve = result.meta["vol_curve"]  # type: ignore[index]
    assert callable(vol_curve)
    assert hasattr(vol_curve, "params")
    assert "method" in vol_curve.params


def test_check_butterfly_reports_min_margin():
    params = SVIParameters(a=0.02, b=0.15, rho=-0.2, m=0.0, sigma=0.25)
    grid = np.linspace(-0.6, 0.6, 41)
    diagnostics = check_butterfly(params, grid)
    assert "min_margin" in diagnostics
    assert np.isscalar(diagnostics["min_margin"])


def test_check_ssvi_constraints_and_calendar():
    theta = [0.05, 0.08, 0.11]
    rho = -0.3
    eta = 0.9
    gamma = 0.3
    inequality = check_ssvi_constraints(theta, rho, eta, gamma)
    assert inequality["min_theta_phi_margin"] >= -1e-8
    assert len(inequality["theta_phi_margins"]) == len(theta)

    calendar = check_ssvi_calendar(
        theta, rho, eta, gamma, k_grid=np.linspace(-0.5, 0.5, 41)
    )
    assert calendar["min_margin"] >= -1e-8
    assert len(calendar["margins"]) == len(theta) - 1
