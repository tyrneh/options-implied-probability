from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from oipd.estimator import ModelParams, RND
from oipd.market_inputs import MarketInputs
from oipd.pricing.black_scholes import black_scholes_call_price
from oipd.core.surface_fitting import SurfaceConfig, SVIFitDiagnostics


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
    diagnostics = getattr(vol_curve, "diagnostics", None)
    assert isinstance(diagnostics, SVIFitDiagnostics)
    assert diagnostics.status == "success"
    assert diagnostics.min_g is not None


def test_rnd_surface_diagnostics_bspline():
    df, market = _simple_chain()
    result = RND.from_dataframe(
        df,
        market,
        model=ModelParams(surface_fit=SurfaceConfig(name="bspline"), pricing_engine="black76"),
    )
    assert result.meta["surface_fit"] == "bspline"
    vol_curve = result.meta["vol_curve"]  # type: ignore[index]
    diagnostics = getattr(vol_curve, "diagnostics", None)
    assert diagnostics is not None
    if isinstance(diagnostics, SVIFitDiagnostics):
        assert diagnostics.status in {"success", "not_run"}
    else:
        assert isinstance(diagnostics, dict)
        assert "points_used" in diagnostics
