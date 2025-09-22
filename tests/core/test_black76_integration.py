import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from oipd.estimator import RND, ModelParams
from oipd.market_inputs import MarketInputs
from oipd.pricing.black_scholes import black_scholes_call_price


def _build_chain(include_puts: bool = True):
    S = 100.0
    r = 0.02
    q = 0.01
    T = 0.5
    sigma = 0.2
    strikes = [80.0, 90.0, 100.0, 110.0]
    rows = []
    for K in strikes:
        call = black_scholes_call_price(S, K, sigma, T, r, q)
        put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
        rows.append({"strike": K, "last_price": call, "option_type": "C"})
        if include_puts:
            rows.append({"strike": K, "last_price": put, "option_type": "P"})
    return pd.DataFrame(rows), S, r, T


def test_black76_estimator_with_puts():
    df, S, r, T = _build_chain(include_puts=True)
    market = MarketInputs(
        valuation_date=date(2024, 1, 1),
        expiry_date=date(2024, 1, 1) + timedelta(days=int(T * 365)),
        underlying_price=S,
        risk_free_rate=r,
        risk_free_rate_mode="continuous",
    )
    result = RND.from_dataframe(
        df, market, model=ModelParams(pricing_engine="black76")
    )
    assert result.pdf is not None


def test_black76_requires_puts():
    df, S, r, T = _build_chain(include_puts=False)
    market = MarketInputs(
        valuation_date=date(2024, 1, 1),
        expiry_date=date(2024, 1, 1) + timedelta(days=int(T * 365)),
        underlying_price=S,
        risk_free_rate=r,
        risk_free_rate_mode="continuous",
    )
    with pytest.raises(ValueError):
        RND.from_dataframe(
            df, market, model=ModelParams(pricing_engine="black76")
        )
