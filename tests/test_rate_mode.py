import math
from datetime import date, timedelta

from oipd.market_inputs import MarketInputs, resolve_market


def test_risk_free_rate_mode_annualized_to_continuous():
    valuation = date(2024, 1, 1)
    days = 120
    expiry = valuation + timedelta(days=days)
    y = 0.05  # 5% annualized nominal

    market_in = MarketInputs(
        valuation_date=valuation,
        expiry_date=expiry,
        risk_free_rate=y,
        risk_free_rate_mode="annualized",
        underlying_price=100.0,
    )

    resolved = resolve_market(market_in, vendor=None)

    T = days / 365.0
    expected_r_cont = math.log1p(y * T) / T
    assert abs(resolved.risk_free_rate - expected_r_cont) < 1e-12


def test_risk_free_rate_mode_continuous_passthrough():
    valuation = date(2024, 1, 1)
    expiry = date(2024, 2, 1)
    r_cont = 0.037

    market_in = MarketInputs(
        valuation_date=valuation,
        expiry_date=expiry,
        risk_free_rate=r_cont,
        risk_free_rate_mode="continuous",
        underlying_price=100.0,
    )

    resolved = resolve_market(market_in, vendor=None)
    assert abs(resolved.risk_free_rate - r_cont) < 1e-16

