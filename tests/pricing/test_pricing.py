import numpy as np
import pandas as pd
from datetime import date, timedelta

from oipd.pricing.black_scholes import black_scholes_call_price
from oipd.pricing.black76 import black76_call_price
from oipd.core.pdf import (
    calculate_pdf,
    _bs_iv_brent_method,
    _bs_iv_newton_method,
    _black76_iv_brent_method,
)


def test_european_price_with_yield():
    S = 100.0
    K = 100.0
    sigma = 0.2
    T = 0.5
    r = 0.0  # for textbook comparison
    q = 0.03

    price = black_scholes_call_price(S, K, sigma, T, r, q)
    assert abs(price - 4.8822) < 0.05  # within 5 cents


def test_iv_solvers_recover_sigma():
    S = 100.0
    K = 100.0
    true_sigma = 0.2
    T = 0.5
    r = 0.0
    q = 0.03
    price = black_scholes_call_price(S, K, true_sigma, T, r, q)

    brent = _bs_iv_brent_method(price, S, K, T, r, q)
    newton = _bs_iv_newton_method(price, S, K, T, r, q)

    for est in (brent, newton):
        assert abs(est - true_sigma) < 1e-4


def test_pdf_integrates_to_one():
    # Build synthetic option chain around spot 100
    S = 100.0
    strikes = np.arange(60, 141, 5)
    sigma = 0.2
    r = 0.01
    q = 0.02
    T = 0.5

    prices = black_scholes_call_price(S, strikes, sigma, T, r, q)
    df = pd.DataFrame({"strike": strikes, "last_price": prices})

    # Add required bid/ask columns for the new price_method functionality
    df["bid"] = prices * 0.95  # Mock bid slightly below last_price
    df["ask"] = prices * 1.05  # Mock ask slightly above last_price
    
    pdf_x, pdf_y = calculate_pdf(
        df,
        underlying_price=S,
        days_to_expiry=int(T * 365),
        risk_free_rate=r,
        solver_method="brent",
        pricing_engine="bs",
        dividend_yield=q,
        price_method="last",
    )

    area = np.trapz(pdf_y, pdf_x)
    assert abs(area - 1.0) < 1e-2  # Relaxed tolerance for numerical integration


def test_discrete_vs_continuous_equivalence():
    S0 = 100.0
    r = 0.05
    T = 0.5  # years
    cash_div = 1.0
    ex_date = date.today() + timedelta(days=90)

    schedule = pd.DataFrame({"ex_date": [ex_date], "amount": [cash_div]})

    # (a) Discrete cash adjustment
    from oipd.pricing.utils import prepare_dividends

    spot_a, q_a = prepare_dividends(
        underlying=S0,
        dividend_schedule=schedule,
        r=r,
        dividend_yield=None,
        valuation_date=date.today(),
    )
    price_a = black_scholes_call_price(spot_a, S0, 0.2, T, r, q_a)

    # (b) Equivalent flat yield
    pv_div = cash_div * np.exp(-r * (90 / 365))
    q_equiv = -np.log((S0 - pv_div) / S0) / T
    price_b = black_scholes_call_price(S0, S0, 0.2, T, r, q_equiv)

    assert abs(price_a - price_b) < 1e-4


def test_black76_round_trip():
    F = 100.0
    K = 100.0
    sigma = 0.2
    T = 0.5
    r = 0.01
    price = black76_call_price(F, K, sigma, T, r)
    est = _black76_iv_brent_method(price, F, K, T, r)
    assert abs(est - sigma) < 1e-4


def test_black76_bs_equivalence():
    S = 95.0
    F = 100.0
    K = 100.0
    sigma = 0.25
    T = 0.5
    r = 0.03
    q_star = r - np.log(F / S) / T
    price_black = black76_call_price(F, K, sigma, T, r)
    price_bs = black_scholes_call_price(S, K, sigma, T, r, q_star)
    assert abs(price_black - price_bs) < 1e-6
