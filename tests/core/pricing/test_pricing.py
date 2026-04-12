import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from oipd.pricing.black_scholes import black_scholes_call_price
from oipd.pricing.black76 import black76_call_price
from oipd.core.data_processing.iv import (
    compute_iv,
    bs_iv_brent_method,
    bs_iv_newton_method,
    black76_iv_brent_method,
    smooth_iv,
)
from oipd.core.data_processing.selection import select_price_column
from oipd.core.errors import InvalidInputError
from oipd.core.probability_density_conversion import (
    price_curve_from_iv,
    pdf_from_price_curve,
)
from oipd.market_inputs import MarketInputs, resolve_market


def _discounted_cashflow(amount: float, r: float, tau_years: float) -> float:
    """Return PV of one discrete dividend cashflow for expected-value checks."""
    return float(amount * np.exp(-r * tau_years))


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

    brent = bs_iv_brent_method(price, S, K, T, r, q=q)
    newton = bs_iv_newton_method(price, S, K, T, r, q=q)

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

    priced, _ = select_price_column(df, "last")
    iv_df = compute_iv(
        priced,
        S,
        time_to_expiry_years=T,
        risk_free_rate=r,
        solver_method="brent",
        pricing_engine="bs",
        dividend_yield=q,
    )

    vol_curve = smooth_iv(
        "bspline",
        iv_df["strike"].to_numpy(),
        iv_df["iv"].to_numpy(),
    )

    strikes_grid, call_prices = price_curve_from_iv(
        vol_curve,
        S,
        time_to_expiry_years=T,
        risk_free_rate=r,
        pricing_engine="bs",
        dividend_yield=q,
    )

    pdf_x, pdf_y = pdf_from_price_curve(
        strikes_grid,
        call_prices,
        risk_free_rate=r,
        time_to_expiry_years=T,
        min_strike=float(priced["strike"].min()),
        max_strike=float(priced["strike"].max()),
    )

    area = np.trapezoid(pdf_y, pdf_x)
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


def test_prepare_dividends_respects_same_day_timestamp_ordering():
    """Discrete dividends should flip inclusion when timing crosses valuation."""
    from oipd.pricing.utils import prepare_dividends

    valuation_timestamp = pd.Timestamp("2025-01-01 09:30:00")
    schedule_before = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2025-01-01 08:00:00")], "amount": [1.0]}
    )
    schedule_after = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2025-01-01 12:00:00")], "amount": [1.0]}
    )

    spot_before, q_before = prepare_dividends(
        underlying=100.0,
        dividend_schedule=schedule_before,
        r=0.05,
        dividend_yield=None,
        valuation_date=valuation_timestamp,
    )
    spot_after, q_after = prepare_dividends(
        underlying=100.0,
        dividend_schedule=schedule_after,
        r=0.05,
        dividend_yield=None,
        valuation_date=valuation_timestamp,
    )

    assert spot_before == pytest.approx(100.0)
    assert q_before == pytest.approx(0.0)
    assert spot_after < 100.0
    assert q_after == pytest.approx(0.0)


def test_prepare_dividends_excludes_cash_dividend_before_valuation():
    """Dividend cashflows before valuation should not reduce adjusted spot."""
    from oipd.pricing.utils import prepare_dividends

    valuation_timestamp = pd.Timestamp("2025-01-01 09:30:00")
    expiry_timestamp = pd.Timestamp("2025-01-10 16:00:00")
    schedule = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2025-01-01 08:00:00")], "amount": [1.25]}
    )

    adjusted_spot, effective_q = prepare_dividends(
        underlying=100.0,
        dividend_schedule=schedule,
        r=0.05,
        dividend_yield=None,
        valuation_date=valuation_timestamp,
        expiry=expiry_timestamp,
    )

    assert adjusted_spot == pytest.approx(100.0)
    assert effective_q == pytest.approx(0.0)


def test_prepare_dividends_excludes_cash_dividend_after_expiry():
    """Dividend cashflows after expiry should not reduce adjusted spot."""
    from oipd.pricing.utils import prepare_dividends

    valuation_timestamp = pd.Timestamp("2025-01-01 09:30:00")
    expiry_timestamp = pd.Timestamp("2025-01-10 16:00:00")
    schedule = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2025-01-11 09:30:00")], "amount": [1.25]}
    )

    adjusted_spot, effective_q = prepare_dividends(
        underlying=100.0,
        dividend_schedule=schedule,
        r=0.05,
        dividend_yield=None,
        valuation_date=valuation_timestamp,
        expiry=expiry_timestamp,
    )

    assert adjusted_spot == pytest.approx(100.0)
    assert effective_q == pytest.approx(0.0)


def test_prepare_dividends_includes_cash_dividend_between_valuation_and_expiry():
    """Dividend cashflows inside the live option window should reduce spot."""
    from oipd.pricing.utils import prepare_dividends

    valuation_timestamp = pd.Timestamp("2025-01-01 09:30:00")
    ex_timestamp = pd.Timestamp("2025-01-05 09:30:00")
    expiry_timestamp = pd.Timestamp("2025-01-10 16:00:00")
    schedule = pd.DataFrame({"ex_date": [ex_timestamp], "amount": [1.25]})

    adjusted_spot, effective_q = prepare_dividends(
        underlying=100.0,
        dividend_schedule=schedule,
        r=0.05,
        dividend_yield=None,
        valuation_date=valuation_timestamp,
        expiry=expiry_timestamp,
    )

    expected_tau = (ex_timestamp - valuation_timestamp).total_seconds() / (
        365.0 * 24.0 * 60.0 * 60.0
    )
    expected_spot = 100.0 - _discounted_cashflow(1.25, 0.05, expected_tau)

    assert adjusted_spot == pytest.approx(expected_spot)
    assert effective_q == pytest.approx(0.0)


def test_prepare_dividends_mixed_schedule_uses_only_pre_expiry_cashflow():
    """Early expiry should ignore later dividends even if still after valuation."""
    from oipd.pricing.utils import prepare_dividends

    valuation_timestamp = pd.Timestamp("2025-01-01 09:30:00")
    early_ex_timestamp = pd.Timestamp("2025-01-05 09:30:00")
    late_ex_timestamp = pd.Timestamp("2025-01-20 09:30:00")
    expiry_timestamp = pd.Timestamp("2025-01-10 16:00:00")
    schedule = pd.DataFrame(
        {"ex_date": [early_ex_timestamp, late_ex_timestamp], "amount": [1.0, 2.0]}
    )

    adjusted_spot, effective_q = prepare_dividends(
        underlying=100.0,
        dividend_schedule=schedule,
        r=0.05,
        dividend_yield=None,
        valuation_date=valuation_timestamp,
        expiry=expiry_timestamp,
    )

    early_tau = (early_ex_timestamp - valuation_timestamp).total_seconds() / (
        365.0 * 24.0 * 60.0 * 60.0
    )
    expected_spot = 100.0 - _discounted_cashflow(1.0, 0.05, early_tau)

    assert adjusted_spot == pytest.approx(expected_spot)
    assert effective_q == pytest.approx(0.0)


def test_prepare_dividends_includes_cashflow_at_valuation_and_expiry_boundaries():
    """Boundary semantics include ex-dates exactly at valuation and expiry."""
    from oipd.pricing.utils import prepare_dividends

    valuation_timestamp = pd.Timestamp("2025-01-01 09:30:00")
    expiry_timestamp = pd.Timestamp("2025-01-10 16:00:00")
    schedule = pd.DataFrame(
        {
            "ex_date": [valuation_timestamp, expiry_timestamp],
            "amount": [1.0, 2.0],
        }
    )

    adjusted_spot, effective_q = prepare_dividends(
        underlying=100.0,
        dividend_schedule=schedule,
        r=0.05,
        dividend_yield=None,
        valuation_date=valuation_timestamp,
        expiry=expiry_timestamp,
    )

    expiry_tau = (expiry_timestamp - valuation_timestamp).total_seconds() / (
        365.0 * 24.0 * 60.0 * 60.0
    )
    expected_spot = 100.0 - 1.0 - _discounted_cashflow(2.0, 0.05, expiry_tau)

    assert adjusted_spot == pytest.approx(expected_spot)
    assert effective_q == pytest.approx(0.0)


def test_resolve_market_normalizes_dividend_schedule_timestamps_and_amounts():
    """resolve_market should canonicalize dividend schedule schema without filtering."""
    inputs = MarketInputs(
        valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
        underlying_price=100.0,
        risk_free_rate=0.05,
        dividend_schedule=pd.DataFrame(
            {
                "ex_date": ["2025-01-01T12:00:00-05:00", "2025-01-02"],
                "amount": ["1.5", 2],
            }
        ),
    )

    resolved_market = resolve_market(inputs)
    schedule = resolved_market.dividend_schedule

    assert schedule is not None
    assert list(schedule["ex_date"]) == [
        pd.Timestamp("2025-01-01 17:00:00"),
        pd.Timestamp("2025-01-02 00:00:00"),
    ]
    assert schedule["amount"].to_list() == pytest.approx([1.5, 2.0])


@pytest.mark.parametrize("bad_amount", [None, np.nan])
def test_resolve_market_rejects_non_finite_dividend_amounts(bad_amount):
    """resolve_market should reject missing or non-finite dividend amounts."""
    inputs = MarketInputs(
        valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
        underlying_price=100.0,
        risk_free_rate=0.05,
        dividend_schedule=pd.DataFrame(
            {
                "ex_date": [pd.Timestamp("2025-01-02 00:00:00")],
                "amount": [bad_amount],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="dividend_schedule amount values must be finite",
    ):
        resolve_market(inputs)


def test_black76_round_trip():
    F = 100.0
    K = 100.0
    sigma = 0.2
    T = 0.5
    r = 0.01
    price = black76_call_price(F, K, sigma, T, r)
    est = black76_iv_brent_method(price, F, K, T, r)
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


def test_time_to_expiry_years_inputs_work_numerically():
    """Canonical year-fraction helper inputs should still work numerically."""
    S = 100.0
    strikes = np.arange(80, 121, 10)
    sigma = 0.2
    r = 0.01
    q = 0.0
    T = 30 / 365.0

    prices = black_scholes_call_price(S, strikes, sigma, T, r, q)
    df = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": prices,
            "bid": prices * 0.99,
            "ask": prices * 1.01,
        }
    )
    priced, _ = select_price_column(df, "last")

    iv_df = compute_iv(
        priced,
        S,
        time_to_expiry_years=T,
        risk_free_rate=r,
        solver_method="brent",
        pricing_engine="bs",
        dividend_yield=q,
    )

    assert not iv_df.empty
    assert np.isfinite(iv_df["iv"].to_numpy(dtype=float)).all()

    vol_curve = smooth_iv(
        "bspline",
        iv_df["strike"].to_numpy(),
        iv_df["iv"].to_numpy(),
    )
    strike_grid, call_prices = price_curve_from_iv(
        vol_curve,
        S,
        time_to_expiry_years=T,
        risk_free_rate=r,
        pricing_engine="bs",
        dividend_yield=q,
    )

    pdf_x, pdf_y = pdf_from_price_curve(
        strike_grid,
        call_prices,
        risk_free_rate=r,
        time_to_expiry_years=T,
    )
    assert pdf_x.shape == pdf_y.shape
    assert np.isfinite(pdf_y).all()
    assert np.all(pdf_y >= 0.0)


def test_days_to_expiry_kwargs_now_raise_type_error():
    """Removed day-count kwargs should fail at the public helper signatures."""
    S = 100.0
    strikes = np.arange(80, 121, 10)
    sigma = 0.2
    r = 0.01
    q = 0.0
    canonical_T = 30 / 365.0

    prices = black_scholes_call_price(S, strikes, sigma, canonical_T, r, q)
    df = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": prices,
            "bid": prices * 0.99,
            "ask": prices * 1.01,
        }
    )
    priced, _ = select_price_column(df, "last")

    canonical = compute_iv(
        priced,
        S,
        time_to_expiry_years=canonical_T,
        risk_free_rate=r,
        solver_method="brent",
        pricing_engine="bs",
        dividend_yield=q,
    )

    assert not canonical.empty

    with pytest.raises(TypeError, match="days_to_expiry"):
        compute_iv(
            priced,
            S,
            days_to_expiry=300,
            risk_free_rate=r,
            solver_method="brent",
            pricing_engine="bs",
            dividend_yield=q,
        )

    vol_curve = smooth_iv(
        "bspline",
        canonical["strike"].to_numpy(),
        canonical["iv"].to_numpy(),
    )

    with pytest.raises(TypeError, match="days_to_expiry"):
        price_curve_from_iv(
            vol_curve,
            S,
            days_to_expiry=300,
            risk_free_rate=r,
            pricing_engine="bs",
            dividend_yield=q,
        )

    strike_grid, call_prices = price_curve_from_iv(
        vol_curve,
        S,
        time_to_expiry_years=canonical_T,
        risk_free_rate=r,
        pricing_engine="bs",
        dividend_yield=q,
    )

    with pytest.raises(TypeError, match="days_to_expiry"):
        pdf_from_price_curve(
            strike_grid,
            call_prices,
            risk_free_rate=r,
            days_to_expiry=300,
        )


def test_missing_time_to_expiry_years_raises_clean_validation_error():
    """Low-level helpers should require canonical year-fraction maturity input."""
    S = 100.0
    strikes = np.arange(80, 121, 10)
    sigma = 0.2
    r = 0.01
    q = 0.0
    T = 30 / 365.0

    prices = black_scholes_call_price(S, strikes, sigma, T, r, q)
    df = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": prices,
            "bid": prices * 0.99,
            "ask": prices * 1.01,
        }
    )
    priced, _ = select_price_column(df, "last")

    with pytest.raises(ValueError, match="time_to_expiry_years"):
        compute_iv(
            priced,
            S,
            risk_free_rate=r,
            solver_method="brent",
            pricing_engine="bs",
            dividend_yield=q,
        )

    vol_curve = smooth_iv(
        "bspline",
        strikes,
        np.full_like(strikes, sigma, dtype=float),
    )

    with pytest.raises(InvalidInputError, match="time_to_expiry_years"):
        price_curve_from_iv(
            vol_curve,
            S,
            risk_free_rate=r,
            pricing_engine="bs",
            dividend_yield=q,
        )

    strike_grid, call_prices = price_curve_from_iv(
        vol_curve,
        S,
        time_to_expiry_years=T,
        risk_free_rate=r,
        pricing_engine="bs",
        dividend_yield=q,
    )

    with pytest.raises(InvalidInputError, match="time_to_expiry_years"):
        pdf_from_price_curve(
            strike_grid,
            call_prices,
            risk_free_rate=r,
        )
