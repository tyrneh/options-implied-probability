import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from oipd.interface.volatility import VolCurve, VolSurface
from oipd.market_inputs import MarketInputs
from oipd.pricing.black76 import black76_call_price
from oipd.pricing.black_scholes import black_scholes_call_price
from oipd.core.utils import resolve_risk_free_rate


# Fixtures
@pytest.fixture
def sample_market():
    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_yield=0.0,
    )


@pytest.fixture
def single_expiry_chain():
    """Create a synthetic option chain for a single expiry (30 days out).

    Includes both Calls (roughly convex, ~20% vol) and Puts (derived via
    Put-Call Parity) to ensure compatibility with Black-76 pricing constraints.
    """
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    expiry = pd.Timestamp("2025-01-31")

    # Approx 20% vol prices
    data = {
        "expiry": [expiry] * len(strikes),
        "strike": strikes,
        "bid": [20.5, 15.8, 11.0, 6.5, 2.8, 0.9, 0.3, 0.1, 0.05],
        "ask": [21.0, 16.2, 11.5, 7.0, 3.2, 1.2, 0.5, 0.2, 0.1],
        "last_price": [20.75, 16.0, 11.25, 6.75, 3.0, 1.05, 0.4, 0.15, 0.08],
        "option_type": ["call"] * len(strikes),
    }
    calls = pd.DataFrame(data)

    # Create Puts using approx parity: P = C - S + K*df
    S = 100.0
    r = 0.05
    t = 30.0 / 365.0
    df = np.exp(-r * t)

    puts = calls.copy()
    puts["option_type"] = "put"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def multi_expiry_chain():
    """Create a synthetic option chain with two expiries (30d and 60d).

    Each expiry includes Calls and corresponding Puts generated via parity.
    """
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    exp1 = pd.Timestamp("2025-01-31")  # 30d
    exp2 = pd.Timestamp("2025-03-02")  # 60d

    data1 = {
        "expiry": [exp1] * len(strikes),
        "strike": strikes,
        "bid": [20.5, 15.8, 11.0, 6.5, 2.8, 0.9, 0.3, 0.1, 0.05],
        "ask": [21.0, 16.2, 11.5, 7.0, 3.2, 1.2, 0.5, 0.2, 0.1],
        "last_price": [20.75, 16.0, 11.25, 6.75, 3.0, 1.05, 0.4, 0.15, 0.08],
        "option_type": ["call"] * len(strikes),
    }
    data2 = {
        "expiry": [exp2] * len(strikes),
        "strike": strikes,
        "bid": [
            21.5,
            17.0,
            12.0,
            7.5,
            3.8,
            1.5,
            0.8,
            0.4,
            0.2,
        ],  # Higher for longer time
        "ask": [22.0, 17.5, 12.5, 8.0, 4.2, 1.8, 1.2, 0.6, 0.3],
        "last_price": [21.75, 17.25, 12.25, 7.75, 4.0, 1.65, 1.0, 0.5, 0.25],
        "option_type": ["call"] * len(strikes),
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Generate Puts for parity (S=100, r=0.05)
    S = 100.0
    r = 0.05

    def add_puts(df_calls, t_days):
        df_puts = df_calls.copy()
        df_puts["option_type"] = "put"
        t = t_days / 365.0
        df_p = np.exp(-r * t)

        # P = C - S + K*df
        # Use abs() to ensure no negative garbage from fake data
        df_puts["last_price"] = (
            df_calls["last_price"] - S + df_calls["strike"] * df_p
        ).abs()
        df_puts["bid"] = (df_calls["bid"] - S + df_calls["strike"] * df_p).abs()
        df_puts["ask"] = (df_calls["ask"] - S + df_calls["strike"] * df_p).abs()
        return df_puts

    df1_puts = add_puts(df1, 30)
    df2_puts = add_puts(df2, 60)

    return pd.concat([df1, df1_puts, df2, df2_puts], ignore_index=True)


@pytest.fixture
def bs_dividend_schedule_curve():
    """Discrete schedule with one live and one post-expiry cash dividend."""
    return pd.DataFrame(
        {
            "ex_date": [
                pd.Timestamp("2025-01-15 00:00:00"),
                pd.Timestamp("2025-02-15 00:00:00"),
            ],
            "amount": [1.0, 1.5],
        }
    )


@pytest.fixture
def bs_dividend_schedule_early_only():
    """Early-expiry schedule retaining only the first dividend."""
    return pd.DataFrame(
        {
            "ex_date": [pd.Timestamp("2025-01-15 00:00:00")],
            "amount": [1.0],
        }
    )


def test_vol_curve_price_black76(sample_market, single_expiry_chain):
    """Test VolCurve.price() using default Black76 engine."""
    vc = VolCurve(pricing_engine="black76", method="svi")
    vc.fit(single_expiry_chain, sample_market)

    strikes = [95, 105]

    # 1. Calculate via method
    prices = vc.price(strikes, call_or_put="call")

    # 2. Manual Verification
    # Recover state
    F = vc.forward_price
    expiry_date = single_expiry_chain["expiry"].iloc[0].date()
    t = (expiry_date - sample_market.valuation_calendar_date).days / 365.0
    r = resolve_risk_free_rate(
        sample_market.risk_free_rate,
        sample_market.risk_free_rate_mode,
        t,
    )
    sigma = vc(strikes)

    expected = black76_call_price(F, np.array(strikes), sigma, t, r)

    np.testing.assert_allclose(prices, expected, rtol=1e-5)


def test_vol_curve_price_bs(sample_market, single_expiry_chain):
    """Test VolCurve.price() using Black-Scholes engine."""
    vc = VolCurve(pricing_engine="bs", method="svi")
    vc.fit(single_expiry_chain, sample_market)

    strikes = [100]

    # 1. Calculate via method
    price_method = vc.price(strikes)

    # 2. Manual Verification
    S = sample_market.underlying_price
    q = 0.0
    expiry_date = single_expiry_chain["expiry"].iloc[0].date()
    t = (expiry_date - sample_market.valuation_calendar_date).days / 365.0
    r = resolve_risk_free_rate(
        sample_market.risk_free_rate,
        sample_market.risk_free_rate_mode,
        t,
    )
    sigma = vc(strikes)

    expected = black_scholes_call_price(S, np.array(strikes), sigma, t, r, q)

    np.testing.assert_allclose(price_method, expected, rtol=1e-5)


def test_vol_curve_parity(sample_market, single_expiry_chain):
    """Test Put-Call Parity consistency in VolCurve.price()."""
    vc = VolCurve(pricing_engine="black76")
    vc.fit(single_expiry_chain, sample_market)

    K = 100.0
    C = vc.price([K], call_or_put="call")[0]
    P = vc.price([K], call_or_put="put")[0]

    # Parity: C - P = D * (F - K)
    F = vc.forward_price
    t = 30.0 / 365.0
    r = resolve_risk_free_rate(
        sample_market.risk_free_rate,
        sample_market.risk_free_rate_mode,
        t,
    )
    df = np.exp(-r * t)

    lhs = C - P
    rhs = df * (F - K)

    np.testing.assert_allclose(lhs, rhs, atol=1e-4)


def test_vol_surface_price_interpolated(sample_market, multi_expiry_chain):
    """Test VolSurface.price() at an interpolated time t."""
    vs = VolSurface()
    vs.fit(multi_expiry_chain, sample_market)

    # Interpolate at 45 days (between 30 and 60)
    t_interp_days = 45
    t_interp_years = 45 / 365.0
    strikes = [100.0]

    # 1. Price via fast method
    prices = vs.price(strikes, t=t_interp_years)

    # 2. Manual check via public helpers
    F_interp = vs.forward_price(t_interp_years)
    sigma_interp = vs.implied_vol(100.0, t_interp_years)

    r = resolve_risk_free_rate(
        sample_market.risk_free_rate,
        sample_market.risk_free_rate_mode,
        t_interp_years,
    )
    expected = black76_call_price(
        F_interp,
        np.array(strikes),
        np.array([sigma_interp]),
        t_interp_years,
        r,
    )

    np.testing.assert_allclose(prices, expected, rtol=1e-8)


def test_vol_surface_date_input(sample_market, multi_expiry_chain):
    """Test that VolSurface.price accepts date strings."""
    vs = VolSurface()
    vs.fit(multi_expiry_chain, sample_market)

    target_date = "2025-02-15"  # 45 days from Jan 1
    t_years = 45 / 365.0

    p1 = vs.price([100], t=target_date)
    p2 = vs.price([100], t=t_years)

    np.testing.assert_allclose(p1, p2, rtol=1e-8)


def test_pricing_bs_missing_dividend_error(single_expiry_chain, multi_expiry_chain):
    """Test that ValueError is raised if dividend_yield is missing for BS pricing."""
    # Create market with implicit None dividend_yield (or explicitly None if supported)
    # MarketInputs defaults use None for dividend_yield if not provided?
    # Let's verify MarketInputs constructor or just pass None explicitly if allowed.
    # Assuming MarketInputs supports explicit None or omits it.

    # Based on usage, MarketInputs(..., dividend_yield=None) is valid.
    market_incomplete = MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_yield=None,  # Explicitly missing
    )

    vc = VolCurve(pricing_engine="bs", method="svi")
    vc.fit(single_expiry_chain, market_incomplete)

    with pytest.raises(ValueError, match="Dividend yield.*required"):
        vc.price([100])

    vs = VolSurface(pricing_engine="bs")
    vs.fit(multi_expiry_chain, market_incomplete)

    with pytest.raises(ValueError, match="Dividend yield.*required"):
        vs.price([100], t=0.1)


def test_vol_curve_price_bs_with_dividend_schedule_matches_adjusted_spot(
    single_expiry_chain, bs_dividend_schedule_curve
):
    """BS curve pricing should use schedule-adjusted spot instead of requiring q."""
    market_schedule = MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_schedule=bs_dividend_schedule_curve,
    )
    vc = VolCurve(pricing_engine="bs", method="svi")
    vc.fit(single_expiry_chain, market_schedule)

    strike = np.array([100.0])
    price_method = vc.price(strike)

    expiry_timestamp = pd.Timestamp(single_expiry_chain["expiry"].iloc[0])
    valuation_timestamp = pd.Timestamp(market_schedule.valuation_date)
    t = (expiry_timestamp - valuation_timestamp).total_seconds() / (
        365.0 * 24.0 * 60.0 * 60.0
    )
    r = resolve_risk_free_rate(
        market_schedule.risk_free_rate,
        market_schedule.risk_free_rate_mode,
        t,
    )
    tau_div = (
        pd.Timestamp("2025-01-15 00:00:00") - valuation_timestamp
    ).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)
    adjusted_spot = 100.0 - np.exp(-r * tau_div)
    expected = black_scholes_call_price(
        adjusted_spot,
        strike,
        vc(strike),
        t,
        r,
        0.0,
    )

    np.testing.assert_allclose(price_method, expected, rtol=1e-6)


def test_vol_surface_price_bs_with_dividend_schedule_respects_early_vs_late_window(
    multi_expiry_chain,
    bs_dividend_schedule_curve,
    bs_dividend_schedule_early_only,
):
    """Early expiry should ignore the later dividend while later expiry includes it."""
    market_full = MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_schedule=bs_dividend_schedule_curve,
    )
    market_early_only = MarketInputs(
        valuation_date=date(2025, 1, 1),
        risk_free_rate=0.05,
        underlying_price=100.0,
        dividend_schedule=bs_dividend_schedule_early_only,
    )

    surface_full = VolSurface(pricing_engine="bs").fit(multi_expiry_chain, market_full)
    surface_early_only = VolSurface(pricing_engine="bs").fit(
        multi_expiry_chain, market_early_only
    )

    early_expiry = pd.Timestamp("2025-01-31 00:00:00")
    late_expiry = pd.Timestamp("2025-03-02 00:00:00")
    early_t = 30 / 365.0
    late_t = 60 / 365.0

    price_early_date = surface_full.price([100.0], t=early_expiry)
    price_early_float = surface_full.price([100.0], t=early_t)
    price_early_only = surface_early_only.price([100.0], t=early_t)
    price_late_full = surface_full.price([100.0], t=late_expiry)
    price_late_early_only = surface_early_only.price([100.0], t=late_t)

    np.testing.assert_allclose(price_early_date, price_early_float, rtol=1e-8)
    np.testing.assert_allclose(price_early_float, price_early_only, rtol=1e-8)
    assert price_late_full[0] != pytest.approx(price_late_early_only[0])
