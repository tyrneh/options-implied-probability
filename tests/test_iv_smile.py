import pandas as pd
from datetime import date

from oipd.estimator import RND, ModelParams
from oipd.market_inputs import MarketInputs


def _build_sample_chain() -> pd.DataFrame:
    """Create a representative options chain with last price quotes.

    Returns:
        pd.DataFrame: Options chain suitable for testing the smile helper.
    """
    return pd.DataFrame(
        {
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            "option_type": ["C", "C", "C", "C", "C"],
        }
    )


def _build_market_inputs() -> MarketInputs:
    """Construct market settings that align with the sample chain.

    Returns:
        MarketInputs: Market configuration aligned with the synthetic data.
    """
    return MarketInputs(
        valuation_date=date(2024, 1, 1),
        expiry_date=date(2024, 2, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
        risk_free_rate_mode="continuous",
    )


def test_iv_smile_default_grid():
    """Verify default grid evaluation for the fitted smile.

    Asserts that the helper produces a well-formed DataFrame with positive
    implied volatilities when no custom strike grid is supplied.
    """
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    smile = result.iv_smile()

    assert isinstance(smile, pd.DataFrame)
    assert {"strike", "fitted_iv"} <= set(smile.columns)
    assert (smile["fitted_iv"] > 0).all()


def test_iv_smile_custom_strikes_and_observed():
    """Check observed IV alignment on a user-provided strike grid.

    Ensures the helper attaches the observed implied volatility data when
    ``include_observed`` is requested.
    """
    chain = _build_sample_chain()
    market = _build_market_inputs()

    result = RND.from_dataframe(
        chain, market, model=ModelParams(price_method="last", pricing_engine="bs")
    )

    strikes = [90.0, 100.0, 110.0]
    smile = result.iv_smile(strikes, include_observed=True)

    assert list(smile["strike"]) == strikes
    assert "observed_iv" in smile.columns
    assert smile["observed_iv"].notna().all()
