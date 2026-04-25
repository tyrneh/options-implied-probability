"""Tests for put-call parity preprocessing functionality."""

import warnings

import numpy as np
import pandas as pd
import pytest

from oipd.core.data_processing.parity import (
    apply_put_call_parity_to_quotes,
    detect_parity_opportunity,
    infer_forward_from_atm,
    infer_forward_from_put_call_parity,
    preprocess_with_parity,
)


def _bid_ask_only_options_data() -> pd.DataFrame:
    """Create bid/ask-only row-format options with coherent parity mids.

    Returns:
        Option quotes for five same-strike call-put pairs. The DataFrame
        intentionally omits ``last_price`` so tests can verify bid/ask-only support
        without manufacturing last-trade prices.
    """
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    forward_price = 101.0
    time_value = 2.0
    half_spread = 0.25
    rows = []

    for strike in strikes:
        call_mid = max(forward_price - strike, 0.0) + time_value
        put_mid = max(strike - forward_price, 0.0) + time_value
        rows.append(
            {
                "strike": strike,
                "option_type": "C",
                "bid": call_mid - half_spread,
                "ask": call_mid + half_spread,
            }
        )
        rows.append(
            {
                "strike": strike,
                "option_type": "P",
                "bid": put_mid - half_spread,
                "ask": put_mid + half_spread,
            }
        )

    return pd.DataFrame(rows)


def _robust_forward_options_data() -> pd.DataFrame:
    """Create five call-put pairs with one ATM forward outlier.

    Returns:
        Row-format option quotes where four same-strike pairs imply ``F=100`` and
        the ATM pair implies a stale forward of ``F=150``.
    """
    pair_prices = {
        90.0: (15.0, 5.0),
        95.0: (10.0, 5.0),
        100.0: (52.0, 2.0),
        105.0: (5.0, 10.0),
        110.0: (5.0, 15.0),
    }
    rows = []

    for strike, (call_price, put_price) in pair_prices.items():
        rows.append(
            {
                "strike": strike,
                "last_price": call_price,
                "option_type": "C",
            }
        )
        rows.append(
            {
                "strike": strike,
                "last_price": put_price,
                "option_type": "P",
            }
        )

    return pd.DataFrame(
        rows,
    )


def _long_call_put_options_data() -> pd.DataFrame:
    """Create long-form call/put quotes with a stable forward.

    Returns:
        Five call-put row pairs, all implying ``F=100`` when the discount factor
        is one.
    """
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    forward_price = 100.0
    time_value = 2.0
    rows = []

    for strike in strikes:
        rows.append(
            {
                "strike": strike,
                "option_type": "C",
                "last_price": max(forward_price - strike, 0.0) + time_value,
            }
        )
        rows.append(
            {
                "strike": strike,
                "option_type": "P",
                "last_price": max(strike - forward_price, 0.0) + time_value,
            }
        )

    return pd.DataFrame(rows)


def _two_valid_pair_options_data() -> pd.DataFrame:
    """Create five-strike long-form data with exactly two valid parity pairs.

    Returns:
        Long-form option quotes where two strikes imply forwards of ``100`` and
        ``103`` while the remaining strikes are missing usable pair prices.
    """
    return _long_rows_from_pair_prices(
        strikes=[90.0, 95.0, 100.0, 105.0, 110.0],
        call_prices=[np.nan, 7.0, np.nan, 4.0, np.nan],
        put_prices=[np.nan, 2.0, np.nan, 6.0, np.nan],
    )


def _single_valid_pair_options_data() -> pd.DataFrame:
    """Create five-strike long-form data with exactly one valid parity pair.

    Returns:
        Long-form option quotes where one strike implies ``F=102`` and all other
        strikes lack usable paired prices.
    """
    return _long_rows_from_pair_prices(
        strikes=[90.0, 95.0, 100.0, 105.0, 110.0],
        call_prices=[np.nan, np.nan, 3.0, np.nan, np.nan],
        put_prices=[np.nan, np.nan, 1.0, np.nan, np.nan],
    )


def _zero_valid_pair_options_data() -> pd.DataFrame:
    """Create long-form data with candidate pairs but no valid forward estimates.

    Returns:
        Long-form option quotes where all five candidate pairs imply non-positive
        forwards, so direct forward inference must fail.
    """
    return _long_rows_from_pair_prices(
        strikes=[90.0, 95.0, 100.0, 105.0, 110.0],
        call_prices=[1.0, 1.0, 1.0, 1.0, 1.0],
        put_prices=[200.0, 200.0, 200.0, 200.0, 200.0],
    )


def _long_rows_from_pair_prices(
    strikes: list[float],
    call_prices: list[float],
    put_prices: list[float],
) -> pd.DataFrame:
    """Create long-form call-put rows from per-strike pair prices.

    Args:
        strikes: Strike values to include in the option chain.
        call_prices: Call prices aligned to ``strikes``.
        put_prices: Put prices aligned to ``strikes``.

    Returns:
        Long-form option quotes with two rows per strike.
    """
    rows = []
    for strike, call_price, put_price in zip(strikes, call_prices, put_prices):
        rows.append(
            {
                "strike": strike,
                "option_type": "C",
                "last_price": call_price,
            }
        )
        rows.append(
            {
                "strike": strike,
                "option_type": "P",
                "last_price": put_price,
            }
        )
    return pd.DataFrame(rows)


def _forward_chain(
    strikes: list[float],
    forward_by_strike: dict[float, float] | None = None,
    default_forward: float = 100.0,
) -> pd.DataFrame:
    """Create positive long-form prices with chosen pair-implied forwards.

    Args:
        strikes: Strike values to include in the option chain.
        forward_by_strike: Optional mapping from strike to desired pair-implied
            forward price.
        default_forward: Pair-implied forward used for strikes not in the mapping.

    Returns:
        Long-form option quotes where each strike has positive call and put prices.
    """
    rows = []
    forward_by_strike = forward_by_strike or {}

    for strike in strikes:
        forward = forward_by_strike.get(float(strike), default_forward)
        time_value = abs(forward - strike) + 2.0
        rows.append(
            {
                "strike": float(strike),
                "option_type": "C",
                "last_price": max(forward - strike, 0.0) + time_value,
            }
        )
        rows.append(
            {
                "strike": float(strike),
                "option_type": "P",
                "last_price": max(strike - forward, 0.0) + time_value,
            }
        )

    return pd.DataFrame(rows)


def _row_quote_pair(
    *,
    strike: float = 100.0,
    call_mid: float | None = None,
    put_mid: float | None = None,
    relative_spread: float | None = None,
    call_last: float | None = None,
    put_last: float | None = None,
    call_volume: float | None = None,
    put_volume: float | None = None,
    uppercase_volume: float | None = None,
) -> pd.DataFrame:
    """Create one row-format call/put pair for quote-source tests.

    Args:
        strike: Shared strike for the call and put.
        call_mid: Optional bid/ask midpoint for the call.
        put_mid: Optional bid/ask midpoint for the put.
        relative_spread: Optional bid/ask relative spread around each midpoint.
        call_last: Optional call ``last_price``.
        put_last: Optional put ``last_price``.
        call_volume: Optional lowercase call volume.
        put_volume: Optional lowercase put volume.
        uppercase_volume: Optional legacy uppercase ``Volume`` for both legs.

    Returns:
        Two-row option quote table containing one call and one put.
    """
    rows = [
        {"strike": strike, "option_type": "C"},
        {"strike": strike, "option_type": "P"},
    ]

    if call_mid is not None and put_mid is not None and relative_spread is not None:
        for row, midpoint in zip(rows, [call_mid, put_mid]):
            half_spread = midpoint * relative_spread / 2.0
            row["bid"] = midpoint - half_spread
            row["ask"] = midpoint + half_spread

    if call_last is not None and put_last is not None:
        rows[0]["last_price"] = call_last
        rows[1]["last_price"] = put_last

    if call_volume is not None or put_volume is not None:
        rows[0]["volume"] = call_volume
        rows[1]["volume"] = put_volume

    if uppercase_volume is not None:
        rows[0]["Volume"] = uppercase_volume
        rows[1]["Volume"] = uppercase_volume

    return pd.DataFrame(rows)


def _assert_parity_forward_requirement_message(message: str) -> None:
    """Assert parity-forward failures do not suggest legacy pricing fallbacks."""
    assert "usable same-strike call/put pairs" in message
    assert "parity-forward inference" in message
    assert "Black-Scholes" not in message
    assert "dividend" not in message.lower()


class TestDetectParityOpportunity:
    """Test detection of when parity preprocessing would be beneficial."""

    def test_detect_option_type_format(self):
        """Test detection with option_type format."""
        df = pd.DataFrame(
            {
                "strike": [95, 95, 100, 100, 105],
                "last_price": [2.5, 1.2, 1.8, 2.1, 0.5],
                "option_type": ["C", "P", "C", "P", "C"],
            }
        )

        assert detect_parity_opportunity(df) is True

    def test_detect_no_opportunity_calls_only(self):
        """Test that calls-only data returns False."""
        df = pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "last_price": [2.5, 1.8, 0.5],
                "option_type": ["C", "C", "C"],
            }
        )

        assert detect_parity_opportunity(df) is False

    def test_detect_requires_pair(self):
        """Parity detection requires at least one same-strike pair."""
        df = pd.DataFrame(
            {
                "strike": [95, 100],
                "last_price": [2.5, 1.2],
                "option_type": ["C", "P"],
            }
        )

        assert detect_parity_opportunity(df) is False

    def test_detect_single_pair(self):
        """Detection should succeed with a single call-put pair."""
        df = pd.DataFrame(
            {
                "strike": [100, 100],
                "last_price": [2.5, 1.2],
                "option_type": ["C", "P"],
            }
        )

        assert detect_parity_opportunity(df) is True

    def test_detect_bid_ask_only_row_format(self):
        """Detection should accept row-format pairs priced only by bid/ask mids."""
        df = _bid_ask_only_options_data()

        assert "last_price" not in df.columns
        assert detect_parity_opportunity(df) is True

    @pytest.mark.parametrize(
        "bid,ask,last_price",
        [
            ([5.0, np.nan], [7.0, np.nan], [np.nan, 1.5]),
            ([np.nan, 2.0], [np.nan, 4.0], [4.0, np.nan]),
        ],
    )
    def test_detect_rejects_mixed_source_only_pair(self, bid, ask, last_price):
        """Detection should reject pairs that require mixing mids and last prices."""
        df = pd.DataFrame(
            {
                "strike": [100, 100],
                "option_type": ["C", "P"],
                "bid": bid,
                "ask": ask,
                "last_price": last_price,
            }
        )

        assert detect_parity_opportunity(df) is False


class TestInferForwardFromATM:
    """Test forward price inference from ATM call-put pairs."""

    @pytest.fixture
    def sample_options_data(self):
        """Create sample options data with known forward price."""
        # Create data where forward should be ~100
        strikes = [95, 100, 105]
        spot = 100
        forward = 100.5  # Slightly above spot due to interest rates
        discount_factor = 0.99

        data = []
        for strike in strikes:
            # Calculate theoretical call and put prices
            call_intrinsic = max(0, discount_factor * (forward - strike))
            put_intrinsic = max(0, discount_factor * (strike - forward))

            # Add some time value
            call_price = call_intrinsic + 1.0
            put_price = put_intrinsic + 1.0

            data.append(
                {
                    "strike": strike,
                    "last_price": call_price,
                    "option_type": "C",
                }
            )
            data.append({"strike": strike, "last_price": put_price, "option_type": "P"})

        return pd.DataFrame(data)

    def test_infer_forward_option_type_format(self, sample_options_data):
        """Test forward inference with option_type format."""
        spot_price = 100.0
        discount_factor = 0.99

        forward = infer_forward_from_atm(
            sample_options_data, spot_price, discount_factor
        )

        # Should be close to our expected forward price of 100.5
        assert 99.5 <= forward <= 101.5  # Allow some tolerance

    def test_infer_forward_from_put_call_parity_uses_mids_before_last_price(self):
        """Forward inference should prefer same-pair bid/ask mids over stale last."""
        df = pd.DataFrame(
            {
                "strike": [100, 100],
                "option_type": ["C", "P"],
                "bid": [5.5, 2.75],
                "ask": [6.5, 3.25],
                "last_price": [20.0, 1.0],
            }
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            forward = infer_forward_from_put_call_parity(df, 100.0, 1.0)

        assert isinstance(forward, float)
        assert forward == pytest.approx(103.0)

    def test_infer_forward_from_put_call_parity_falls_back_to_last_when_mids_unavailable(
        self,
    ):
        """Forward inference should fall back to both last prices, not mixed sources."""
        df = pd.DataFrame(
            {
                "strike": [100, 100],
                "option_type": ["C", "P"],
                "bid": [5.0, np.nan],
                "ask": [7.0, np.nan],
                "last_price": [4.0, 1.5],
            }
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            forward = infer_forward_from_put_call_parity(df, 100.0, 1.0)

        assert isinstance(forward, float)
        assert forward == pytest.approx(102.5)

    @pytest.mark.parametrize("discount_factor", [0.0, -1.0, np.nan, np.inf])
    def test_infer_forward_from_put_call_parity_rejects_invalid_discount_factor(
        self, discount_factor
    ):
        """Forward inference should reject non-positive or non-finite discounts."""
        df = pd.DataFrame(
            {
                "strike": [100],
                "call_price": [2.0],
                "put_price": [1.0],
            }
        )

        with pytest.raises(
            ValueError, match="discount_factor must be finite and strictly positive"
        ):
            infer_forward_from_put_call_parity(df, 100.0, discount_factor)

    def test_infer_forward_from_put_call_parity_uses_robust_median_for_three_plus_pairs(
        self,
    ):
        """Forward inference should reject one stale pair among five valid pairs."""
        df = _robust_forward_options_data()

        forward = infer_forward_from_put_call_parity(df, 100.0, 1.0)

        assert forward == pytest.approx(100.0)

    def test_infer_forward_from_put_call_parity_two_pairs_uses_median_and_warns(self):
        """Two-pair forward inference should average via median and warn once."""
        df = _two_valid_pair_options_data()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            forward = infer_forward_from_put_call_parity(df, 100.0, 1.0)

        assert forward == pytest.approx(101.5)
        assert len(captured_warnings) == 1
        warning_message = str(captured_warnings[0].message)
        assert "2 valid pairs" in warning_message
        assert "low_two_pairs" in warning_message

    def test_infer_forward_from_put_call_parity_zero_valid_pairs_raises(self):
        """Direct forward inference should fail when no candidate pair is valid."""
        df = _zero_valid_pair_options_data()

        with pytest.raises(
            ValueError,
            match="No valid put-call parity pairs.*0 valid pairs",
        ):
            infer_forward_from_put_call_parity(df, 100.0, 1.0)

    def test_infer_forward_from_atm_wraps_new_helper(self, monkeypatch):
        """Legacy forward inference should delegate to the new public helper."""
        import oipd.core.data_processing.parity as parity_module

        df = pd.DataFrame(
            {
                "strike": [100],
                "call_price": [2.0],
                "put_price": [1.0],
            }
        )
        observed_args = {}

        def fake_infer_forward_from_put_call_parity(
            options_df,
            underlying_price,
            discount_factor,
            **kwargs,
        ):
            """Return a sentinel forward and capture delegated arguments.

            Args:
                options_df: Option quote table passed by the wrapper.
                underlying_price: Underlying price passed by the wrapper.
                discount_factor: Discount factor passed by the wrapper.
                **kwargs: Keyword-only parity configuration.

            Returns:
                Sentinel forward price used to prove delegation.
            """
            observed_args["options_df"] = options_df
            observed_args["underlying_price"] = underlying_price
            observed_args["discount_factor"] = discount_factor
            observed_args["kwargs"] = kwargs
            return 123.45

        monkeypatch.setattr(
            parity_module,
            "infer_forward_from_put_call_parity",
            fake_infer_forward_from_put_call_parity,
        )

        forward = parity_module.infer_forward_from_atm(df, 101.0, 0.99)

        assert forward == pytest.approx(123.45)
        assert observed_args["options_df"] is df
        assert observed_args["underlying_price"] == pytest.approx(101.0)
        assert observed_args["discount_factor"] == pytest.approx(0.99)
        assert observed_args["kwargs"]["max_forward_pairs"] == 5

    def test_infer_forward_no_pairs(self):
        """Test that error is raised when no call-put pairs exist."""
        df = pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "last_price": [2.5, 1.8, 0.5],
                "option_type": ["C", "C", "C"],  # No puts
            }
        )

        with pytest.raises(ValueError, match="No strikes found with both call and put"):
            infer_forward_from_atm(df, 100.0, 0.99)


class TestApplyPutCallParity:
    """Test the core put-call parity application logic."""

    @pytest.fixture
    def balanced_options_data(self):
        """Create balanced options data for testing."""
        return pd.DataFrame(
            {
                "strike": [95, 95, 100, 100, 105, 105],
                "last_price": [6.0, 1.0, 2.5, 2.0, 0.5, 4.5],
                "option_type": ["C", "P", "C", "P", "C", "P"],
            }
        )

    def test_parity_basic_application(self, balanced_options_data):
        """Test basic put-call parity application."""
        forward_price = 100.0
        discount_factor = 0.99

        result = apply_put_call_parity_to_quotes(
            balanced_options_data, forward_price, discount_factor
        )

        # Should have one row per strike
        assert len(result) == 3
        assert "last_price" in result.columns
        assert "source" in result.columns

        # Above forward (105): should use original call
        call_105 = result[result["strike"] == 105]
        assert len(call_105) == 1
        assert call_105["source"].iloc[0] == "call"
        assert call_105["last_price"].iloc[0] == 0.5  # Original call price

        # At/below forward (95, 100): should use synthetic calls from puts
        put_95 = result[result["strike"] == 95]
        assert put_95["source"].iloc[0] == "put_converted"

    def test_parity_boundary_condition(self, balanced_options_data):
        """Test boundary condition where K = F exactly."""
        forward_price = 100.0  # Exactly at 100 strike
        discount_factor = 0.99

        result = apply_put_call_parity_to_quotes(
            balanced_options_data, forward_price, discount_factor
        )

        # Strike 100 should be treated as K <= F (use put)
        strike_100 = result[result["strike"] == 100]
        assert strike_100["source"].iloc[0] == "put_converted"

    def test_parity_preserves_lowercase_volume(self, balanced_options_data):
        """Test parity output retains volume for downstream SVI weighting."""
        forward_price = 100.0
        discount_factor = 0.99
        balanced_options_data = balanced_options_data.copy()
        balanced_options_data["volume"] = [10, 20, 30, 40, 50, 60]

        result = apply_put_call_parity_to_quotes(
            balanced_options_data, forward_price, discount_factor
        )

        assert "volume" in result.columns
        assert "Volume" not in result.columns
        assert result.loc[result["strike"] == 95, "volume"].iloc[0] == 20
        assert result.loc[result["strike"] == 105, "volume"].iloc[0] == 50

    def test_parity_ignores_legacy_capitalized_volume(self, balanced_options_data):
        """Test legacy ``Volume`` input is not treated as canonical volume."""
        forward_price = 100.0
        discount_factor = 0.99
        balanced_options_data = balanced_options_data.copy()
        balanced_options_data["Volume"] = [10, 20, 30, 40, 50, 60]

        result = apply_put_call_parity_to_quotes(
            balanced_options_data, forward_price, discount_factor
        )

        assert "Volume" not in result.columns
        assert "volume" in result.columns
        assert result["volume"].isna().all()

    def test_parity_accepts_bid_ask_only_row_format(self):
        """Parity conversion should preserve bid/ask-derived prices without last."""
        df = _bid_ask_only_options_data()

        result = apply_put_call_parity_to_quotes(
            df, forward_price=101.0, discount_factor=1.0
        )

        assert len(result) == 5
        assert set(result["strike"]) == {90.0, 95.0, 100.0, 105.0, 110.0}
        assert result["F_used"].eq(101.0).all()
        assert result["mid"].notna().all()
        assert result["bid"].notna().all()
        assert result["ask"].notna().all()
        assert "last_price" in result.columns
        assert result["last_price"].isna().all()
        assert result.loc[result["strike"] == 100.0, "source"].iloc[0] == (
            "put_converted"
        )
        assert result.loc[result["strike"] == 105.0, "source"].iloc[0] == "call"


class TestPreprocessWithParity:
    """Test the main preprocessing entry point."""

    def test_preprocess_with_parity_beneficial(self):
        """Test preprocessing when parity is beneficial."""
        df = pd.DataFrame(
            {
                "strike": [95, 95, 100, 100, 105, 105],
                "last_price": [6.0, 1.0, 2.5, 2.0, 0.5, 4.5],
                "option_type": ["C", "P", "C", "P", "C", "P"],
            }
        )

        spot_price = 100.0
        discount_factor = 0.99

        result = preprocess_with_parity(df, spot_price, discount_factor)

        # Should return cleaned data with 'last_price' column
        assert "last_price" in result.columns
        assert len(result) == 3  # One row per strike

    def test_preprocess_no_benefit_calls_only(self):
        """Test preprocessing with calls-only data (no benefit)."""
        df = pd.DataFrame({"strike": [95, 100, 105], "last_price": [2.5, 1.8, 0.5]})

        spot_price = 100.0
        discount_factor = 0.99

        result = preprocess_with_parity(df, spot_price, discount_factor)

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_preprocess_accepts_bid_ask_only_row_format(self):
        """Preprocessing should parity-adjust bid/ask-only row-format quotes."""
        df = _bid_ask_only_options_data()

        with warnings.catch_warnings(record=True) as captured_warnings:
            result = preprocess_with_parity(df, 100.0, 1.0)

        assert not any(
            "Put-call parity preprocessing failed" in str(warning.message)
            for warning in captured_warnings
        )
        assert len(result) == 5
        assert result["F_used"].eq(101.0).all()
        assert result["mid"].notna().all()
        assert result["bid"].notna().all()
        assert result["ask"].notna().all()
        assert result["last_price"].isna().all()

    def test_preprocess_attaches_robust_parity_report(self):
        """Preprocessing should attach robust parity diagnostics to attrs."""
        df = _robust_forward_options_data()

        result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert result["F_used"].eq(100.0).all()
        assert len(result) == 5
        assert report["confidence"] == "robust"
        assert report["valid_pair_count"] >= 3
        assert report["outlier_count"] == 1
        assert report["pairs_used_count"] == 4
        assert 100.0 in report["outlier_strikes"]
        outlier_pairs = [pair for pair in report["pairs"] if pair["outlier"]]
        assert len(outlier_pairs) == 1
        assert outlier_pairs[0]["strike"] == 100.0
        assert outlier_pairs[0]["forward_price"] == pytest.approx(150.0)

    def test_preprocess_long_call_put_prices_attaches_parity_report(self):
        """Long-form preprocessing should convert quotes and attach diagnostics."""
        df = _long_call_put_options_data()

        with warnings.catch_warnings(record=True) as captured_warnings:
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert not any(
            "Put-call parity preprocessing failed" in str(warning.message)
            for warning in captured_warnings
        )
        assert len(result) == 5
        assert result["F_used"].eq(100.0).all()
        assert result["last_price"].notna().all()
        assert report["confidence"] == "robust"
        assert report["candidate_pair_count"] == 5
        assert report["valid_pair_count"] == 5
        assert report["outlier_count"] == 0
        assert report["pairs_used_count"] == 5

    def test_preprocess_two_valid_pairs_warns_once_and_reports_low_confidence(self):
        """Two-pair preprocessing should warn once and report low confidence."""
        df = _two_valid_pair_options_data()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert result["F_used"].eq(101.5).all()
        assert len(captured_warnings) == 1
        warning_message = str(captured_warnings[0].message)
        assert "2 valid pairs" in warning_message
        assert "low_two_pairs" in warning_message
        assert report["confidence"] == "low_two_pairs"
        assert report["candidate_pair_count"] == 5
        assert report["valid_pair_count"] == 2
        assert report["pairs_used_count"] == 2
        assert report["outlier_count"] == 0
        invalid_pairs = [pair for pair in report["pairs"] if not pair["valid"]]
        assert len(invalid_pairs) == 3
        assert {pair["excluded_reason"] for pair in invalid_pairs} == {
            "invalid_last_price_pair"
        }
        pair_estimates = sorted(
            pair["forward_price"] for pair in report["pairs"] if pair["valid"]
        )
        assert pair_estimates == [100.0, 103.0]

    def test_preprocess_single_valid_pair_warns_once_and_reports_low_confidence(self):
        """Single-pair preprocessing should warn once and use the only estimate."""
        df = _single_valid_pair_options_data()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert result["F_used"].eq(102.0).all()
        assert len(captured_warnings) == 1
        warning_message = str(captured_warnings[0].message)
        assert "1 valid pair" in warning_message
        assert "low_single_pair" in warning_message
        assert report["confidence"] == "low_single_pair"
        assert report["candidate_pair_count"] == 5
        assert report["valid_pair_count"] == 1
        assert report["pairs_used_count"] == 1
        assert report["outlier_count"] == 0
        invalid_pairs = [pair for pair in report["pairs"] if not pair["valid"]]
        assert len(invalid_pairs) == 4
        assert {pair["excluded_reason"] for pair in invalid_pairs} == {
            "invalid_last_price_pair"
        }
        pair_estimates = [
            pair["forward_price"] for pair in report["pairs"] if pair["valid"]
        ]
        assert pair_estimates == [102.0]

    def test_preprocess_zero_valid_pairs_raises_clear_error(self):
        """Zero-valid-pair preprocessing should fail instead of fallback silently."""
        df = _zero_valid_pair_options_data()

        with pytest.raises(ValueError) as exc_info:
            preprocess_with_parity(df, 100.0, 1.0)

        message = str(exc_info.value)
        assert "Put-call parity preprocessing could not infer a forward" in message
        _assert_parity_forward_requirement_message(message)

    def test_preprocess_fallback_on_error(self, monkeypatch):
        """Test that preprocessing falls back gracefully on errors."""
        df = pd.DataFrame(
            {
                "strike": [100, 100],
                "last_price": [2.0, 1.5],
                "option_type": ["C", "P"],
            }
        )

        spot_price = 100.0
        discount_factor = 0.99

        def boom(*args, **kwargs):
            """Raise an error in place of forward inference.

            Args:
                *args: Positional arguments accepted for monkeypatch compatibility.
                **kwargs: Keyword arguments accepted for monkeypatch compatibility.

            Raises:
                ValueError: Always raised to exercise fallback behavior.
            """
            raise ValueError("boom")

        monkeypatch.setattr(
            "oipd.core.data_processing.parity._infer_forward_from_put_call_parity_with_report",
            boom,
        )

        with warnings.catch_warnings(record=True) as w:
            result = preprocess_with_parity(df, spot_price, discount_factor)

            assert any(
                "Put-call parity preprocessing failed" in str(msg.message) for msg in w
            )

        pd.testing.assert_frame_equal(result, df)

    def test_parity_report_pairs_include_price_source(self):
        """Parity diagnostics should identify the source used for each pair."""
        mid_df = _bid_ask_only_options_data()
        last_df = _robust_forward_options_data()
        explicit_df = _long_call_put_options_data()

        mid_report = preprocess_with_parity(mid_df, 100.0, 1.0).attrs["parity_report"]
        last_report = preprocess_with_parity(last_df, 100.0, 1.0).attrs["parity_report"]
        explicit_report = preprocess_with_parity(explicit_df, 100.0, 1.0).attrs[
            "parity_report"
        ]

        assert {pair["price_source"] for pair in mid_report["pairs"]} == {"mid"}
        assert {pair["price_source"] for pair in last_report["pairs"]} == {"last_price"}
        assert {pair["price_source"] for pair in explicit_report["pairs"]} == {
            "last_price"
        }

    def test_all_valid_chain_selects_only_nearest_five_pairs(self):
        """Forward inference should cap aggregation to five nearest valid pairs."""
        strikes = [float(strike) for strike in range(65, 135)]
        df = _forward_chain(strikes)

        result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert report["candidate_pair_count"] == 70
        assert report["valid_pair_count"] == 70
        assert report["selected_pair_count"] == 5
        assert report["pairs_used_count"] == 5
        assert set(report["selected_strikes"]) == {98.0, 99.0, 100.0, 101.0, 102.0}
        assert report["valid_not_selected_count"] == 65

    def test_far_valid_pairs_are_valid_not_selected_not_outliers(self):
        """Far valid pairs should not be classified as MAD outliers."""
        strikes = [float(strike) for strike in range(65, 135)]
        far_forwards = {
            strike: 150.0
            for strike in strikes
            if strike not in {98.0, 99.0, 100.0, 101.0, 102.0}
        }
        df = _forward_chain(strikes, forward_by_strike=far_forwards)

        result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert result["F_used"].iloc[0] == pytest.approx(100.0)
        assert report["outlier_count"] == 0
        assert 65.0 in report["valid_not_selected_strikes"]
        far_pair = next(pair for pair in report["pairs"] if pair["strike"] == 65.0)
        assert far_pair["valid"] is True
        assert far_pair["selected"] is False
        assert far_pair["outlier"] is False
        assert far_pair["status"] == "valid_not_selected"

    def test_mad_filtering_applies_only_inside_selected_subset(self):
        """Far valid forwards should not change the selected-subset median."""
        strikes = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0]
        far_forwards = {
            95.0: 180.0,
            96.0: 180.0,
            97.0: 180.0,
            103.0: 180.0,
            104.0: 180.0,
        }
        df = _forward_chain(strikes, forward_by_strike=far_forwards)

        result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert result["F_used"].iloc[0] == pytest.approx(100.0)
        assert report["selected_pair_count"] == 5
        assert report["outlier_count"] == 0
        assert set(report["valid_not_selected_strikes"]) == {
            95.0,
            96.0,
            97.0,
            103.0,
            104.0,
        }

    def test_invalid_near_atm_pair_does_not_consume_selection_cap(self):
        """Invalid near-ATM candidates should not count against max pairs."""
        df = _forward_chain([99.0, 101.0, 98.0, 102.0, 97.0])
        invalid_atm = pd.DataFrame(
            [
                {"strike": 100.0, "option_type": "C", "last_price": 1.0},
                {"strike": 100.0, "option_type": "P", "last_price": 200.0},
            ]
        )
        df = pd.concat([invalid_atm, df], ignore_index=True)

        result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert report["candidate_pair_count"] == 6
        assert report["valid_pair_count"] == 5
        assert report["selected_pair_count"] == 5
        assert set(report["selected_strikes"]) == {97.0, 98.0, 99.0, 101.0, 102.0}
        invalid_pair = next(pair for pair in report["pairs"] if pair["strike"] == 100.0)
        assert invalid_pair["valid"] is False
        assert invalid_pair["selected"] is False
        assert invalid_pair["status"] == "invalid"

    def test_max_forward_pairs_three_selects_exactly_three(self):
        """The internal max pair cap should be configurable."""
        strikes = [float(strike) for strike in range(65, 135)]
        df = _forward_chain(strikes)

        result = preprocess_with_parity(df, 100.0, 1.0, max_forward_pairs=3)
        report = result.attrs["parity_report"]

        assert report["max_forward_pairs"] == 3
        assert report["selected_pair_count"] == 3
        assert report["pairs_used_count"] == 3
        assert set(report["selected_strikes"]) == {99.0, 100.0, 101.0}

    @pytest.mark.parametrize("max_forward_pairs", [True, False, 0, -1, np.nan, np.inf])
    def test_max_forward_pairs_rejects_invalid_values(self, max_forward_pairs):
        """The max pair cap must be a positive non-boolean integer."""
        df = _forward_chain([95.0, 100.0, 105.0])

        with pytest.raises(ValueError, match="max_forward_pairs"):
            infer_forward_from_put_call_parity(
                df,
                100.0,
                1.0,
                max_forward_pairs=max_forward_pairs,
            )

    @pytest.mark.parametrize(
        "min_last_leg_volume",
        [True, False, np.nan, np.inf, -1.0],
    )
    def test_min_last_leg_volume_rejects_invalid_values(self, min_last_leg_volume):
        """The last-price volume floor must be finite, non-boolean, or None."""
        df = _forward_chain([95.0, 100.0, 105.0])

        with pytest.raises(ValueError, match="min_last_leg_volume"):
            infer_forward_from_put_call_parity(
                df,
                100.0,
                1.0,
                min_last_leg_volume=min_last_leg_volume,
            )

    @pytest.mark.parametrize(
        "max_bid_ask_relative_spread",
        [True, False, 0.0, -0.1, np.nan, np.inf],
    )
    def test_max_bid_ask_relative_spread_rejects_invalid_values(
        self, max_bid_ask_relative_spread
    ):
        """The bid/ask spread ceiling must be finite, positive, non-boolean, or None."""
        df = _forward_chain([95.0, 100.0, 105.0])

        with pytest.raises(ValueError, match="max_bid_ask_relative_spread"):
            infer_forward_from_put_call_parity(
                df,
                100.0,
                1.0,
                max_bid_ask_relative_spread=max_bid_ask_relative_spread,
            )

    def test_optional_volume_and_spread_config_none_values_are_accepted(self):
        """None should continue disabling optional volume and spread filters."""
        df = _forward_chain([95.0, 100.0, 105.0])

        forward = infer_forward_from_put_call_parity(
            df,
            100.0,
            1.0,
            min_last_leg_volume=None,
            max_bid_ask_relative_spread=None,
        )

        assert forward == pytest.approx(100.0)

    def test_min_last_leg_volume_zero_is_accepted(self):
        """A zero volume floor should be accepted because the floor is non-negative."""
        df = _forward_chain([95.0, 100.0, 105.0])

        forward = infer_forward_from_put_call_parity(
            df,
            100.0,
            1.0,
            min_last_leg_volume=0.0,
        )

        assert forward == pytest.approx(100.0)

    def test_bid_ask_relative_spread_at_threshold_passes(self):
        """A bid/ask relative spread equal to the threshold should be valid."""
        df = _row_quote_pair(
            call_mid=4.0,
            put_mid=2.0,
            relative_spread=0.25,
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]
        pair = report["pairs"][0]

        assert result["F_used"].iloc[0] == pytest.approx(102.0)
        assert pair["status"] == "used"
        assert pair["price_source"] == "mid"
        assert pair["call_relative_spread"] == pytest.approx(0.25)
        assert pair["put_relative_spread"] == pytest.approx(0.25)

    def test_wide_bid_ask_accepted_by_default_when_spread_gate_disabled(self):
        """Wide bid/ask spreads should be accepted by the default midquote path."""
        df = _row_quote_pair(
            call_mid=4.0,
            put_mid=2.0,
            relative_spread=0.50,
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]
        pair = report["pairs"][0]

        assert detect_parity_opportunity(df) is True
        assert result["F_used"].iloc[0] == pytest.approx(102.0)
        assert pair["price_source"] == "mid"
        assert pair["call_relative_spread"] == pytest.approx(0.50)
        assert pair["put_relative_spread"] == pytest.approx(0.50)

    def test_explicit_bid_ask_spread_gate_rejects_wide_mids(self):
        """An explicit spread ceiling should still reject too-wide midquotes."""
        df = _row_quote_pair(
            call_mid=4.0,
            put_mid=2.0,
            relative_spread=0.50,
        )

        with pytest.raises(ValueError, match="No valid put-call parity pairs"):
            infer_forward_from_put_call_parity(
                df,
                100.0,
                1.0,
                max_bid_ask_relative_spread=0.25,
            )

    def test_wide_bid_ask_uses_valid_last_price_fallback(self):
        """An explicit spread ceiling should allow liquid last-price fallback."""
        df = _row_quote_pair(
            call_mid=20.0,
            put_mid=1.0,
            relative_spread=0.50,
            call_last=4.0,
            put_last=2.0,
            call_volume=10.0,
            put_volume=10.0,
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            result = preprocess_with_parity(
                df,
                100.0,
                1.0,
                max_bid_ask_relative_spread=0.25,
            )
        report = result.attrs["parity_report"]
        pair = report["pairs"][0]

        assert result["F_used"].iloc[0] == pytest.approx(102.0)
        assert pair["price_source"] == "last_price"
        assert pair["volume_filter_status"] == "passed"

    def test_last_price_fallback_rejects_low_lowercase_volume(self):
        """Lowercase volume below the floor should reject last-price fallback."""
        df = _row_quote_pair(
            call_last=4.0,
            put_last=2.0,
            call_volume=0.0,
            put_volume=10.0,
        )

        with pytest.raises(ValueError, match="No valid put-call parity pairs"):
            infer_forward_from_put_call_parity(df, 100.0, 1.0)
        assert detect_parity_opportunity(df) is False

    @pytest.mark.parametrize(
        "call_volume,put_volume",
        [(0.0, None), (None, 0.0), (0.0, np.nan), (np.nan, 0.0)],
    )
    def test_last_price_fallback_rejects_observed_low_volume_with_other_missing(
        self, call_volume, put_volume
    ):
        """An observed below-floor lowercase volume should reject the last pair."""
        df = _row_quote_pair(
            call_last=4.0,
            put_last=2.0,
            call_volume=call_volume,
            put_volume=put_volume,
        )

        with pytest.raises(ValueError, match="No valid put-call parity pairs"):
            infer_forward_from_put_call_parity(df, 100.0, 1.0)
        assert detect_parity_opportunity(df) is False

    def test_last_price_fallback_missing_volume_is_allowed_low_confidence(self):
        """Missing lowercase volume should be marked unavailable and lower confidence."""
        df = _row_quote_pair(call_last=4.0, put_last=2.0)

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]
        pair = report["pairs"][0]

        assert result["F_used"].iloc[0] == pytest.approx(102.0)
        assert report["quote_liquidity_confidence"] == "low"
        assert pair["volume_filter_status"] == "unavailable"
        assert pair["call_volume"] is None
        assert pair["put_volume"] is None

    def test_uppercase_volume_is_ignored_for_last_price_fallback(self):
        """Legacy uppercase Volume should not satisfy the liquidity filter."""
        df = _row_quote_pair(
            call_last=4.0,
            put_last=2.0,
            uppercase_volume=100.0,
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]
        pair = report["pairs"][0]

        assert result["F_used"].iloc[0] == pytest.approx(102.0)
        assert "Volume" not in result.columns
        assert pair["volume_filter_status"] == "unavailable"
        assert report["quote_liquidity_confidence"] == "low"

    def test_precomputed_mid_with_nan_bid_ask_beats_stale_last_prices(self):
        """Both-leg precomputed mids should be used when bid/ask values are NaN."""
        df = pd.DataFrame(
            {
                "strike": [100.0, 100.0],
                "option_type": ["C", "P"],
                "bid": [np.nan, np.nan],
                "ask": [np.nan, np.nan],
                "mid": [4.0, 2.0],
                "last_price": [20.0, 1.0],
            }
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]
        pair = report["pairs"][0]

        assert result["F_used"].iloc[0] == pytest.approx(102.0)
        assert pair["price_source"] == "mid"
        assert pair["call_relative_spread"] is None
        assert pair["put_relative_spread"] is None
        assert report["quote_liquidity_confidence"] == "medium"

    def test_preprocess_no_benefit_preserves_mid_when_bid_ask_are_nan(self):
        """No-benefit fallback must not overwrite valid mids with NaN bid/ask mids."""
        df = pd.DataFrame(
            {
                "strike": [100.0, 105.0],
                "option_type": ["C", "C"],
                "bid": [np.nan, np.nan],
                "ask": [np.nan, np.nan],
                "mid": [2.0, 1.5],
            }
        )

        result = preprocess_with_parity(df, 100.0, 1.0)

        assert result["mid"].tolist() == [2.0, 1.5]

    def test_preprocess_normalizes_lowercase_full_option_types(self):
        """Preprocessing should handle full lowercase option type labels."""
        df = pd.DataFrame(
            {
                "strike": [100.0, 100.0],
                "option_type": ["call", "put"],
                "last_price": [4.0, 2.0],
            }
        )

        with pytest.warns(UserWarning, match="1 valid pair.*low_single_pair"):
            result = preprocess_with_parity(df, 100.0, 1.0)

        assert len(result) == 1
        assert result["F_used"].iloc[0] == pytest.approx(102.0)
        assert result["source"].iloc[0] == "put_converted"

    def test_diagnostics_counts_reconcile(self):
        """Report-level selected, used, outlier, and unselected counts should add up."""
        strikes = [96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0]
        df = _forward_chain(strikes, forward_by_strike={100.0: 150.0})

        result = preprocess_with_parity(df, 100.0, 1.0)
        report = result.attrs["parity_report"]

        assert result["F_used"].iloc[0] == pytest.approx(100.0)
        assert report["valid_pair_count"] == (
            report["selected_pair_count"] + report["valid_not_selected_count"]
        )
        assert report["selected_pair_count"] == (
            report["pairs_used_count"] + report["outlier_count"]
        )
        assert report["outlier_count"] == 1
        assert report["valid_not_selected_count"] == 3


class TestIntegrationScenarios:
    """Integration tests with realistic market scenarios."""

    def test_realistic_market_scenario(self):
        """Test with realistic market data scenario."""
        # Create realistic options data
        # Spot = 100, forward ~= 100.25 (5% rate, 90 days)
        strikes = [85, 90, 95, 100, 105, 110, 115]

        data = []
        for strike in strikes:
            # Realistic call prices (decreasing with strike)
            call_price = max(0.1, 15 - 0.15 * strike + np.random.uniform(-0.2, 0.2))
            # Realistic put prices (increasing with strike)
            put_price = max(0.1, 0.05 * strike - 2 + np.random.uniform(-0.2, 0.2))

            data.append(
                {
                    "strike": strike,
                    "last_price": call_price,
                    "option_type": "C",
                }
            )
            data.append({"strike": strike, "last_price": put_price, "option_type": "P"})

        df = pd.DataFrame(data)

        spot_price = 100.0
        discount_factor = np.exp(-0.05 * 90 / 365)  # 5% rate, 90 days

        result = preprocess_with_parity(df, spot_price, discount_factor)

        # Basic checks
        assert "last_price" in result.columns
        assert len(result) == len(strikes)
        assert all(result["last_price"] > 0)  # All prices should be positive
        assert "source" in result.columns

        # Should have mix of call and put_converted
        sources = result["source"].value_counts()
        assert "call" in sources
        assert "put_converted" in sources

    def test_missing_data_handling(self):
        """Test handling of missing data in realistic scenarios."""
        # Create data with some missing puts or calls
        df = pd.DataFrame(
            {
                "strike": [
                    95,
                    95,
                    100,
                    105,
                    105,
                ],  # Missing put at 100, call at 105
                "last_price": [6.0, 1.0, 2.5, 0.5, 4.5],
                "option_type": [
                    "C",
                    "P",
                    "C",
                    "C",
                    "P",
                ],  # Note: 105 has both C and P
            }
        )

        spot_price = 100.0
        discount_factor = 0.99

        with pytest.warns(UserWarning, match="2 valid pairs.*low_two_pairs"):
            result = preprocess_with_parity(df, spot_price, discount_factor)

        # Should return rows only where usable prices exist
        assert len(result) == 2
        assert set(result["strike"]) == {95, 105}
        assert result.loc[result["strike"] == 95, "source"].iloc[0] == "put_converted"
        assert result.loc[result["strike"] == 105, "source"].iloc[0] == "call"


if __name__ == "__main__":
    pytest.main([__file__])
