"""
Test Skeleton for VolCurve Interface
=====================================
Tests for the single-expiry volatility smile fitting API.

Based on first-principles analysis of oipd.interface.volatility.VolCurve
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_option_chain():
    """Minimal option chain with bid/ask/last prices (calls and puts)."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    expiry = pd.Timestamp("2025-03-21")

    calls = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            "bid": [12.0, 7.8, 4.9, 2.9, 1.4],
            "ask": [13.0, 8.6, 5.3, 3.3, 1.8],
            "option_type": ["C", "C", "C", "C", "C"],
            "expiry": [expiry] * 5,
        }
    )

    # Generate puts via parity: P = C - S + K * df
    S, r, T = 100.0, 0.05, 79 / 365.0  # ~79 days to expiry
    df = np.exp(-r * T)
    puts = calls.copy()
    puts["option_type"] = "P"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def market_inputs():
    """Standard MarketInputs for testing."""
    from oipd import MarketInputs

    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
    )


# =============================================================================
# VolCurve.__init__() Tests
# =============================================================================


class TestVolCurveInit:
    """Tests for VolCurve initialization."""

    def test_default_initialization(self):
        """VolCurve initializes with default parameters."""
        from oipd import VolCurve

        vc = VolCurve()
        # Just verify it initializes without error
        assert vc is not None

    def test_custom_method(self):
        """VolCurve accepts custom fitting method."""
        from oipd import VolCurve

        # Just verify it initializes with custom method without error
        vc = VolCurve(method="svi")  # Use known valid method
        assert vc is not None


# =============================================================================
# VolCurve.fit() Tests
# =============================================================================


class TestVolCurveFit:
    """Tests for VolCurve.fit() method."""

    def test_fit_returns_self(self, sample_option_chain, market_inputs):
        """fit() returns self for chaining."""
        from oipd import VolCurve

        vc = VolCurve()
        result = vc.fit(sample_option_chain, market_inputs)
        assert result is vc

    def test_fit_populates_params(self, sample_option_chain, market_inputs):
        """fit() populates .params attribute."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        assert vc.params is not None
        assert "a" in vc.params  # SVI parameters

    def test_fit_populates_forward(self, sample_option_chain, market_inputs):
        """fit() infers and stores forward price."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        assert vc.forward_price is not None
        assert vc.forward_price > 0

    def test_fit_populates_atm_vol(self, sample_option_chain, market_inputs):
        """fit() populates ATM volatility."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        atm_vol = vc.atm_vol
        assert np.isfinite(atm_vol)
        assert atm_vol > 0

    def test_fit_with_column_mapping(self, market_inputs):
        """fit() works with custom column mapping."""
        from oipd import VolCurve

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        expiry = pd.Timestamp("2025-03-21")

        # Create calls
        calls = pd.DataFrame(
            {
                "K": strikes,
                "price": [12.0, 7.0, 4.0, 2.0, 1.0],
                "type": ["C", "C", "C", "C", "C"],
                "exp": [expiry] * 5,
            }
        )

        # Generate puts via parity
        S, r, T = 100.0, 0.05, 79 / 365.0
        df_val = np.exp(-r * T)
        puts = calls.copy()
        puts["type"] = "P"
        puts["price"] = (calls["price"] - S + np.array(strikes) * df_val).abs()

        df = pd.concat([calls, puts], ignore_index=True)

        mapping = {
            "K": "strike",
            "price": "last_price",
            "type": "option_type",
            "exp": "expiry",
        }
        vc = VolCurve()
        vc.fit(df, market_inputs, column_mapping=mapping)
        assert vc.params is not None

    def test_fit_rejects_multiple_expiries(self, sample_option_chain, market_inputs):
        """fit() raises when chain contains multiple expiries."""
        from oipd import VolCurve

        vc = VolCurve()
        bad_chain = sample_option_chain.copy()
        bad_chain.loc[bad_chain.index[:5], "expiry"] = pd.Timestamp("2025-04-21")
        with pytest.raises(ValueError, match="single expiry"):
            vc.fit(bad_chain, market_inputs)


# =============================================================================
# VolCurve Pre-fit Validation Tests
# =============================================================================


class TestVolCurvePreFitErrors:
    """Tests that accessing attributes before fit() raises errors."""

    def test_params_before_fit_raises(self):
        """Accessing .params before fit() raises ValueError."""
        from oipd import VolCurve

        vc = VolCurve()
        with pytest.raises(ValueError, match="Call fit before"):
            _ = vc.params

    def test_forward_before_fit_raises(self):
        """Accessing .forward_price before fit() raises ValueError."""
        from oipd import VolCurve

        vc = VolCurve()
        with pytest.raises(ValueError, match="Call fit before"):
            _ = vc.forward_price

    def test_call_before_fit_raises(self):
        """Calling vc(strikes) before fit() raises ValueError."""
        from oipd import VolCurve

        vc = VolCurve()
        with pytest.raises(ValueError, match="Call fit before"):
            _ = vc([100.0])


# =============================================================================
# VolCurve.__call__() Tests
# =============================================================================


class TestVolCurveCall:
    """Tests for VolCurve callable interface (implied vol evaluation)."""

    def test_call_returns_array(self, sample_option_chain, market_inputs):
        """Calling fitted curve returns numpy array of IVs."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        ivs = vc([95.0, 100.0, 105.0])
        assert isinstance(ivs, np.ndarray)
        assert len(ivs) == 3

    def test_call_returns_positive_vols(self, sample_option_chain, market_inputs):
        """Implied vols from fitted curve are positive."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        ivs = vc([95.0, 100.0, 105.0])
        assert np.all(ivs > 0)
        assert np.all(ivs < 2.0)  # Reasonable upper bound

    def test_call_with_single_strike(self, sample_option_chain, market_inputs):
        """Callable works with single strike."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        iv = vc([100.0])
        assert len(iv) == 1


# =============================================================================
# VolCurve.iv_smile() Tests
# =============================================================================


class TestVolCurveIvSmile:
    """Tests for VolCurve.iv_results() DataFrame output."""

    def test_iv_smile_returns_dataframe(self, sample_option_chain, market_inputs):
        """iv_results() returns a DataFrame."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        smile = vc.iv_results()
        assert isinstance(smile, pd.DataFrame)

    def test_iv_smile_has_expected_columns(self, sample_option_chain, market_inputs):
        """iv_results() DataFrame has expected columns."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        smile = vc.iv_results()
        expected_cols = {"strike", "fitted_iv"}
        assert expected_cols.issubset(set(smile.columns))

    def test_iv_smile_respects_points_arg(self, sample_option_chain, market_inputs):
        """iv_results(points=N) returns approximately N rows."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        smile = vc.iv_results(points=50)
        # May include observed points, so check >= requested
        assert len(smile) >= 50


# =============================================================================
# VolCurve.plot() Tests
# =============================================================================


class TestVolCurvePlot:
    """Tests for VolCurve.plot() visualization."""

    def test_plot_returns_figure(self, sample_option_chain, market_inputs):
        """plot() returns a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        fig = vc.plot()
        assert fig is not None
        plt.close(fig)

    def test_plot_does_not_crash(self, sample_option_chain, market_inputs):
        """plot() executes without raising."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        # Should not raise
        fig = vc.plot(include_observed=True)
        plt.close(fig)


# =============================================================================
# VolCurve.implied_distribution() Tests
# =============================================================================


class TestVolCurveImpliedDistribution:
    """Tests for deriving ProbCurve from VolCurve."""

    def test_implied_distribution_returns_probcurve(
        self, sample_option_chain, market_inputs
    ):
        """implied_distribution() returns a ProbCurve."""
        from oipd import VolCurve
        from oipd.interface.probability import ProbCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        prob = vc.implied_distribution()
        assert isinstance(prob, ProbCurve)

    def test_implied_distribution_has_valid_pdf(
        self, sample_option_chain, market_inputs
    ):
        """Derived ProbCurve has non-negative PDF."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        prob = vc.implied_distribution()
        assert np.all(prob.pdf_values >= 0)


# =============================================================================
# VolCurve.diagnostics Tests
# =============================================================================


class TestVolCurveDiagnostics:
    """Tests for calibration diagnostics."""

    def test_diagnostics_available_after_fit(self, sample_option_chain, market_inputs):
        """diagnostics property is available after fit."""
        from oipd import VolCurve

        vc = VolCurve()
        vc.fit(sample_option_chain, market_inputs)
        diagnostics = vc.diagnostics
        assert diagnostics is not None
        assert hasattr(diagnostics, "status")
