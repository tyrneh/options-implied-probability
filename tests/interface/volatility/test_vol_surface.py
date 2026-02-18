"""
Test Skeleton for VolSurface Interface
=======================================
Tests for the multi-expiry volatility surface fitting API.

Based on first-principles analysis of oipd.interface.volatility.VolSurface
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def multi_expiry_chain():
    """Option chain with multiple expiries (calls and puts)."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]

    exp1 = pd.Timestamp("2025-02-21")
    calls1 = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [12.0, 8.0, 5.0, 3.0, 1.5],
            "bid": [11.5, 7.5, 4.5, 2.5, 1.2],
            "ask": [12.5, 8.5, 5.5, 3.5, 1.8],
            "option_type": ["C", "C", "C", "C", "C"],
            "expiry": [exp1] * 5,
        }
    )

    exp2 = pd.Timestamp("2025-05-21")
    calls2 = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [14.0, 10.0, 7.0, 4.5, 2.5],
            "bid": [13.5, 9.5, 6.5, 4.0, 2.2],
            "ask": [14.5, 10.5, 7.5, 5.0, 2.8],
            "option_type": ["C", "C", "C", "C", "C"],
            "expiry": [exp2] * 5,
        }
    )

    calls = pd.concat([calls1, calls2], ignore_index=True)

    # Generate puts via parity: P = C - S + K * df
    S, r = 100.0, 0.05
    t_array = (calls["expiry"] - pd.Timestamp("2025-01-01")).dt.days / 365.0
    df_array = np.exp(-r * t_array)

    puts = calls.copy()
    puts["option_type"] = "P"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df_array).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df_array).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df_array).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def market_inputs():
    """Standard MarketInputs for surface fitting (no expiry_date)."""
    from oipd import MarketInputs

    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
    )


@pytest.fixture
def mixed_quality_multi_expiry_chain(multi_expiry_chain):
    """Option chain with one intentionally under-specified expiry slice."""
    bad_expiry = pd.Timestamp("2025-07-21")
    bad_strikes = [92.0, 97.0, 102.0, 107.0]

    bad_calls = pd.DataFrame(
        {
            "strike": bad_strikes,
            "last_price": [10.5, 7.1, 4.2, 2.3],
            "bid": [10.1, 6.8, 3.9, 2.0],
            "ask": [10.9, 7.4, 4.5, 2.6],
            "option_type": ["C", "C", "C", "C"],
            "expiry": [bad_expiry] * len(bad_strikes),
        }
    )

    t_years = (bad_expiry - pd.Timestamp("2025-01-01")).days / 365.0
    discount_factor = np.exp(-0.05 * t_years)

    bad_puts = bad_calls.copy()
    bad_puts["option_type"] = "P"
    bad_puts["last_price"] = (
        bad_calls["last_price"] - 100.0 + bad_calls["strike"] * discount_factor
    ).abs()
    bad_puts["bid"] = (
        bad_calls["bid"] - 100.0 + bad_calls["strike"] * discount_factor
    ).abs()
    bad_puts["ask"] = (
        bad_calls["ask"] - 100.0 + bad_calls["strike"] * discount_factor
    ).abs()

    return pd.concat([multi_expiry_chain, bad_calls, bad_puts], ignore_index=True)


@pytest.fixture
def mostly_bad_multi_expiry_chain(mixed_quality_multi_expiry_chain):
    """Option chain with one good expiry and one under-specified expiry."""
    selected_expiries = [
        pd.Timestamp("2025-02-21"),
        pd.Timestamp("2025-07-21"),
    ]
    return mixed_quality_multi_expiry_chain[
        mixed_quality_multi_expiry_chain["expiry"].isin(selected_expiries)
    ].copy()


# =============================================================================
# VolSurface.__init__() Tests
# =============================================================================


class TestVolSurfaceInit:
    """Tests for VolSurface initialization."""

    def test_default_initialization(self):
        """VolSurface initializes with default parameters."""
        from oipd import VolSurface

        vs = VolSurface()
        # Just verify it initializes without error
        assert vs is not None


# =============================================================================
# VolSurface.fit() Tests
# =============================================================================


class TestVolSurfaceFit:
    """Tests for VolSurface.fit() method."""

    def test_fit_returns_self(self, multi_expiry_chain, market_inputs):
        """fit() returns self for chaining."""
        from oipd import VolSurface

        vs = VolSurface()
        result = vs.fit(multi_expiry_chain, market_inputs)
        assert result is vs

    def test_fit_populates_expiries(self, multi_expiry_chain, market_inputs):
        """fit() populates .expiries property."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        assert len(vs.expiries) == 2

    def test_fit_with_horizon_filter(self, multi_expiry_chain, market_inputs):
        """fit() respects horizon parameter."""
        from oipd import VolSurface
        from oipd.core.errors import CalculationError

        vs = VolSurface()
        # horizon="2m" from Jan 1 2025 should filter to ~Feb expiry
        with pytest.raises(CalculationError, match="at least two"):
            vs.fit(multi_expiry_chain, market_inputs, horizon="2m")

    def test_fit_rejects_single_expiry(self, multi_expiry_chain, market_inputs):
        """fit() raises when only one expiry is provided."""
        from oipd import VolSurface
        from oipd.core.errors import CalculationError

        vs = VolSurface()
        single_expiry_chain = multi_expiry_chain[
            multi_expiry_chain["expiry"] == multi_expiry_chain["expiry"].iloc[0]
        ]
        with pytest.raises(CalculationError, match="at least two"):
            vs.fit(single_expiry_chain, market_inputs)

    def test_fit_default_skip_warn_skips_failed_expiries(
        self, mixed_quality_multi_expiry_chain, market_inputs
    ):
        """Default policy skips failed slices and emits aggregate warning."""
        from oipd import VolSurface

        vs = VolSurface()
        with pytest.warns(UserWarning, match="Skipped .* expiries during surface fit"):
            vs.fit(mixed_quality_multi_expiry_chain, market_inputs)
        assert len(vs.expiries) == 2
        assert pd.Timestamp("2025-07-21") not in vs.expiries

    def test_fit_raise_policy_suggests_skip_warn(
        self, mixed_quality_multi_expiry_chain, market_inputs
    ):
        """Strict mode includes actionable guidance for partial calibration."""
        from oipd import VolSurface
        from oipd.core.errors import CalculationError

        vs = VolSurface()
        with pytest.raises(CalculationError, match="failure_policy='skip_warn'"):
            vs.fit(
                mixed_quality_multi_expiry_chain,
                market_inputs,
                failure_policy="raise",
            )

    def test_fit_skip_warn_requires_two_successful_expiries(
        self, mostly_bad_multi_expiry_chain, market_inputs
    ):
        """Skip mode still raises when fewer than two slices calibrate."""
        from oipd import VolSurface
        from oipd.core.errors import CalculationError

        vs = VolSurface()
        with pytest.raises(
            CalculationError,
            match="at least two successfully calibrated expiries",
        ):
            vs.fit(
                mostly_bad_multi_expiry_chain,
                market_inputs,
                failure_policy="skip_warn",
            )

    def test_fit_invalid_failure_policy_raises(self, multi_expiry_chain, market_inputs):
        """Invalid failure policy is rejected at interface boundary."""
        from oipd import VolSurface

        vs = VolSurface()
        with pytest.raises(ValueError, match="failure_policy must be either"):
            vs.fit(
                multi_expiry_chain,
                market_inputs,
                failure_policy="bad",  # type: ignore[arg-type]
            )


# =============================================================================
# VolSurface.slice() Tests
# =============================================================================


class TestVolSurfaceSlice:
    """Tests for VolSurface.slice() method."""

    def test_slice_returns_volcurve(self, multi_expiry_chain, market_inputs):
        """slice(expiry) returns a VolCurve object."""
        from oipd import VolSurface, VolCurve

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        first_exp = vs.expiries[0]
        curve = vs.slice(first_exp)
        assert isinstance(curve, VolCurve)

    def test_slice_has_params(self, multi_expiry_chain, market_inputs):
        """Sliced VolCurve has accessible params."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        curve = vs.slice(vs.expiries[0])
        assert curve.params is not None

    def test_slice_has_atm_vol_and_diagnostics(self, multi_expiry_chain, market_inputs):
        """Sliced VolCurve exposes ATM vol and diagnostics."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        curve = vs.slice(vs.expiries[0])
        atm_vol = curve.atm_vol
        assert np.isfinite(atm_vol)
        assert atm_vol > 0
        diagnostics = curve.diagnostics
        assert diagnostics is not None
        assert hasattr(diagnostics, "status")

    def test_slice_invalid_expiry_raises(self, multi_expiry_chain, market_inputs):
        """slice() with invalid expiry raises ValueError."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        with pytest.raises(ValueError):
            vs.slice("2030-12-31")

    def test_slice_rejects_float_t(self, multi_expiry_chain, market_inputs):
        """slice() requires date-like expiry and rejects float t input."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        with pytest.raises(ValueError, match="date-like expiry"):
            vs.slice(45 / 365.0)  # type: ignore[arg-type]


# =============================================================================
# VolSurface with Interpolation Tests
# =============================================================================


class TestVolSurfaceInterpolation:
    """Tests for VolSurface interpolation functionality."""

    def test_fit_creates_interpolator(self, multi_expiry_chain, market_inputs):
        """fit() automatically creates interpolator."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        # Access private attribute (or check via implied_vol method)
        assert vs._interpolator is not None

    def test_implied_vol_at_arbitrary_time(self, multi_expiry_chain, market_inputs):
        """implied_vol(K, t) works for arbitrary expiry time."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        iv = vs.implied_vol(100.0, 60 / 365.0)  # 60 days
        assert 0.0 < iv < 2.0

    def test_total_variance_at_arbitrary_time(self, multi_expiry_chain, market_inputs):
        """total_variance(K, t) returns positive value."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        w = vs.total_variance(100.0, 60 / 365.0)
        assert w > 0


# =============================================================================
# VolSurface Maturity Domain Tests
# =============================================================================


class TestVolSurfaceMaturityDomain:
    """Strict maturity-domain behavior across VolSurface query methods."""

    def test_implied_vol_date_and_float_are_consistent(
        self, multi_expiry_chain, market_inputs
    ):
        """implied_vol accepts date-like t and matches equivalent float t."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)

        iv_date = vs.implied_vol(100.0, "2025-02-15")
        iv_float = vs.implied_vol(100.0, 45 / 365.0)
        np.testing.assert_allclose(iv_date, iv_float, rtol=1e-8)

    def test_non_positive_t_raises_across_surface_queries(
        self, multi_expiry_chain, market_inputs
    ):
        """Methods with maturity input reject non-positive t uniformly."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)

        with pytest.raises(ValueError, match="strictly positive"):
            vs.implied_vol(100.0, 0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            vs.total_variance(100.0, 0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            vs.forward_price(0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            vs.price([100.0], t=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            vs.atm_vol(0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            vs.greeks([100.0], t=0.0)

    def test_t_beyond_last_pillar_raises_across_surface_queries(
        self, multi_expiry_chain, market_inputs
    ):
        """Methods with maturity input reject long-end extrapolation uniformly."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)

        beyond_last_expiry = "2030-12-31"

        with pytest.raises(ValueError, match="last fitted pillar"):
            vs.implied_vol(100.0, beyond_last_expiry)
        with pytest.raises(ValueError, match="last fitted pillar"):
            vs.total_variance(100.0, beyond_last_expiry)
        with pytest.raises(ValueError, match="last fitted pillar"):
            vs.forward_price(beyond_last_expiry)
        with pytest.raises(ValueError, match="last fitted pillar"):
            vs.price([100.0], t=beyond_last_expiry)
        with pytest.raises(ValueError, match="last fitted pillar"):
            vs.atm_vol(beyond_last_expiry)
        with pytest.raises(ValueError, match="last fitted pillar"):
            vs.greeks([100.0], t=beyond_last_expiry)


# =============================================================================
# VolSurface.implied_distribution() Tests
# =============================================================================


class TestVolSurfaceImpliedDistribution:
    """Tests for deriving ProbSurface from VolSurface."""

    def test_implied_distribution_returns_probsurface(
        self, multi_expiry_chain, market_inputs
    ):
        """implied_distribution() returns a ProbSurface."""
        from oipd import VolSurface
        from oipd.interface.probability import ProbSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        prob_surface = vs.implied_distribution()
        assert isinstance(prob_surface, ProbSurface)

    def test_probsurface_has_matching_expiries(self, multi_expiry_chain, market_inputs):
        """ProbSurface has same expiries as VolSurface."""
        from oipd import VolSurface

        vs = VolSurface()
        vs.fit(multi_expiry_chain, market_inputs)
        prob_surface = vs.implied_distribution()
        assert len(prob_surface.expiries) == len(vs.expiries)


# =============================================================================
# VolSurface.plot() Tests
# =============================================================================


class TestVolSurfacePlot:
    """Tests for VolSurface plotting methods."""

    def test_plot_term_structure_uses_interpolated_grid(
        self, multi_expiry_chain, market_inputs
    ):
        """plot_term_structure() uses an interpolated grid, not just pillars."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from oipd import VolSurface

        vs = VolSurface(pricing_engine="bs")
        vs.fit(multi_expiry_chain, market_inputs)
        fig = vs.plot_term_structure()
        ax = fig.axes[0]
        assert ax.lines, "Expected a line to be plotted."
        xdata = ax.lines[0].get_xdata()
        assert len(xdata) > len(vs.expiries)
        plt.close(fig)
