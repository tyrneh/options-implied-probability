"""
Test Skeleton for ProbSurface Interface
========================================
Tests for the multi-expiry probability surface API.

Based on first-principles analysis of oipd.interface.probability.ProbSurface
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
    """Standard MarketInputs for probability surface tests."""
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
def fitted_vol_surface(multi_expiry_chain, market_inputs):
    """A pre-fitted VolSurface for deriving probability surface."""
    from oipd import VolSurface

    vs = VolSurface(pricing_engine="bs")
    vs.fit(multi_expiry_chain, market_inputs)
    return vs


@pytest.fixture
def prob_surface(fitted_vol_surface):
    """A ProbSurface derived from a fitted VolSurface."""
    return fitted_vol_surface.implied_distribution()


# =============================================================================
# ProbSurface.from_chain() Tests
# =============================================================================


class TestProbSurfaceFromChain:
    """Tests for ProbSurface.from_chain() constructor."""

    def test_from_chain_returns_probsurface(self, multi_expiry_chain, market_inputs):
        """from_chain() returns a ProbSurface."""
        from oipd import ProbSurface

        prob = ProbSurface.from_chain(multi_expiry_chain, market_inputs)
        assert isinstance(prob, ProbSurface)

    def test_from_chain_accepts_max_staleness(self, multi_expiry_chain, market_inputs):
        """from_chain() accepts max_staleness_days."""
        from oipd import ProbSurface

        prob = ProbSurface.from_chain(
            multi_expiry_chain,
            market_inputs,
            max_staleness_days=1,
        )
        assert isinstance(prob, ProbSurface)

    def test_from_chain_rejects_single_expiry(self, multi_expiry_chain, market_inputs):
        """from_chain() raises when only one expiry is provided."""
        from oipd import ProbSurface
        from oipd.core.errors import CalculationError

        single_expiry_chain = multi_expiry_chain[
            multi_expiry_chain["expiry"] == multi_expiry_chain["expiry"].iloc[0]
        ]
        with pytest.raises(CalculationError, match="at least two"):
            ProbSurface.from_chain(single_expiry_chain, market_inputs)

    def test_from_chain_default_skip_warn_succeeds(
        self, mixed_quality_multi_expiry_chain, market_inputs
    ):
        """Default policy skips failed slices and still returns ProbSurface."""
        from oipd import ProbSurface

        with pytest.warns(
            UserWarning, match="Skipped .* expiries during surface fit"
        ):
            prob = ProbSurface.from_chain(mixed_quality_multi_expiry_chain, market_inputs)
        assert isinstance(prob, ProbSurface)
        assert len(prob.expiries) == 2
        assert pd.Timestamp("2025-07-21") not in prob.expiries

    def test_from_chain_raise_policy_suggests_skip_warn(
        self, mixed_quality_multi_expiry_chain, market_inputs
    ):
        """Strict mode passes through actionable guidance."""
        from oipd import ProbSurface
        from oipd.core.errors import CalculationError

        with pytest.raises(CalculationError, match="failure_policy='skip_warn'"):
            ProbSurface.from_chain(
                mixed_quality_multi_expiry_chain,
                market_inputs,
                failure_policy="raise",
            )

    def test_from_chain_invalid_failure_policy_raises(
        self, multi_expiry_chain, market_inputs
    ):
        """Invalid failure policy is rejected at the interface boundary."""
        from oipd import ProbSurface

        with pytest.raises(ValueError, match="failure_policy must be either"):
            ProbSurface.from_chain(
                multi_expiry_chain,
                market_inputs,
                failure_policy="bad",  # type: ignore[arg-type]
            )


# =============================================================================
# ProbSurface Constructor Strictness Tests
# =============================================================================


class TestProbSurfaceConstructorStrictness:
    """Tests for strict VolSurface-backed ProbSurface construction."""

    def test_constructor_requires_vol_surface_keyword(self):
        """ProbSurface() without required keyword raises TypeError."""
        from oipd.interface.probability import ProbSurface

        with pytest.raises(TypeError):
            ProbSurface()  # type: ignore[call-arg]

    def test_constructor_rejects_unfitted_vol_surface(self):
        """ProbSurface rejects unfitted VolSurface immediately."""
        from oipd import VolSurface
        from oipd.interface.probability import ProbSurface

        unfitted_surface = VolSurface()
        with pytest.raises(ValueError, match="requires a fitted VolSurface"):
            ProbSurface(vol_surface=unfitted_surface)


# =============================================================================
# ProbSurface Basic Properties Tests
# =============================================================================


class TestProbSurfaceProperties:
    """Tests for ProbSurface basic properties."""

    def test_expiries_returns_tuple(self, prob_surface):
        """expiries property returns a tuple."""
        assert isinstance(prob_surface.expiries, tuple)

    def test_expiries_matches_vol_surface(self, fitted_vol_surface, prob_surface):
        """ProbSurface has same expiries as source VolSurface."""
        assert len(prob_surface.expiries) == len(fitted_vol_surface.expiries)


# =============================================================================
# ProbSurface.slice() Tests
# =============================================================================


class TestProbSurfaceSlice:
    """Tests for ProbSurface.slice() method."""

    def test_slice_returns_probcurve(self, prob_surface):
        """slice(expiry) returns a ProbCurve object."""
        from oipd.interface.probability import ProbCurve

        first_exp = prob_surface.expiries[0]
        curve = prob_surface.slice(first_exp)
        assert isinstance(curve, ProbCurve)

    def test_slice_has_valid_pdf(self, prob_surface):
        """Sliced ProbCurve has valid PDF."""
        curve = prob_surface.slice(prob_surface.expiries[0])
        assert np.all(curve.pdf_values >= 0)

    def test_slice_metadata_and_market(self, prob_surface):
        """Sliced ProbCurve exposes metadata and resolved market."""
        curve = prob_surface.slice(prob_surface.expiries[0])
        metadata = curve.metadata
        assert isinstance(metadata, dict)
        assert "expiry_date" in metadata
        assert "forward_price" in metadata
        assert "at_money_vol" in metadata
        assert np.isfinite(metadata["at_money_vol"])
        resolved_market = curve.resolved_market
        assert resolved_market is not None
        assert hasattr(resolved_market, "valuation_date")

    def test_slice_interpolates_interior_expiry(self, prob_surface):
        """slice() interpolates when expiry lies between fitted pillars."""
        from oipd.interface.probability import ProbCurve

        interpolated_expiry = "2025-03-21"
        curve = prob_surface.slice(interpolated_expiry)
        assert isinstance(curve, ProbCurve)
        assert bool(curve.metadata.get("interpolated"))

    def test_slice_matches_surface_queries(self, prob_surface, market_inputs):
        """slice(expiry) is consistent with unified surface query methods."""
        interpolated_expiry = pd.Timestamp("2025-03-21")
        t_years = (
            interpolated_expiry - pd.Timestamp(market_inputs.valuation_date)
        ).days / 365.0

        curve = prob_surface.slice(interpolated_expiry)

        q50_curve = curve.quantile(0.5)
        q50_surface_date = prob_surface.quantile(0.5, t=interpolated_expiry)
        q50_surface_float = prob_surface.quantile(0.5, t=t_years)

        np.testing.assert_allclose(q50_curve, q50_surface_date, rtol=1e-6)
        np.testing.assert_allclose(q50_surface_date, q50_surface_float, rtol=1e-6)

        test_prices = np.array([95.0, 100.0, 105.0])
        cdf_from_curve = np.interp(test_prices, curve.prices, curve.cdf_values)
        cdf_from_surface = prob_surface.cdf(test_prices, t=interpolated_expiry)
        np.testing.assert_allclose(cdf_from_curve, cdf_from_surface, rtol=1e-6)

    def test_slice_interpolated_is_cached(self, prob_surface):
        """Repeated interpolated slice requests should hit cache."""
        interpolated_expiry = "2025-03-21"
        first = prob_surface.slice(interpolated_expiry)
        second = prob_surface.slice(interpolated_expiry)
        assert first is second

    def test_slice_invalid_expiry_raises(self, prob_surface):
        """slice() rejects expiries beyond the last calibrated pillar."""
        with pytest.raises(ValueError, match="beyond the last fitted pillar"):
            prob_surface.slice("2030-12-31")

    def test_slice_accepts_string_date(self, prob_surface):
        """slice() accepts string date format."""
        first_exp = prob_surface.expiries[0]
        exp_str = first_exp.strftime("%Y-%m-%d")
        curve = prob_surface.slice(exp_str)
        assert curve is not None


# =============================================================================
# ProbSurface Query API Tests
# =============================================================================


class TestProbSurfaceQueryApi:
    """Tests for callable probability queries on ProbSurface."""

    def test_pdf_with_float_t(self, prob_surface):
        """pdf(price, t) accepts year-fraction maturities."""
        values = prob_surface.pdf(np.array([95.0, 100.0, 105.0]), t=45 / 365.0)
        assert isinstance(values, np.ndarray)
        assert values.shape == (3,)
        assert np.all(values >= 0.0)

    def test_pdf_date_and_float_t_are_consistent(self, prob_surface):
        """Date-like and equivalent float maturities produce similar PDFs."""
        value_from_date = prob_surface.pdf(100.0, t="2025-02-15")
        value_from_float = prob_surface.pdf(100.0, t=45 / 365.0)
        np.testing.assert_allclose(value_from_date, value_from_float, rtol=1e-4)

    def test_call_alias_matches_pdf(self, prob_surface):
        """__call__ delegates to pdf."""
        p1 = prob_surface(100.0, t=45 / 365.0)
        p2 = prob_surface.pdf(100.0, t=45 / 365.0)
        np.testing.assert_allclose(p1, p2, rtol=1e-8)

    def test_cdf_is_bounded(self, prob_surface):
        """cdf(price, t) remains within [0, 1]."""
        cdf_values = prob_surface.cdf(np.array([90.0, 100.0, 110.0]), t=45 / 365.0)
        assert np.all(cdf_values >= 0.0)
        assert np.all(cdf_values <= 1.0)

    def test_quantile_ordering(self, prob_surface):
        """Quantiles should be monotonic in probability level."""
        q10 = prob_surface.quantile(0.10, t=45 / 365.0)
        q50 = prob_surface.quantile(0.50, t=45 / 365.0)
        q90 = prob_surface.quantile(0.90, t=45 / 365.0)
        assert q10 < q50 < q90

    def test_pillar_date_and_float_t_are_consistent(self, prob_surface, market_inputs):
        """Exact pillar date and equivalent float t should agree."""
        pillar_expiry = prob_surface.expiries[0]
        t_years = (
            pillar_expiry - pd.Timestamp(market_inputs.valuation_date)
        ).days / 365.0

        cdf_date = prob_surface.cdf(100.0, t=pillar_expiry)
        cdf_float = prob_surface.cdf(100.0, t=t_years)
        np.testing.assert_allclose(cdf_date, cdf_float, rtol=1e-6)

    def test_non_positive_t_raises(self, prob_surface):
        """Non-positive maturities are rejected."""
        with pytest.raises(ValueError, match="strictly positive"):
            prob_surface.pdf(100.0, t=0.0)

    def test_t_beyond_last_pillar_raises(self, prob_surface):
        """Maturities beyond the last fitted pillar are rejected."""
        with pytest.raises(ValueError, match="last fitted pillar"):
            prob_surface.pdf(100.0, t="2030-12-31")


# =============================================================================
# ProbSurface.plot_fan() Tests
# =============================================================================


class TestProbSurfacePlotFan:
    """Tests for ProbSurface.plot_fan() visualization."""

    def test_plot_fan_does_not_crash(self, prob_surface):
        """plot_fan() executes without raising."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = prob_surface.plot_fan()
        assert fig is not None
        plt.close(fig)

    def test_plot_fan_uses_daily_sampling(self, prob_surface):
        """plot_fan() should sample one maturity point per calendar day."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = prob_surface.plot_fan()
        ax = fig.axes[0]
        assert ax.lines, "Expected an implied-median line on the fan chart."
        xdata = ax.lines[0].get_xdata()

        first_expiry = min(prob_surface.expiries)
        last_expiry = max(prob_surface.expiries)
        expected_points = (last_expiry - first_expiry).days + 1

        assert len(xdata) == expected_points
        plt.close(fig)

    def test_plot_fan_has_no_pillar_regime_jump(self, prob_surface, market_inputs):
        """Daily medians should not show a large discontinuity at pillar dates."""
        first_expiry = min(prob_surface.expiries)
        last_expiry = max(prob_surface.expiries)
        sample_expiries = pd.date_range(first_expiry, last_expiry, freq="D")
        valuation_timestamp = pd.Timestamp(market_inputs.valuation_date)

        medians = []
        for expiry_timestamp in sample_expiries:
            t_years = (expiry_timestamp - valuation_timestamp).days / 365.0
            medians.append(prob_surface.quantile(0.5, t=t_years))
        medians = np.asarray(medians, dtype=float)
        jumps = np.abs(np.diff(medians))

        if len(jumps) < 10:
            pytest.skip("Insufficient daily points for jump-statistic check.")

        pillar_jump_indices = []
        for jump_index, right_day in enumerate(sample_expiries[1:]):
            if right_day in set(prob_surface.expiries):
                pillar_jump_indices.append(jump_index)

        if not pillar_jump_indices:
            pytest.skip("No pillar transitions found in sampled horizon.")

        threshold = float(np.percentile(jumps, 95))
        for jump_index in pillar_jump_indices:
            assert jumps[jump_index] <= threshold + 1e-8
