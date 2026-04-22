"""
Test Skeleton for ProbCurve Interface
======================================
Tests for the single-expiry probability distribution API.

Based on first-principles analysis of oipd.interface.probability.ProbCurve
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

from oipd.core.maturity import resolve_maturity
from oipd.pipelines.probability import derive_distribution_from_curve, quantile_from_cdf


def _derive_direct_probability_arrays(vol_curve, *, points=None):
    """Derive probability arrays through the stateless pipeline.

    Args:
        vol_curve: Fitted public VolCurve object.
        points: Optional native grid resolution for direct derivation.

    Returns:
        tuple: Direct ``prices``, ``pdf_values``, ``cdf_values``, and metadata.
    """
    return derive_distribution_from_curve(
        vol_curve._vol_curve,
        vol_curve.resolved_market,
        pricing_engine=vol_curve.pricing_engine,
        vol_metadata=vol_curve._metadata,
        points=points,
    )


def _assert_direct_cdf_diagnostics_pass(metadata):
    """Assert canonical direct-CDF metadata satisfies current thresholds.

    Args:
        metadata: Probability metadata emitted during materialization.
    """
    assert metadata["cdf_method"] == "call_price_first_derivative"
    assert metadata["cdf_cleanup_policy"] == "minimal_epsilon_cleanup"
    assert (
        metadata["cdf_upper_tail_clip_policy"] == "clip_finite_monotone_small_overshoot"
    )
    assert np.isfinite(metadata["raw_cdf_start"])
    assert np.isfinite(metadata["raw_cdf_end"])
    assert np.isfinite(metadata["raw_cdf_min"])
    assert np.isfinite(metadata["raw_cdf_max"])
    assert np.isfinite(metadata["raw_cdf_max_negative_step"])
    assert np.isfinite(metadata["cdf_pdf_interval_max_error"])
    assert np.isfinite(metadata["cdf_pdf_interval_mean_error"])
    assert metadata["native_grid_policy"] in {"auto", "fixed"}
    assert metadata["native_grid_points"] > 0
    assert metadata["native_grid_min_points"] == 241
    assert metadata["native_grid_max_points"] == 2500
    assert metadata["native_grid_actual_step"] > 0.0
    assert metadata["native_grid_reference_price"] >= 1.0
    assert metadata["native_grid_domain_width"] > 0.0
    raw_lower_boundary_tolerance = 1e-4
    raw_upper_boundary_tolerance = metadata["cdf_upper_tail_clip_tolerance"]
    raw_monotonicity_tolerance = 1e-6
    assert metadata["raw_cdf_is_monotone"]
    assert metadata["raw_cdf_negative_step_count"] == 0
    assert metadata["raw_cdf_min"] >= -raw_lower_boundary_tolerance
    assert metadata["raw_cdf_max"] <= 1.0 + raw_upper_boundary_tolerance
    assert abs(metadata["raw_cdf_start"]) <= raw_lower_boundary_tolerance
    assert abs(metadata["raw_cdf_end"] - 1.0) <= raw_upper_boundary_tolerance
    assert metadata["raw_cdf_max_negative_step"] >= -raw_monotonicity_tolerance
    assert metadata["cdf_upper_tail_clip_tolerance"] == pytest.approx(1e-3)
    assert metadata["cdf_upper_tail_max_excess"] <= raw_upper_boundary_tolerance
    if metadata["cdf_upper_tail_clip_applied"]:
        assert metadata["cdf_upper_tail_clip_count"] > 0
    assert metadata["cdf_pdf_interval_max_error"] <= 1e-2
    assert metadata["cdf_pdf_interval_mean_error"] <= 2e-3


def _build_repriced_single_expiry_chain(
    template_chain,
    market_inputs,
    call_prices,
):
    """Build a same-expiry chain with repriced calls and parity-derived puts.

    Args:
        template_chain: Existing single-expiry option chain.
        market_inputs: Market inputs aligned with the template chain.
        call_prices: New call mid prices ordered by ascending strike.

    Returns:
        pd.DataFrame: Repriced option chain preserving the original expiry.
    """
    call_mask = template_chain["option_type"].astype(str).str.upper().str[0] == "C"
    calls = (
        template_chain.loc[call_mask]
        .sort_values("strike")
        .reset_index(drop=True)
        .copy()
    )
    calls["last_price"] = np.asarray(call_prices, dtype=float)
    calls["bid"] = calls["last_price"] - 0.25
    calls["ask"] = calls["last_price"] + 0.25

    expiry = pd.Timestamp(calls["expiry"].iloc[0])
    valuation_date = pd.Timestamp(market_inputs.valuation_date)
    time_to_expiry_years = (expiry - valuation_date).days / 365.0
    discount_factor = np.exp(-market_inputs.risk_free_rate * time_to_expiry_years)

    puts = calls.copy()
    puts["option_type"] = "P"
    for price_column in ("last_price", "bid", "ask"):
        puts[price_column] = (
            calls[price_column]
            - market_inputs.underlying_price
            + calls["strike"] * discount_factor
        ).abs()

    return pd.concat([calls, puts], ignore_index=True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def single_expiry_chain():
    """Single-expiry chain with calls and puts for parity."""
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

    S, r, T = 100.0, 0.05, 79 / 365.0
    df = np.exp(-r * T)
    puts = calls.copy()
    puts["option_type"] = "P"
    puts["last_price"] = (calls["last_price"] - S + calls["strike"] * df).abs()
    puts["bid"] = (calls["bid"] - S + calls["strike"] * df).abs()
    puts["ask"] = (calls["ask"] - S + calls["strike"] * df).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def market_inputs():
    """Standard MarketInputs for probability tests."""
    from oipd import MarketInputs

    return MarketInputs(
        valuation_date=date(2025, 1, 1),
        underlying_price=100.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )


@pytest.fixture
def same_day_intraday_chain():
    """Single-expiry chain with an expiry later on the valuation day."""
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    expiry = pd.Timestamp("2025-01-01 16:00:00")

    calls = pd.DataFrame(
        {
            "strike": strikes,
            "last_price": [20.75, 16.0, 11.25, 6.75, 3.0, 1.05, 0.4, 0.15, 0.08],
            "bid": [20.5, 15.8, 11.0, 6.5, 2.8, 0.9, 0.3, 0.1, 0.05],
            "ask": [21.0, 16.2, 11.5, 7.0, 3.2, 1.2, 0.5, 0.2, 0.1],
            "option_type": ["call"] * len(strikes),
            "expiry": [expiry] * len(strikes),
        }
    )

    spot = 100.0
    rate = 0.05
    time_to_expiry_years = 6.5 / (24.0 * 365.0)
    discount_factor = np.exp(-rate * time_to_expiry_years)

    puts = calls.copy()
    puts["option_type"] = "put"
    puts["last_price"] = (
        calls["last_price"] - spot + calls["strike"] * discount_factor
    ).abs()
    puts["bid"] = (calls["bid"] - spot + calls["strike"] * discount_factor).abs()
    puts["ask"] = (calls["ask"] - spot + calls["strike"] * discount_factor).abs()

    return pd.concat([calls, puts], ignore_index=True)


@pytest.fixture
def timestamp_dividend_schedules():
    """Same-day dividend schedules straddling the valuation timestamp."""
    return {
        "before": pd.DataFrame(
            {"ex_date": [pd.Timestamp("2025-01-01 08:00:00")], "amount": [1.0]}
        ),
        "after": pd.DataFrame(
            {"ex_date": [pd.Timestamp("2025-01-01 12:00:00")], "amount": [1.0]}
        ),
    }


@pytest.fixture
def fitted_vol_curve(single_expiry_chain, market_inputs):
    """A pre-fitted VolCurve for deriving probability."""
    from oipd import VolCurve

    vc = VolCurve(pricing_engine="bs")
    vc.fit(single_expiry_chain, market_inputs)
    return vc


@pytest.fixture
def prob_curve(fitted_vol_curve):
    """A ProbCurve derived from a fitted VolCurve."""
    return fitted_vol_curve.implied_distribution()


# =============================================================================
# ProbCurve Lazy Materialization Tests
# =============================================================================


class TestProbCurveLazyMaterialization:
    """Characterization tests for lazy ProbCurve materialization."""

    def test_implied_distribution_waits_until_probability_access(
        self,
        fitted_vol_curve,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """VolCurve.implied_distribution() should not materialize arrays eagerly."""
        import oipd.interface.probability as probability_interface

        materialization_calls = 0
        original_materializer = (
            probability_interface.materialize_distribution_from_definition
        )

        def counting_materializer(*args, **kwargs):
            """Count native materializations while preserving behavior."""
            nonlocal materialization_calls
            materialization_calls += 1
            return original_materializer(*args, **kwargs)

        monkeypatch.setattr(
            probability_interface,
            "materialize_distribution_from_definition",
            counting_materializer,
        )

        prob_curve = fitted_vol_curve.implied_distribution()
        assert materialization_calls == 0

        _ = prob_curve.resolved_market
        assert materialization_calls == 0

        _ = prob_curve.pdf(100.0)
        assert materialization_calls == 1

        _ = prob_curve.pdf_values
        _ = prob_curve.cdf_values
        _ = prob_curve.density_results()
        assert materialization_calls == 1

    def test_lazy_probcurve_matches_direct_pipeline_results(self, fitted_vol_curve):
        """Lazy ProbCurve arrays and queries should match direct derivation."""
        expected_prices, expected_pdf, expected_cdf, _ = (
            _derive_direct_probability_arrays(fitted_vol_curve)
        )
        prob_curve = fitted_vol_curve.implied_distribution()

        np.testing.assert_allclose(prob_curve.prices, expected_prices)
        np.testing.assert_allclose(prob_curve.pdf_values, expected_pdf)
        np.testing.assert_allclose(prob_curve.cdf_values, expected_cdf)

        sample_indices = np.linspace(0, len(expected_prices) - 1, num=5, dtype=int)
        sample_prices = expected_prices[sample_indices]
        np.testing.assert_allclose(
            prob_curve.pdf(sample_prices),
            expected_pdf[sample_indices],
        )
        np.testing.assert_allclose(
            [prob_curve.prob_below(float(price)) for price in sample_prices],
            np.interp(
                sample_prices, expected_prices, np.maximum.accumulate(expected_cdf)
            ),
        )
        assert prob_curve.quantile(0.1) == pytest.approx(
            quantile_from_cdf(expected_prices, expected_cdf, 0.1)
        )
        assert prob_curve.quantile(0.5) == pytest.approx(
            quantile_from_cdf(expected_prices, expected_cdf, 0.5)
        )
        assert prob_curve.quantile(0.9) == pytest.approx(
            quantile_from_cdf(expected_prices, expected_cdf, 0.9)
        )
        assert prob_curve.mean() == pytest.approx(
            np.trapezoid(expected_prices * expected_pdf, expected_prices)
        )
        expected_mean = np.trapezoid(expected_prices * expected_pdf, expected_prices)
        assert prob_curve.variance() == pytest.approx(
            np.trapezoid(
                ((expected_prices - expected_mean) ** 2) * expected_pdf,
                expected_prices,
            )
        )

    def test_default_implied_distribution_uses_smart_native_grid(
        self,
        fitted_vol_curve,
    ):
        """Default probability materialization should use the auto grid policy."""
        prob_curve = fitted_vol_curve.implied_distribution()

        assert prob_curve.metadata["native_grid_policy"] == "auto"
        assert len(prob_curve.prices) == prob_curve.metadata["native_grid_points"]
        assert len(prob_curve.prices) != 200
        assert np.all(np.isfinite(prob_curve.prices))
        assert np.all(np.isfinite(prob_curve.pdf_values))
        assert np.all(np.isfinite(prob_curve.cdf_values))
        assert np.all(np.diff(prob_curve.cdf_values) >= -1e-6)

    def test_implied_distribution_fixed_grid_points_preserves_resolution(
        self,
        fitted_vol_curve,
    ):
        """Explicit grid_points should bypass auto sizing and keep fixed length."""
        expected_prices, expected_pdf, expected_cdf, expected_metadata = (
            _derive_direct_probability_arrays(fitted_vol_curve, points=400)
        )

        prob_curve = fitted_vol_curve.implied_distribution(grid_points=400)

        assert prob_curve.metadata["native_grid_policy"] == "fixed"
        assert prob_curve.metadata["native_grid_points"] == 400
        assert len(prob_curve.prices) == 400
        np.testing.assert_allclose(prob_curve.prices, expected_prices)
        np.testing.assert_allclose(prob_curve.pdf_values, expected_pdf)
        np.testing.assert_allclose(prob_curve.cdf_values, expected_cdf)
        assert expected_metadata["native_grid_policy"] == "fixed"

    @pytest.mark.parametrize(
        "bad_grid_points",
        [True, False, 4, 0, -1, 4.5, "400", object()],
    )
    def test_implied_distribution_rejects_invalid_grid_points_immediately(
        self,
        fitted_vol_curve,
        bad_grid_points,
    ):
        """VolCurve should reject invalid grid_points before returning ProbCurve."""
        with pytest.raises(
            ValueError,
            match="grid_points must be at least 5 for finite differences",
        ):
            fitted_vol_curve.implied_distribution(grid_points=bad_grid_points)

    def test_probcurve_snapshot_survives_parent_volcurve_refit(
        self,
        single_expiry_chain,
        market_inputs,
    ):
        """Existing ProbCurve should not change after its source VolCurve is refit."""
        from oipd import VolCurve

        vol_curve = VolCurve(pricing_engine="bs")
        vol_curve.fit(single_expiry_chain, market_inputs)
        expected_prices, expected_pdf, expected_cdf, _ = (
            _derive_direct_probability_arrays(vol_curve)
        )

        prob_curve = vol_curve.implied_distribution()

        refit_chain = _build_repriced_single_expiry_chain(
            single_expiry_chain,
            market_inputs,
            call_prices=[13.5, 9.4, 6.2, 3.8, 2.0],
        )
        vol_curve.fit(refit_chain, market_inputs)
        _, refit_pdf, _, _ = _derive_direct_probability_arrays(vol_curve)
        assert not np.allclose(refit_pdf, expected_pdf, rtol=1e-8, atol=1e-10)

        np.testing.assert_allclose(prob_curve.prices, expected_prices)
        np.testing.assert_allclose(prob_curve.pdf_values, expected_pdf)
        np.testing.assert_allclose(prob_curve.cdf_values, expected_cdf)

    def test_domain_export_before_native_access_is_downstream_resampling_only(
        self,
        fitted_vol_curve,
    ):
        """Explicit export domains should not replace the lazy native support."""
        expected_prices, expected_pdf, expected_cdf, _ = (
            _derive_direct_probability_arrays(fitted_vol_curve)
        )
        prob_curve = fitted_vol_curve.implied_distribution()

        result = prob_curve.density_results(domain=(85.0, 115.0), points=11)

        assert len(result) == 11
        assert result["price"].iloc[0] == pytest.approx(85.0)
        assert result["price"].iloc[-1] == pytest.approx(115.0)
        np.testing.assert_allclose(prob_curve.prices, expected_prices)
        np.testing.assert_allclose(prob_curve.pdf_values, expected_pdf)
        np.testing.assert_allclose(prob_curve.cdf_values, expected_cdf)


# =============================================================================
# ProbCurve.from_chain() Tests
# =============================================================================


class TestProbCurveFromChain:
    """Tests for ProbCurve.from_chain() constructor."""

    def test_from_chain_returns_probcurve(self, single_expiry_chain, market_inputs):
        """from_chain() returns a ProbCurve."""
        from oipd import ProbCurve

        prob = ProbCurve.from_chain(single_expiry_chain, market_inputs)
        assert isinstance(prob, ProbCurve)

    def test_from_chain_accepts_max_staleness(self, single_expiry_chain, market_inputs):
        """from_chain() accepts max_staleness_days."""
        from oipd import ProbCurve

        prob = ProbCurve.from_chain(
            single_expiry_chain,
            market_inputs,
            max_staleness_days=1,
        )
        assert isinstance(prob, ProbCurve)

    def test_from_chain_rejects_multiple_expiries(
        self, single_expiry_chain, market_inputs
    ):
        """from_chain() raises on multi-expiry input."""
        from oipd import ProbCurve

        bad_chain = single_expiry_chain.copy()
        bad_chain.loc[bad_chain.index[:5], "expiry"] = pd.Timestamp("2025-04-21")
        with pytest.raises(ValueError, match="single expiry"):
            ProbCurve.from_chain(bad_chain, market_inputs)


# =============================================================================
# ProbCurve Basic Properties Tests
# =============================================================================


class TestProbCurveProperties:
    """Tests for ProbCurve basic properties."""

    def test_prices_is_numpy_array(self, prob_curve):
        """prices property returns numpy array."""
        assert isinstance(prob_curve.prices, np.ndarray)

    def test_pdf_is_numpy_array(self, prob_curve):
        """pdf_values property returns numpy array."""
        assert isinstance(prob_curve.pdf_values, np.ndarray)

    def test_cdf_is_numpy_array(self, prob_curve):
        """cdf_values property returns numpy array."""
        assert isinstance(prob_curve.cdf_values, np.ndarray)

    def test_pdf_is_non_negative(self, prob_curve):
        """PDF values are non-negative."""
        assert np.all(prob_curve.pdf_values >= 0)

    def test_cdf_is_monotonic(self, prob_curve):
        """CDF is monotonically increasing."""
        cdf = prob_curve.cdf_values
        assert np.all(np.diff(cdf) >= -1e-10)  # Allow tiny numerical noise

    def test_cdf_ends_near_one(self, prob_curve):
        """CDF approaches 1 at high prices."""
        # Check that the CDF converges to 1.0 for a very high price
        # Using functional access allows checking far OTM
        assert prob_curve.prob_below(500.0) > 0.99
        assert prob_curve.prob_below(500.0) <= 1.01

    def test_metadata_has_core_fields(self, prob_curve):
        """metadata exposes core calibration fields."""
        metadata = prob_curve.metadata
        assert isinstance(metadata, dict)
        for key in (
            "expiry",
            "time_to_expiry_years",
            "time_to_expiry_days",
            "forward_price",
            "at_money_vol",
            "resolved_domain",
            "observed_domain",
            "tail_mass_beyond_upper",
            "raw_observed_domain",
            "post_iv_survival_domain",
            "added_mass_last_expansion",
            "domain_grid_spacing",
            "native_grid_policy",
            "native_grid_points",
            "native_grid_min_points",
            "native_grid_max_points",
            "native_grid_target_step",
            "native_grid_actual_step",
            "native_grid_reference_price",
            "native_grid_observed_gap",
            "native_grid_hit_min_cap",
            "native_grid_hit_max_cap",
            "native_grid_domain_width",
            "default_view_domain",
            "default_view_domain_source",
            "default_view_quantiles",
            "cdf_method",
            "cdf_cleanup_policy",
            "raw_cdf_start",
            "raw_cdf_end",
            "raw_cdf_min",
            "raw_cdf_max",
            "raw_cdf_is_monotone",
            "raw_cdf_negative_step_count",
            "raw_cdf_max_negative_step",
            "raw_cdf_below_zero_count",
            "raw_cdf_above_one_count",
            "raw_cdf_points",
            "cdf_left_endpoint_snapped",
            "cdf_right_endpoint_snapped",
            "cdf_lower_clip_count",
            "cdf_upper_clip_count",
            "cdf_upper_tail_clip_policy",
            "cdf_upper_tail_clip_applied",
            "cdf_upper_tail_clip_tolerance",
            "cdf_upper_tail_max_excess",
            "cdf_upper_tail_clip_count",
            "cdf_pdf_interval_max_error",
            "cdf_pdf_interval_mean_error",
        ):
            assert key in metadata
        assert np.isfinite(metadata["at_money_vol"])
        assert metadata["resolved_domain"][0] > 0.0
        assert metadata["resolved_domain"][1] > metadata["resolved_domain"][0]

        resolved = resolve_maturity(
            metadata["expiry"],
            prob_curve.resolved_market.valuation_timestamp,
        )
        assert metadata["time_to_expiry_years"] == pytest.approx(
            resolved.time_to_expiry_years
        )
        assert metadata["time_to_expiry_days"] == pytest.approx(
            resolved.time_to_expiry_days
        )

        assert metadata["expiry"] == resolved.expiry
        assert metadata["calendar_days_to_expiry"] == resolved.calendar_days_to_expiry
        _assert_direct_cdf_diagnostics_pass(metadata)

    def test_default_view_domain_metadata_is_native_bounded_and_inclusive(
        self,
        prob_curve,
    ):
        """Default view domain metadata should include market anchors."""
        metadata = prob_curve.metadata
        native_low, native_high = metadata["resolved_domain"]
        view_low, view_high = metadata["default_view_domain"]

        assert metadata["default_view_domain_source"] == (
            "observed_plus_0.1pct_99.9pct_quantiles"
        )
        assert metadata["default_view_quantiles"] == (0.001, 0.999)
        assert native_low <= view_low < view_high <= native_high

        observed_domain = metadata.get("observed_domain")
        if observed_domain is not None:
            assert view_low <= observed_domain[0]
            assert view_high >= observed_domain[1]

        for anchor in (
            prob_curve.resolved_market.underlying_price,
            metadata.get("forward_price"),
        ):
            if anchor is None:
                continue
            anchor_value = float(anchor)
            if native_low <= anchor_value <= native_high:
                assert view_low <= anchor_value <= view_high

    def test_canonical_cdf_uses_direct_first_derivative(self, prob_curve):
        """Returned CDF values should use the direct first-derivative method."""
        assert prob_curve.metadata["cdf_method"] == "call_price_first_derivative"
        assert prob_curve.metadata["cdf_cleanup_policy"] == "minimal_epsilon_cleanup"
        assert np.all(np.isfinite(prob_curve.cdf_values))
        assert np.all((0.0 <= prob_curve.cdf_values) & (prob_curve.cdf_values <= 1.0))
        assert np.all(np.diff(prob_curve.cdf_values) >= -1e-6)

    def test_aapl_curve_direct_cdf_diagnostics_pass_thresholds(self):
        """Included AAPL data should pass direct-CDF diagnostic thresholds."""
        from oipd import MarketInputs, VolCurve

        data_path = Path(__file__).resolve().parents[2] / "data" / "AAPL_data.csv"
        aapl_chain = pd.read_csv(data_path)
        df_slice = aapl_chain[aapl_chain["expiration"] == "2026-01-16"]
        market = MarketInputs(
            valuation_date=date(2025, 10, 6),
            risk_free_rate=0.045,
            underlying_price=220.0,
        )
        column_mapping = {
            "strike": "strike",
            "last_price": "last_price",
            "type": "option_type",
            "bid": "bid",
            "ask": "ask",
            "expiration": "expiry",
        }

        vol_curve = VolCurve(method="svi")
        vol_curve.method_options = {"random_seed": 42}
        vol_curve.fit(df_slice, market, column_mapping=column_mapping)
        expected_prices, expected_pdf, expected_cdf, expected_metadata = (
            _derive_direct_probability_arrays(vol_curve)
        )
        prob_curve = vol_curve.implied_distribution()

        _assert_direct_cdf_diagnostics_pass(prob_curve.metadata)
        _assert_direct_cdf_diagnostics_pass(expected_metadata)
        np.testing.assert_allclose(prob_curve.prices, expected_prices)
        np.testing.assert_allclose(prob_curve.pdf_values, expected_pdf)
        np.testing.assert_allclose(prob_curve.cdf_values, expected_cdf)

    def test_resolved_market_available(self, prob_curve):
        """resolved_market is available on ProbCurve."""
        resolved_market = prob_curve.resolved_market
        assert resolved_market is not None
        assert hasattr(resolved_market, "valuation_date")

    def test_metadata_keeps_same_day_intraday_expiry(self, same_day_intraday_chain):
        """Derived probability metadata should preserve sub-day maturity inputs."""
        from oipd import MarketInputs, VolCurve

        market_intraday = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
        )

        curve = VolCurve(pricing_engine="bs")
        curve.fit(same_day_intraday_chain, market_intraday)
        prob_curve = curve.implied_distribution()

        assert prob_curve.metadata["expiry"] == pd.Timestamp("2025-01-01 16:00:00")
        assert prob_curve.metadata["time_to_expiry_days"] == pytest.approx(6.5 / 24.0)
        assert prob_curve.metadata["calendar_days_to_expiry"] == 0
        assert prob_curve.metadata["time_to_expiry_years"] == pytest.approx(
            6.5 / (24.0 * 365.0)
        )

    def test_same_day_intraday_distribution_queries_are_finite(
        self, same_day_intraday_chain
    ):
        """Same-day intraday probability curves should produce stable numeric outputs."""
        from oipd import MarketInputs, VolCurve

        market_intraday = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
        )

        curve = VolCurve(pricing_engine="bs")
        curve.fit(same_day_intraday_chain, market_intraday)
        prob_curve = curve.implied_distribution()

        pdf_value = prob_curve.pdf(100.0)
        cdf_value = prob_curve.prob_below(100.0)
        median = prob_curve.quantile(0.5)
        export = prob_curve.density_results()

        assert np.isfinite(pdf_value)
        assert np.isfinite(cdf_value)
        assert np.isfinite(median)
        assert 0.0 <= cdf_value <= 1.0
        assert np.all(np.diff(export["price"].to_numpy(dtype=float)) > 0.0)
        assert np.all(np.isfinite(export["pdf"].to_numpy(dtype=float)))
        assert np.all(np.isfinite(export["cdf"].to_numpy(dtype=float)))

    def test_bs_distribution_accepts_timestamp_dividend_schedule(
        self, same_day_intraday_chain, timestamp_dividend_schedules
    ):
        """BS probability curves should support timestamped dividend schedules."""
        from oipd import MarketInputs, VolCurve

        market_intraday = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=timestamp_dividend_schedules["after"],
        )

        prob_curve = (
            VolCurve(pricing_engine="bs")
            .fit(same_day_intraday_chain, market_intraday)
            .implied_distribution()
        )

        assert np.isfinite(prob_curve.pdf(100.0))
        assert np.isfinite(prob_curve.prob_below(100.0))

    def test_bs_distribution_distinguishes_same_day_dividend_timing(
        self, same_day_intraday_chain, timestamp_dividend_schedules
    ):
        """BS distributions should change when dividend timing crosses valuation."""
        from oipd import MarketInputs, VolCurve

        market_before = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=timestamp_dividend_schedules["before"],
        )
        market_after = MarketInputs(
            valuation_date=pd.Timestamp("2025-01-01 09:30:00"),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=timestamp_dividend_schedules["after"],
        )

        prob_before = (
            VolCurve(pricing_engine="bs")
            .fit(same_day_intraday_chain, market_before)
            .implied_distribution()
        )
        prob_after = (
            VolCurve(pricing_engine="bs")
            .fit(same_day_intraday_chain, market_after)
            .implied_distribution()
        )

        assert prob_before.pdf(100.0) != pytest.approx(prob_after.pdf(100.0))

    def test_bs_distribution_ignores_post_expiry_dividends(self, single_expiry_chain):
        """BS distributions should ignore discrete dividends after the curve expiry."""
        from oipd import MarketInputs, VolCurve

        full_schedule = pd.DataFrame(
            {
                "ex_date": [
                    pd.Timestamp("2025-02-15 00:00:00"),
                    pd.Timestamp("2025-04-15 00:00:00"),
                ],
                "amount": [1.0, 1.5],
            }
        )
        early_only_schedule = pd.DataFrame(
            {
                "ex_date": [pd.Timestamp("2025-02-15 00:00:00")],
                "amount": [1.0],
            }
        )

        market_full = MarketInputs(
            valuation_date=date(2025, 1, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=full_schedule,
        )
        market_early_only = MarketInputs(
            valuation_date=date(2025, 1, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
            dividend_schedule=early_only_schedule,
        )

        prob_full = (
            VolCurve(pricing_engine="bs")
            .fit(single_expiry_chain, market_full)
            .implied_distribution()
        )
        prob_early_only = (
            VolCurve(pricing_engine="bs")
            .fit(single_expiry_chain, market_early_only)
            .implied_distribution()
        )

        np.testing.assert_allclose(
            prob_full.density_results()["pdf"].to_numpy(dtype=float),
            prob_early_only.density_results()["pdf"].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-10,
        )


# =============================================================================
# ProbCurve.density_results() Tests
# =============================================================================


class TestProbCurveDensityResults:
    """Tests for ProbCurve.density_results() exports."""

    def test_density_results_returns_expected_columns(self, prob_curve):
        """density_results() returns the canonical export schema."""
        result = prob_curve.density_results()
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["price", "pdf", "cdf"]

    def test_density_results_uses_default_view_domain_by_default(self, prob_curve):
        """density_results() defaults to compact 200-row view export."""
        result = prob_curve.density_results()
        view_low, view_high = prob_curve.metadata["default_view_domain"]

        assert len(result) == 200
        assert result["price"].iloc[0] == pytest.approx(view_low)
        assert result["price"].iloc[-1] == pytest.approx(view_high)

    def test_density_results_full_domain_returns_native_arrays(self, prob_curve):
        """full_domain=True exports native arrays exactly when domain is omitted."""
        result = prob_curve.density_results(full_domain=True)

        np.testing.assert_allclose(
            result["price"].to_numpy(dtype=float), prob_curve.prices
        )
        np.testing.assert_allclose(
            result["pdf"].to_numpy(dtype=float),
            prob_curve.pdf_values,
        )
        np.testing.assert_allclose(
            result["cdf"].to_numpy(dtype=float),
            prob_curve.cdf_values,
        )

    def test_native_distribution_uses_full_domain_by_default(self, prob_curve):
        """Native arrays use the resolved full-domain support."""
        assert prob_curve.prices[0] == pytest.approx(0.01)
        assert prob_curve.metadata["resolved_domain"][0] == pytest.approx(0.01)
        assert prob_curve.metadata["resolved_domain"][1] >= prob_curve.prices[-1]

    def test_density_results_resamples_explicit_domain(self, prob_curve):
        """Explicit domain triggers interpolation onto the requested grid."""
        result = prob_curve.density_results(domain=(80.0, 120.0), points=25)
        assert len(result) == 25
        assert np.isclose(result["price"].iloc[0], 80.0)
        assert np.isclose(result["price"].iloc[-1], 120.0)

    def test_density_results_explicit_domain_overrides_full_domain(self, prob_curve):
        """Explicit domain and points should win over full_domain=True."""
        result = prob_curve.density_results(
            domain=(80.0, 120.0),
            points=25,
            full_domain=True,
        )

        assert len(result) == 25
        assert result["price"].iloc[0] == pytest.approx(80.0)
        assert result["price"].iloc[-1] == pytest.approx(120.0)

    def test_density_results_domain_defaults_to_200_points(self, prob_curve):
        """Explicit domain with omitted points defaults to 200 rows."""
        result = prob_curve.density_results(domain=(80.0, 120.0))
        assert len(result) == 200

    def test_density_results_matches_probability_queries(self, prob_curve):
        """Exported values stay consistent with PDF and CDF query methods."""
        result = prob_curve.density_results(domain=(85.0, 115.0), points=11)
        prices = result["price"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            result["pdf"].to_numpy(dtype=float),
            prob_curve.pdf(prices),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            result["cdf"].to_numpy(dtype=float),
            [prob_curve.prob_below(price) for price in prices],
            rtol=1e-8,
        )

    def test_density_results_domain_is_downstream_resampling_only(self, prob_curve):
        """Explicit export domains do not mutate the native cached arrays."""
        native_prices = prob_curve.prices.copy()
        native_pdf = prob_curve.pdf_values.copy()
        native_cdf = prob_curve.cdf_values.copy()

        _ = prob_curve.density_results(domain=(85.0, 115.0), points=11)

        np.testing.assert_allclose(prob_curve.prices, native_prices)
        np.testing.assert_allclose(prob_curve.pdf_values, native_pdf)
        np.testing.assert_allclose(prob_curve.cdf_values, native_cdf)

    def test_density_results_rejects_invalid_domain(self, prob_curve):
        """Invalid domains are rejected at the export boundary."""
        with pytest.raises(ValueError, match="strictly increasing"):
            prob_curve.density_results(domain=(120.0, 80.0))


# =============================================================================
# ProbCurve.plot() Tests
# =============================================================================


class TestProbCurvePlot:
    """Tests for ProbCurve.plot() display-grid behavior."""

    def _capture_plot_rnd(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Capture arguments passed to the presentation plotting helper.

        Args:
            monkeypatch: Pytest monkeypatch fixture.

        Returns:
            dict: Mutable dictionary populated with plot helper keyword args.
        """
        import oipd.interface.probability as probability_interface

        captured: dict = {}

        def fake_plot_rnd(**kwargs):
            """Record plot inputs and return them as a stand-in figure."""
            captured.update(kwargs)
            return captured

        monkeypatch.setattr(probability_interface, "plot_rnd", fake_plot_rnd)
        return captured

    def test_plot_defaults_to_800_point_view_domain(
        self,
        prob_curve,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Default plot should use the compact view domain with 800 points."""
        captured = self._capture_plot_rnd(monkeypatch)

        result = prob_curve.plot()

        view_low, view_high = prob_curve.metadata["default_view_domain"]
        assert result is captured
        assert len(captured["prices"]) == 800
        assert captured["prices"][0] == pytest.approx(view_low)
        assert captured["prices"][-1] == pytest.approx(view_high)
        assert captured["kind"] == "both"

    def test_plot_xlim_uses_requested_domain_and_points(
        self,
        prob_curve,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Explicit xlim should define the plot domain and display resolution."""
        captured = self._capture_plot_rnd(monkeypatch)

        prob_curve.plot(
            kind="pdf",
            xlim=(90.0, 110.0),
            ylim=(0.0, 0.2),
            points=300,
            full_domain=True,
            color="tab:green",
        )

        assert len(captured["prices"]) == 300
        assert captured["prices"][0] == pytest.approx(90.0)
        assert captured["prices"][-1] == pytest.approx(110.0)
        assert captured["xlim"] == (90.0, 110.0)
        assert captured["ylim"] == (0.0, 0.2)
        assert captured["kind"] == "pdf"
        assert captured["color"] == "tab:green"

    def test_plot_full_domain_spans_native_domain(
        self,
        prob_curve,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """full_domain=True should plot the full native probability domain."""
        captured = self._capture_plot_rnd(monkeypatch)

        prob_curve.plot(full_domain=True)

        assert len(captured["prices"]) == 800
        assert captured["prices"][0] == pytest.approx(prob_curve.prices[0])
        assert captured["prices"][-1] == pytest.approx(prob_curve.prices[-1])

    def test_plot_does_not_mutate_native_snapshot(
        self,
        prob_curve,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Plot display resampling should not mutate native arrays or metadata."""
        self._capture_plot_rnd(monkeypatch)
        native_prices = prob_curve.prices.copy()
        native_pdf = prob_curve.pdf_values.copy()
        native_cdf = prob_curve.cdf_values.copy()
        metadata_before = {
            key: prob_curve.metadata.get(key)
            for key in (
                "resolved_domain",
                "default_view_domain",
                "default_view_quantiles",
                "native_grid_points",
                "cdf_method",
            )
        }

        prob_curve.plot(points=500)

        np.testing.assert_allclose(prob_curve.prices, native_prices)
        np.testing.assert_allclose(prob_curve.pdf_values, native_pdf)
        np.testing.assert_allclose(prob_curve.cdf_values, native_cdf)
        assert {
            key: prob_curve.metadata.get(key) for key in metadata_before
        } == metadata_before

    def test_plot_visible_xlim_has_dense_display_grid(
        self,
        prob_curve,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Narrow visible xlim should still receive the requested point count."""
        captured = self._capture_plot_rnd(monkeypatch)

        prob_curve.plot(xlim=(95.0, 105.0), points=300)

        assert len(captured["prices"]) == 300
        assert captured["prices"][0] == pytest.approx(95.0)
        assert captured["prices"][-1] == pytest.approx(105.0)
        assert np.all(np.diff(captured["prices"]) > 0.0)


# =============================================================================
# ProbCurve.prob_below() Tests
# =============================================================================


class TestProbCurveProbBelow:
    """Tests for prob_below() method."""

    def test_prob_below_returns_float(self, prob_curve):
        """prob_below() returns a float."""
        p = prob_curve.prob_below(100.0)
        assert isinstance(p, float)

    def test_prob_below_in_valid_range(self, prob_curve):
        """prob_below() returns value in [0, 1]."""
        p = prob_curve.prob_below(100.0)
        assert 0.0 <= p <= 1.0

    def test_prob_below_increases_with_price(self, prob_curve):
        """prob_below(low) < prob_below(high)."""
        p_low = prob_curve.prob_below(90.0)
        p_high = prob_curve.prob_below(110.0)
        assert p_low < p_high

    def test_prob_below_zero_at_left_edge(self, prob_curve):
        """prob_below() returns ~0 at extreme low price."""
        p = prob_curve.prob_below(1.0)
        assert p < 0.01

    def test_prob_below_one_at_right_edge(self, prob_curve):
        """prob_below() returns ~1 at extreme high price."""
        p = prob_curve.prob_below(1000.0)
        assert p > 0.99


# =============================================================================
# ProbCurve.prob_above() Tests
# =============================================================================


class TestProbCurveProbAbove:
    """Tests for prob_above() method."""

    def test_prob_above_complements_prob_below(self, prob_curve):
        """prob_above(x) + prob_below(x) ≈ 1."""
        price = 100.0
        p_below = prob_curve.prob_below(price)
        p_above = prob_curve.prob_above(price)
        assert np.isclose(p_below + p_above, 1.0)


# =============================================================================
# ProbCurve.prob_between() Tests
# =============================================================================


class TestProbCurveProbBetween:
    """Tests for prob_between() method."""

    def test_prob_between_valid_range(self, prob_curve):
        """prob_between() returns value in [0, 1]."""
        p = prob_curve.prob_between(90.0, 110.0)
        assert 0.0 <= p <= 1.0

    def test_prob_between_equals_cdf_diff(self, prob_curve):
        """prob_between(low, high) = prob_below(high) - prob_below(low)."""
        low, high = 95.0, 105.0
        p_between = prob_curve.prob_between(low, high)
        p_diff = prob_curve.prob_below(high) - prob_curve.prob_below(low)
        assert np.isclose(p_between, p_diff)

    def test_prob_between_raises_on_invalid_range(self, prob_curve):
        """prob_between() raises if low > high."""
        with pytest.raises(ValueError):
            prob_curve.prob_between(110.0, 90.0)


# =============================================================================
# ProbCurve.expected_value() Tests
# =============================================================================


class TestProbCurveMean:
    """Tests for mean() method."""

    def test_mean_returns_float(self, prob_curve):
        """mean() returns a float."""
        ev = prob_curve.mean()
        assert isinstance(ev, float)

    def test_mean_is_positive(self, prob_curve):
        """Mean is positive for stock prices."""
        ev = prob_curve.mean()
        assert ev > 0

    def test_mean_near_forward(self, prob_curve):
        """Mean value should be positive (roughly near spot/forward)."""
        ev = prob_curve.mean()
        # For synthetic data, just check it's positive and reasonable
        assert 0.0 < ev < 500.0  # Rough bounds


# =============================================================================
# ProbCurve.variance() Tests
# =============================================================================


class TestProbCurveVariance:
    """Tests for variance() method."""

    def test_variance_returns_float(self, prob_curve):
        """variance() returns a float."""
        var = prob_curve.variance()
        assert isinstance(var, float)

    def test_variance_is_positive(self, prob_curve):
        """Variance is positive."""
        var = prob_curve.variance()
        assert var > 0


# =============================================================================
# ProbCurve Callable Interface Tests
# =============================================================================


class TestProbCurveCallable:
    """Tests for ProbCurve callable interface."""

    def test_call_returns_pdf_value(self, prob_curve):
        """Calling prob_curve(price) returns PDF value."""
        pdf_val = prob_curve(100.0)
        assert pdf_val >= 0
