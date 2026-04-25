"""
Test Skeleton for ProbSurface Interface
========================================
Tests for the multi-expiry probability surface API.

Based on first-principles analysis of oipd.interface.probability.ProbSurface
"""

import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oipd.pipelines.probability import derive_distribution_from_curve

INTERIOR_INTERPOLATED_EXPIRY = pd.Timestamp("2025-03-21")


def _assert_metadata_value_matches(expected, actual) -> None:
    """Assert that one metadata field matches across two probability slices.

    Args:
        expected: Metadata value from the direct ProbCurve path.
        actual: Metadata value from the ProbSurface.slice(...) path.
    """
    if expected is None or actual is None:
        assert actual == expected
        return

    if isinstance(expected, tuple) or isinstance(actual, tuple):
        assert actual == pytest.approx(expected)
        return

    if isinstance(expected, (int, float, np.integer, np.floating)) and isinstance(
        actual, (int, float, np.integer, np.floating)
    ):
        assert float(actual) == pytest.approx(float(expected))
        return

    assert actual == expected


def _assert_probcurve_contract_matches(
    surface_curve,
    direct_curve,
) -> None:
    """Assert that two probability-slice contracts match.

    Args:
        surface_curve: ProbCurve returned by ``ProbSurface.slice(...)``.
        direct_curve: ProbCurve returned by ``VolSurface.slice(...).implied_distribution()``.
    """
    np.testing.assert_allclose(
        surface_curve.prices,
        direct_curve.prices,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        surface_curve.pdf_values,
        direct_curve.pdf_values,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        surface_curve.cdf_values,
        direct_curve.cdf_values,
        rtol=1e-8,
        atol=1e-10,
    )

    for metadata_key in (
        "resolved_domain",
        "default_view_domain",
        "default_view_domain_source",
        "default_view_quantiles",
        "domain_expansions",
        "tail_mass_beyond_upper",
    ):
        _assert_metadata_value_matches(
            direct_curve.metadata.get(metadata_key),
            surface_curve.metadata.get(metadata_key),
        )
    assert bool(surface_curve.metadata.get("interpolated")) == bool(
        direct_curve.metadata.get("interpolated")
    )


def _assert_direct_cdf_diagnostics_pass(metadata) -> None:
    """Assert canonical direct-CDF metadata satisfies current thresholds.

    Args:
        metadata: Probability metadata emitted during materialization.
    """
    assert metadata["cdf_method"] == "call_price_first_derivative"
    assert metadata["cdf_cleanup_policy"] == "minimal_epsilon_cleanup"
    assert metadata["cdf_violation_policy"] == "warn"
    assert metadata["cdf_monotonicity_repair_tolerance"] == pytest.approx(5e-6)
    assert metadata["cdf_total_negative_variation_tolerance"] == pytest.approx(1e-4)
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
    raw_lower_boundary_tolerance = 1e-4
    raw_upper_boundary_tolerance = metadata["cdf_upper_tail_clip_tolerance"]
    raw_monotonicity_tolerance = 5e-6
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


def _derive_direct_probability_arrays(vol_curve, *, points: int | None = None):
    """Derive probability arrays through the stateless single-expiry pipeline.

    Args:
        vol_curve: Fitted public VolCurve object.
        points: Optional native grid resolution for the direct derivation.

    Returns:
        tuple: Direct ``prices``, ``pdf_values``, ``cdf_values``, and metadata.
    """
    return derive_distribution_from_curve(
        vol_curve._vol_curve,
        vol_curve.resolved_market,
        pricing_engine=vol_curve._pricing_engine,
        vol_metadata=vol_curve._metadata,
        points=points,
    )


def _assert_public_parity_requirement_message(message: str) -> None:
    """Assert public parity errors describe the required option coverage."""
    assert "usable same-strike call/put pairs" in message
    assert "parity-forward inference" in message
    assert "Black-Scholes" not in message
    assert "dividend" not in message.lower()


def _append_call_only_expiry(chain: pd.DataFrame) -> pd.DataFrame:
    """Append one expiry with calls but no same-strike put pairs."""
    call_mask = chain["option_type"].astype(str).str.upper().str[0] == "C"
    bad_calls = chain.loc[call_mask].copy()
    first_expiry = pd.Timestamp(chain["expiry"].iloc[0])
    bad_calls = bad_calls[pd.to_datetime(bad_calls["expiry"]) == first_expiry].copy()
    bad_calls["expiry"] = pd.Timestamp("2025-07-21")
    return pd.concat([chain, bad_calls], ignore_index=True)


def _build_repriced_multi_expiry_chain(
    template_chain,
    market_inputs,
    call_prices_by_expiry,
):
    """Build a multi-expiry chain with repriced calls and parity-derived puts.

    Args:
        template_chain: Existing multi-expiry option chain.
        market_inputs: Market inputs aligned with the template chain.
        call_prices_by_expiry: Mapping from expiry timestamp to call prices
            ordered by ascending strike.

    Returns:
        pd.DataFrame: Repriced multi-expiry option chain.
    """
    call_mask = template_chain["option_type"].astype(str).str.upper().str[0] == "C"
    repriced_calls = []
    for expiry, call_prices in call_prices_by_expiry.items():
        expiry_timestamp = pd.Timestamp(expiry)
        expiry_mask = pd.to_datetime(template_chain["expiry"]) == expiry_timestamp
        calls = (
            template_chain.loc[call_mask & expiry_mask]
            .sort_values("strike")
            .reset_index(drop=True)
            .copy()
        )
        calls["last_price"] = np.asarray(call_prices, dtype=float)
        calls["bid"] = calls["last_price"] - 0.25
        calls["ask"] = calls["last_price"] + 0.25
        repriced_calls.append(calls)

    calls_frame = pd.concat(repriced_calls, ignore_index=True)
    valuation_date = pd.Timestamp(market_inputs.valuation_date)
    time_to_expiry_years = (
        pd.to_datetime(calls_frame["expiry"]) - valuation_date
    ).dt.days / 365.0
    discount_factors = np.exp(-market_inputs.risk_free_rate * time_to_expiry_years)

    puts_frame = calls_frame.copy()
    puts_frame["option_type"] = "P"
    for price_column in ("last_price", "bid", "ask"):
        puts_frame[price_column] = (
            calls_frame[price_column]
            - market_inputs.underlying_price
            + calls_frame["strike"] * discount_factors
        ).abs()

    return pd.concat([calls_frame, puts_frame], ignore_index=True)


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

    vs = VolSurface()
    vs.fit(multi_expiry_chain, market_inputs)
    return vs


@pytest.fixture
def prob_surface(fitted_vol_surface):
    """A ProbSurface derived from a fitted VolSurface."""
    return fitted_vol_surface.implied_distribution()


@pytest.fixture
def prob_surface_with_native_resolution(fitted_vol_surface):
    """A ProbSurface with explicit resolution aligned to direct ProbCurve defaults."""
    from oipd.interface.probability import ProbSurface

    return ProbSurface(vol_surface=fitted_vol_surface, grid_points=200)


# =============================================================================
# ProbSurface Lazy Materialization Tests
# =============================================================================


class TestProbSurfaceLazyMaterialization:
    """Characterization tests for lazy ProbSurface materialization."""

    def test_metadata_access_materializes_once_for_curve_and_surface_slice(
        self,
        multi_expiry_chain,
        market_inputs,
        fitted_vol_surface,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Metadata access should trigger one native materialization per curve."""
        import oipd.interface.probability as probability_interface
        from oipd import VolCurve

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

        first_expiry = multi_expiry_chain["expiry"].iloc[0]
        single_expiry_chain = multi_expiry_chain[
            pd.to_datetime(multi_expiry_chain["expiry"]) == pd.Timestamp(first_expiry)
        ]
        prob_curve = (
            VolCurve().fit(single_expiry_chain, market_inputs).implied_distribution()
        )

        assert materialization_calls == 0
        _ = prob_curve.metadata
        assert materialization_calls == 1
        _ = prob_curve.metadata
        assert materialization_calls == 1

        prob_surface = fitted_vol_surface.implied_distribution()
        surface_curve = prob_surface.slice(prob_surface.expiries[0])
        assert materialization_calls == 1
        _ = surface_curve.metadata
        assert materialization_calls == 2
        _ = surface_curve.metadata
        assert materialization_calls == 2

    def test_implied_distribution_and_slice_wait_until_probability_access(
        self,
        fitted_vol_surface,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """VolSurface and slice creation should not materialize arrays eagerly."""
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

        prob_surface = fitted_vol_surface.implied_distribution()
        assert materialization_calls == 0

        pillar_expiry = prob_surface.expiries[0]
        curve = prob_surface.slice(pillar_expiry)
        assert materialization_calls == 0

        _ = curve.resolved_market
        assert materialization_calls == 0

        _ = curve.pdf_values
        assert materialization_calls == 1

        _ = prob_surface.pdf(100.0, t=pillar_expiry)
        assert materialization_calls == 2

    @pytest.mark.parametrize(
        "expiry",
        [
            pytest.param("pillar", id="pillar"),
            pytest.param(INTERIOR_INTERPOLATED_EXPIRY, id="interpolated"),
        ],
    )
    def test_lazy_surface_slice_matches_direct_pipeline(
        self,
        fitted_vol_surface,
        expiry,
    ):
        """Lazy surface slices should match the direct single-expiry pipeline."""
        prob_surface = fitted_vol_surface.implied_distribution()
        resolved_expiry = (
            fitted_vol_surface.expiries[0] if expiry == "pillar" else expiry
        )

        direct_vol_curve = fitted_vol_surface.slice(resolved_expiry)
        expected_prices, expected_pdf, expected_cdf, _ = (
            _derive_direct_probability_arrays(direct_vol_curve)
        )
        surface_curve = prob_surface.slice(resolved_expiry)

        np.testing.assert_allclose(surface_curve.prices, expected_prices)
        np.testing.assert_allclose(surface_curve.pdf_values, expected_pdf)
        np.testing.assert_allclose(surface_curve.cdf_values, expected_cdf)

    def test_implied_distribution_propagates_raise_cdf_policy_to_interpolated_slice(
        self,
        fitted_vol_surface,
    ):
        """VolSurface.implied_distribution() should carry policy to slices."""
        prob_surface = fitted_vol_surface.implied_distribution(
            cdf_violation_policy="raise",
        )
        surface_curve = prob_surface.slice(INTERIOR_INTERPOLATED_EXPIRY)

        assert surface_curve.metadata["cdf_violation_policy"] == "raise"

    def test_probsurface_constructor_propagates_raise_cdf_policy(
        self,
        fitted_vol_surface,
    ):
        """Direct ProbSurface construction should carry policy to slices."""
        from oipd.interface.probability import ProbSurface

        prob_surface = ProbSurface(
            vol_surface=fitted_vol_surface,
            cdf_violation_policy="raise",
        )
        surface_curve = prob_surface.slice(prob_surface.expiries[0])

        assert surface_curve.metadata["cdf_violation_policy"] == "raise"

    @pytest.mark.parametrize(
        "expiry",
        [
            pytest.param("pillar", id="aapl-pillar"),
            pytest.param(pd.Timestamp("2026-06-18"), id="aapl-june-pillar"),
            pytest.param(pd.Timestamp("2026-12-18"), id="aapl-december-pillar"),
            pytest.param(pd.Timestamp("2026-04-17"), id="aapl-interpolated"),
        ],
    )
    def test_lazy_surface_slice_matches_direct_pipeline_for_aapl_chain(self, expiry):
        """Lazy surface slices should match direct derivation on included AAPL data."""
        from oipd import MarketInputs, VolSurface

        data_path = Path(__file__).resolve().parents[2] / "data" / "AAPL_data.csv"
        aapl_chain = pd.read_csv(data_path)
        selected_expiries = ["2026-01-16", "2026-06-18", "2026-12-18"]
        surface_chain = aapl_chain[
            aapl_chain["expiration"].isin(selected_expiries)
        ].copy()

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

        vol_surface = VolSurface(method="svi")
        vol_surface.method_options = {"random_seed": 42}
        vol_surface.fit(
            surface_chain,
            market,
            column_mapping=column_mapping,
            failure_policy="raise",
        )
        prob_surface = vol_surface.implied_distribution()
        resolved_expiry = vol_surface.expiries[0] if expiry == "pillar" else expiry

        direct_vol_curve = vol_surface.slice(resolved_expiry)
        expected_prices, expected_pdf, expected_cdf, _ = (
            _derive_direct_probability_arrays(direct_vol_curve)
        )
        surface_curve = prob_surface.slice(resolved_expiry)

        np.testing.assert_allclose(surface_curve.prices, expected_prices)
        np.testing.assert_allclose(surface_curve.pdf_values, expected_pdf)
        np.testing.assert_allclose(surface_curve.cdf_values, expected_cdf)
        _assert_direct_cdf_diagnostics_pass(surface_curve.metadata)

    def test_probsurface_snapshot_survives_parent_volsurface_refit(
        self,
        multi_expiry_chain,
        market_inputs,
    ):
        """Existing ProbSurface should not change after its source VolSurface is refit."""
        from oipd import VolSurface

        vol_surface = VolSurface()
        vol_surface.fit(multi_expiry_chain, market_inputs)
        prob_surface = vol_surface.implied_distribution()

        pillar_expiry = vol_surface.expiries[0]
        expected_by_expiry = {}
        for expiry in (pillar_expiry, INTERIOR_INTERPOLATED_EXPIRY):
            direct_vol_curve = vol_surface.slice(expiry)
            expected_by_expiry[pd.Timestamp(expiry)] = (
                _derive_direct_probability_arrays(direct_vol_curve)
            )

        refit_chain = _build_repriced_multi_expiry_chain(
            multi_expiry_chain,
            market_inputs,
            call_prices_by_expiry={
                pd.Timestamp("2025-02-21"): [13.2, 9.2, 6.0, 3.7, 2.1],
                pd.Timestamp("2025-05-21"): [15.5, 11.2, 8.0, 5.4, 3.2],
            },
        )
        vol_surface.fit(refit_chain, market_inputs)
        _, refit_pdf, _, _ = _derive_direct_probability_arrays(
            vol_surface.slice(pillar_expiry)
        )
        _, original_pillar_pdf, _, _ = expected_by_expiry[pillar_expiry]
        assert not np.allclose(refit_pdf, original_pillar_pdf, rtol=1e-8, atol=1e-10)

        for expiry, (
            expected_prices,
            expected_pdf,
            expected_cdf,
            _,
        ) in expected_by_expiry.items():
            surface_curve = prob_surface.slice(expiry)
            np.testing.assert_allclose(surface_curve.prices, expected_prices)
            np.testing.assert_allclose(surface_curve.pdf_values, expected_pdf)
            np.testing.assert_allclose(surface_curve.cdf_values, expected_cdf)


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

    def test_from_chain_propagates_raise_cdf_policy(
        self, multi_expiry_chain, market_inputs
    ):
        """from_chain() should carry the requested CDF policy to slices."""
        from oipd import ProbSurface

        prob = ProbSurface.from_chain(
            multi_expiry_chain,
            market_inputs,
            cdf_violation_policy="raise",
        )

        curve = prob.slice(prob.expiries[0])
        assert curve.metadata["cdf_violation_policy"] == "raise"

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
        from oipd.warnings import WorkflowWarning

        with pytest.warns(
            WorkflowWarning,
            match=r"VolSurface\.fit recorded 1 workflow warning event",
        ):
            prob = ProbSurface.from_chain(
                mixed_quality_multi_expiry_chain, market_inputs
            )
        assert isinstance(prob, ProbSurface)
        assert len(prob.expiries) == 2
        assert pd.Timestamp("2025-07-21") not in prob.expiries

    def test_from_chain_skip_warn_records_missing_parity_pairs_in_diagnostics(
        self, multi_expiry_chain, market_inputs
    ):
        """Skipped probability-surface expiries should expose parity requirements."""
        from oipd import ProbSurface
        from oipd.warnings import WorkflowWarning

        chain = _append_call_only_expiry(multi_expiry_chain)

        with pytest.warns(
            WorkflowWarning,
            match=r"VolSurface\.fit recorded 1 workflow warning event",
        ):
            prob = ProbSurface.from_chain(chain, market_inputs)

        skipped_events = [
            event
            for event in prob.warning_diagnostics.events
            if event.event_type == "skipped_expiry"
        ]
        assert len(skipped_events) == 1
        _assert_public_parity_requirement_message(
            str(skipped_events[0].details["reason"])
        )

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

    def test_from_chain_raise_policy_missing_parity_pairs_uses_public_requirement(
        self, multi_expiry_chain, market_inputs
    ):
        """Strict probability-surface fits should expose parity requirements."""
        from oipd import ProbSurface
        from oipd.core.errors import CalculationError

        chain = _append_call_only_expiry(multi_expiry_chain)
        bad_expiry = pd.Timestamp("2025-07-21")
        keep_expiries = [multi_expiry_chain["expiry"].iloc[0], bad_expiry]
        chain = chain[pd.to_datetime(chain["expiry"]).isin(keep_expiries)]

        with pytest.raises(CalculationError) as exc_info:
            ProbSurface.from_chain(
                chain,
                market_inputs,
                failure_policy="raise",
            )

        message = str(exc_info.value)
        _assert_public_parity_requirement_message(message)
        assert "failure_policy='skip_warn'" in message

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
        assert "expiry" in metadata
        assert "forward_price" in metadata
        assert "at_money_vol" in metadata
        assert np.isfinite(metadata["at_money_vol"])
        resolved_market = curve.resolved_market
        assert resolved_market is not None
        assert hasattr(resolved_market, "valuation_date")

    @pytest.mark.parametrize(
        "expiry_kind, expected_source",
        [
            pytest.param("pillar", "put_call_parity", id="pillar"),
            pytest.param(
                "interpolated",
                "surface_interpolation",
                id="interpolated",
            ),
        ],
    )
    def test_slice_metadata_exposes_forward_source_lineage(
        self,
        prob_surface,
        expiry_kind,
        expected_source,
    ):
        """Probability slices should preserve volatility forward diagnostics."""
        expiry = (
            prob_surface.expiries[0]
            if expiry_kind == "pillar"
            else INTERIOR_INTERPOLATED_EXPIRY
        )

        metadata = prob_surface.slice(expiry).metadata

        assert metadata["forward_price_source"] == expected_source
        if expiry_kind == "pillar":
            parity_report = metadata["parity_report"]
            assert parity_report["confidence"] in {
                "robust",
                "low_two_pairs",
                "low_single_pair",
            }
            assert parity_report["valid_pair_count"] >= 1
            assert "outlier_count" in parity_report
            assert "quote_liquidity_confidence" in parity_report
            return

        source_forward_pillars = metadata["source_forward_pillars"]
        assert len(source_forward_pillars) == 2
        assert all(
            pillar["forward_price_source"] == "put_call_parity"
            for pillar in source_forward_pillars
        )
        assert all("parity_report" in pillar for pillar in source_forward_pillars)

    def test_slice_matches_direct_probcurve_for_pillar_expiry(
        self, fitted_vol_surface, prob_surface_with_native_resolution
    ):
        """Pillar slices should match the direct VolCurve -> ProbCurve path."""
        pillar_expiry = prob_surface_with_native_resolution.expiries[0]

        surface_curve = prob_surface_with_native_resolution.slice(pillar_expiry)
        direct_curve = fitted_vol_surface.slice(pillar_expiry).implied_distribution(
            grid_points=200,
        )

        _assert_probcurve_contract_matches(surface_curve, direct_curve)

    @pytest.mark.parametrize("expiry_kind", ["pillar", "interpolated"])
    def test_default_slice_matches_direct_probcurve_path(
        self,
        fitted_vol_surface,
        prob_surface,
        expiry_kind,
    ):
        """Default ProbSurface slices should match the direct public ProbCurve path."""
        resolved_expiry = (
            prob_surface.expiries[0]
            if expiry_kind == "pillar"
            else INTERIOR_INTERPOLATED_EXPIRY
        )
        surface_curve = prob_surface.slice(resolved_expiry)
        direct_curve = fitted_vol_surface.slice(resolved_expiry).implied_distribution()

        _assert_probcurve_contract_matches(surface_curve, direct_curve)

    def test_slice_interpolates_interior_expiry(self, prob_surface):
        """slice() interpolates when expiry lies between fitted pillars."""
        from oipd.interface.probability import ProbCurve

        interpolated_expiry = INTERIOR_INTERPOLATED_EXPIRY
        curve = prob_surface.slice(interpolated_expiry)
        assert isinstance(curve, ProbCurve)
        assert bool(curve.metadata.get("interpolated"))

    def test_slice_matches_direct_probcurve_for_interpolated_expiry(
        self, fitted_vol_surface, prob_surface_with_native_resolution
    ):
        """Interpolated slices should match the direct VolCurve -> ProbCurve path."""
        surface_curve = prob_surface_with_native_resolution.slice(
            INTERIOR_INTERPOLATED_EXPIRY
        )
        direct_curve = fitted_vol_surface.slice(
            INTERIOR_INTERPOLATED_EXPIRY
        ).implied_distribution(grid_points=200)

        _assert_probcurve_contract_matches(surface_curve, direct_curve)

    def test_slice_propagates_domain_provenance_for_pillar(
        self, fitted_vol_surface, prob_surface_with_native_resolution
    ):
        """Pillar slices should preserve direct-path domain provenance metadata."""
        pillar_expiry = prob_surface_with_native_resolution.expiries[0]

        surface_curve = prob_surface_with_native_resolution.slice(pillar_expiry)
        direct_curve = fitted_vol_surface.slice(pillar_expiry).implied_distribution(
            grid_points=200,
        )

        for metadata_key in (
            "raw_observed_domain",
            "post_iv_survival_domain",
            "observed_domain",
            "resolved_domain",
            "domain_expansions",
            "tail_mass_beyond_upper",
        ):
            _assert_metadata_value_matches(
                direct_curve.metadata.get(metadata_key),
                surface_curve.metadata.get(metadata_key),
            )

    def test_slice_matches_surface_queries(self, prob_surface, market_inputs):
        """slice(expiry) is consistent with unified surface query methods."""
        interpolated_expiry = INTERIOR_INTERPOLATED_EXPIRY
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

    def test_slice_rejects_float_t(self, prob_surface):
        """slice() requires date-like expiry and rejects float t input."""
        with pytest.raises(ValueError, match="date-like expiry"):
            prob_surface.slice(45 / 365.0)  # type: ignore[arg-type]


# =============================================================================
# ProbSurface Query API Tests
# =============================================================================


class TestProbSurfaceQueryApi:
    """Tests for callable probability queries on ProbSurface."""

    @pytest.mark.parametrize("expiry_kind", ["pillar", "interpolated"])
    def test_surface_queries_match_slice_probcurve(
        self,
        prob_surface,
        expiry_kind,
    ):
        """Surface pdf/cdf/quantile queries should match the slice ProbCurve."""
        expiry = (
            prob_surface.expiries[0]
            if expiry_kind == "pillar"
            else INTERIOR_INTERPOLATED_EXPIRY
        )
        curve = prob_surface.slice(expiry)
        sample_indices = np.linspace(0, len(curve.prices) - 1, num=5, dtype=int)
        sample_prices = curve.prices[sample_indices]

        np.testing.assert_allclose(
            prob_surface.pdf(sample_prices, t=expiry),
            curve.pdf(sample_prices),
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            prob_surface.cdf(sample_prices, t=expiry),
            np.asarray([curve.prob_below(float(price)) for price in sample_prices]),
            rtol=1e-8,
            atol=1e-10,
        )
        for quantile_level in (0.1, 0.5, 0.9):
            assert prob_surface.quantile(quantile_level, t=expiry) == pytest.approx(
                curve.quantile(quantile_level)
            )

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
# ProbSurface.density_results() Tests
# =============================================================================


class TestProbSurfaceDensityResults:
    """Tests for ProbSurface.density_results() exports."""

    def test_density_results_returns_long_format_schema(self, prob_surface):
        """density_results() returns expiry-indexed long format."""
        result = prob_surface.density_results()
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["expiry", "price", "pdf", "cdf"]
        unique_expiries = pd.to_datetime(result["expiry"]).unique()
        assert pd.Timestamp(min(prob_surface.expiries)) == pd.Timestamp(
            unique_expiries.min()
        )
        assert pd.Timestamp(max(prob_surface.expiries)) == pd.Timestamp(
            unique_expiries.max()
        )

    def test_density_results_defaults_to_daily_grid(self, prob_surface):
        """Default export contains daily expiries with compact per-slice grids."""
        result = prob_surface.density_results()
        unique_expiries = pd.to_datetime(result["expiry"]).drop_duplicates()
        expected_days = (
            max(prob_surface.expiries) - min(prob_surface.expiries)
        ).days + 1
        assert unique_expiries.min() == min(prob_surface.expiries)
        assert unique_expiries.max() == max(prob_surface.expiries)
        assert len(unique_expiries) == expected_days
        assert len(result) == expected_days * 200
        assert (result.groupby("expiry").size() == 200).all()

    def test_density_results_step_grid_includes_off_step_pillars(self, prob_surface):
        """Off-step fitted pillars are preserved alongside stepped dates."""
        result = prob_surface.density_results(step_days=30)
        unique_expiries = tuple(pd.to_datetime(result["expiry"]).drop_duplicates())
        assert prob_surface.expiries[0] in unique_expiries
        assert prob_surface.expiries[1] in unique_expiries
        assert len(unique_expiries) > len(prob_surface.expiries)

    def test_density_results_respects_hard_date_bounds(self, prob_surface):
        """Start and end act as hard bounds on output rows."""
        result = prob_surface.density_results(
            start="2025-02-25",
            end="2025-05-01",
            step_days=7,
        )
        unique_expiries = pd.to_datetime(result["expiry"]).drop_duplicates()
        assert unique_expiries.min() >= pd.Timestamp("2025-02-25")
        assert unique_expiries.max() <= pd.Timestamp("2025-05-01")
        assert pd.Timestamp("2025-02-21") not in set(unique_expiries)
        assert pd.Timestamp("2025-05-21") not in set(unique_expiries)

    def test_density_results_matches_slice_export_for_pillar(self, prob_surface):
        """Surface export rows match the corresponding single-slice export."""
        pillar_expiry = prob_surface.expiries[0]
        surface_frame = prob_surface.density_results(
            start=pillar_expiry, end=pillar_expiry
        )
        slice_frame = prob_surface.slice(pillar_expiry).density_results()

        assert surface_frame["expiry"].nunique() == 1
        assert pd.Timestamp(surface_frame["expiry"].iloc[0]) == pillar_expiry
        pd.testing.assert_frame_equal(
            surface_frame.drop(columns=["expiry"]).reset_index(drop=True),
            slice_frame.reset_index(drop=True),
        )

    def test_density_results_domain_does_not_mutate_cached_interpolated_slice(
        self, prob_surface
    ):
        """Explicit surface export domains should not mutate cached native slices."""
        interpolated_curve = prob_surface.slice(INTERIOR_INTERPOLATED_EXPIRY)
        native_prices = interpolated_curve.prices.copy()
        native_pdf = interpolated_curve.pdf_values.copy()
        native_cdf = interpolated_curve.cdf_values.copy()
        native_metadata = copy.deepcopy(interpolated_curve.metadata)

        _ = prob_surface.density_results(
            domain=(80.0, 120.0),
            points=21,
            start=INTERIOR_INTERPOLATED_EXPIRY,
            end=INTERIOR_INTERPOLATED_EXPIRY,
        )

        cached_curve = prob_surface.slice(INTERIOR_INTERPOLATED_EXPIRY)
        assert cached_curve is interpolated_curve
        np.testing.assert_allclose(cached_curve.prices, native_prices)
        np.testing.assert_allclose(cached_curve.pdf_values, native_pdf)
        np.testing.assert_allclose(cached_curve.cdf_values, native_cdf)
        assert cached_curve.metadata == native_metadata

    def test_density_results_full_domain_exports_native_pillar_rows(
        self,
        prob_surface,
    ):
        """full_domain=True with pillar export returns each native slice grid."""
        expected_lengths = {
            pd.Timestamp(expiry): len(prob_surface.slice(expiry).prices)
            for expiry in prob_surface.expiries
        }

        result = prob_surface.density_results(full_domain=True, step_days=None)
        observed_lengths = result.groupby("expiry").size()

        assert set(pd.to_datetime(observed_lengths.index)) == set(expected_lengths)
        for expiry, expected_length in expected_lengths.items():
            assert observed_lengths.loc[expiry] == expected_length

    def test_density_results_default_pillar_export_is_not_native_grid(
        self,
        prob_surface,
    ):
        """Default pillar export should use compact view rows, not native rows."""
        pillar_expiry = prob_surface.expiries[0]
        native_length = len(prob_surface.slice(pillar_expiry).prices)

        result = prob_surface.density_results(
            start=pillar_expiry,
            end=pillar_expiry,
            step_days=None,
        )

        assert len(result) == 200
        assert len(result) != native_length

    def test_density_results_does_not_persist_daily_transient_slice_cache(
        self,
        prob_surface,
    ):
        """Bulk daily exports should not retain every sampled slice in cache."""
        cached_pillar = prob_surface.slice(prob_surface.expiries[0])
        preserved_cache_keys = set(prob_surface._curve_cache)

        result = prob_surface.density_results(step_days=1)

        unique_export_expiries = pd.to_datetime(result["expiry"]).nunique()
        assert set(prob_surface._curve_cache) == preserved_cache_keys
        assert prob_surface.slice(prob_surface.expiries[0]) is cached_pillar
        assert len(prob_surface._curve_cache) < unique_export_expiries

    def test_density_results_rejects_invalid_inputs(self, prob_surface):
        """Invalid domains and step sizes are rejected."""
        with pytest.raises(ValueError, match="strictly increasing"):
            prob_surface.density_results(domain=(120.0, 80.0))
        with pytest.raises(ValueError, match="strictly positive integer"):
            prob_surface.density_results(step_days=0)

    def test_density_results_domain_defaults_to_200_points(self, prob_surface):
        """Explicit domain with omitted points defaults to 200 rows per expiry."""
        pillar_expiry = prob_surface.expiries[0]
        result = prob_surface.density_results(
            domain=(80.0, 120.0),
            start=pillar_expiry,
            end=pillar_expiry,
        )
        assert len(result) == 200


# =============================================================================
# ProbSurface.plot_fan() Tests
# =============================================================================


class TestProbSurfacePlotFan:
    """Tests for ProbSurface.plot_fan() visualization."""

    def test_plot_fan_uses_daily_sampling(self, prob_surface):
        """plot_fan() should draw the fixed fan on a daily maturity grid."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.collections import PathCollection, PolyCollection

        fig = prob_surface.plot_fan()
        ax = fig.axes[0]

        band_collections = [
            artist for artist in ax.collections if isinstance(artist, PolyCollection)
        ]
        pillar_collections = [
            artist for artist in ax.collections if isinstance(artist, PathCollection)
        ]
        median_lines = [
            line for line in ax.lines if "median" in line.get_label().lower()
        ]

        assert len(band_collections) == 4
        assert len(pillar_collections) == 1
        assert len(median_lines) == 1
        assert median_lines[0].get_linestyle() == "--"

        xdata = median_lines[0].get_xdata()

        first_expiry = min(prob_surface.expiries)
        last_expiry = max(prob_surface.expiries)
        expected_points = (last_expiry - first_expiry).days + 1
        export_points = prob_surface.density_results(step_days=1)["expiry"].nunique()

        assert len(xdata) == expected_points
        assert len(xdata) == export_points

        pillar_offsets = pillar_collections[0].get_offsets()
        assert pillar_offsets.shape[0] == len(prob_surface.expiries)

        expected_pillar_expiries = pd.DatetimeIndex(
            pd.to_datetime(prob_surface.expiries)
        )
        expected_pillar_x = mdates.date2num(expected_pillar_expiries.to_pydatetime())
        np.testing.assert_allclose(
            np.asarray(pillar_offsets[:, 0], dtype=float),
            np.asarray(expected_pillar_x, dtype=float),
            atol=1e-8,
        )

        expected_medians = np.asarray(
            [
                prob_surface.quantile(0.5, t=expiry)
                for expiry in expected_pillar_expiries
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            np.asarray(pillar_offsets[:, 1], dtype=float), expected_medians
        )
        plt.close(fig)

    def test_plot_fan_skips_invalid_interpolated_slice_and_warns(
        self,
        prob_surface,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """plot_fan() should still render when one non-pillar slice is invalid."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from oipd.interface.probability import ProbCurve
        from oipd.warnings import WorkflowWarning

        original_slice = prob_surface._internal_slice
        invalid_expiry = min(prob_surface.expiries) + pd.Timedelta(days=1)

        def _slice_with_invalid_curve(
            expiry_timestamp: str | date | pd.Timestamp,
        ) -> ProbCurve:
            """Return a malformed ProbCurve for one sampled non-pillar expiry."""
            curve = original_slice(expiry_timestamp)
            if pd.Timestamp(expiry_timestamp) == invalid_expiry:
                invalid_cdf = np.full_like(curve.cdf_values, np.nan, dtype=float)
                return ProbCurve.from_arrays(
                    resolved_market=curve.resolved_market,
                    metadata=dict(curve.metadata),
                    prices=curve.prices,
                    pdf_values=curve.pdf_values,
                    cdf_values=invalid_cdf,
                )
            return curve

        monkeypatch.setattr(
            prob_surface,
            "_internal_slice",
            _slice_with_invalid_curve,
        )

        with pytest.warns(WorkflowWarning, match="warning event"):
            fig = prob_surface.plot_fan()

        ax = fig.axes[0]
        median_lines = [
            line for line in ax.lines if "median" in line.get_label().lower()
        ]
        assert len(median_lines) == 1

        first_expiry = min(prob_surface.expiries)
        last_expiry = max(prob_surface.expiries)
        expected_points = (last_expiry - first_expiry).days + 1
        assert len(median_lines[0].get_xdata()) == expected_points - 1
        events = prob_surface.warning_diagnostics.events
        assert len(events) == 1
        assert events[0].event_type == "skipped_expiry"
        assert events[0].details["skipped_count"] == 1

        plt.close(fig)

    def test_plot_fan_raises_when_all_sampled_slices_invalid(
        self,
        prob_surface,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """plot_fan() should still fail when every sampled slice is invalid."""
        from oipd.interface.probability import ProbCurve

        original_slice = prob_surface._internal_slice

        def _slice_with_invalid_curves(
            expiry_timestamp: str | date | pd.Timestamp,
        ) -> ProbCurve:
            """Return a malformed ProbCurve for every sampled expiry."""
            curve = original_slice(expiry_timestamp)
            invalid_cdf = np.full_like(curve.cdf_values, np.nan, dtype=float)
            return ProbCurve.from_arrays(
                resolved_market=curve.resolved_market,
                metadata=dict(curve.metadata),
                prices=curve.prices,
                pdf_values=curve.pdf_values,
                cdf_values=invalid_cdf,
            )

        monkeypatch.setattr(
            prob_surface,
            "_internal_slice",
            _slice_with_invalid_curves,
        )

        with pytest.raises(
            ValueError,
            match="No valid probability slices available for fan summary generation",
        ):
            prob_surface.plot_fan()

    def test_plot_fan_has_no_pillar_regime_jump(self, prob_surface):
        """Daily fan quantiles should not jump sharply at fitted pillar dates."""
        first_expiry = min(prob_surface.expiries)
        last_expiry = max(prob_surface.expiries)
        sample_expiries = pd.date_range(first_expiry, last_expiry, freq="D")

        if len(sample_expiries) < 11:
            pytest.skip("Insufficient daily points for jump-statistic check.")

        pillar_jump_indices = []
        pillar_expiries = set(prob_surface.expiries)
        for jump_index, right_day in enumerate(sample_expiries[1:]):
            if right_day in pillar_expiries:
                pillar_jump_indices.append(jump_index)

        if not pillar_jump_indices:
            pytest.skip("No pillar transitions found in sampled horizon.")

        for quantile_level in (0.10, 0.50, 0.90):
            quantile_path = np.asarray(
                [
                    prob_surface.quantile(quantile_level, t=expiry_timestamp)
                    for expiry_timestamp in sample_expiries
                ],
                dtype=float,
            )
            jumps = np.abs(np.diff(quantile_path))
            threshold = float(np.percentile(jumps, 95))
            for jump_index in pillar_jump_indices:
                assert jumps[jump_index] <= threshold * 1.01 + 1e-8
