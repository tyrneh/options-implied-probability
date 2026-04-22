"""Tests for direct first-derivative CDF recovery from call-price curves."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from oipd.core.errors import InvalidInputError
from oipd.core.probability_density_conversion.rnd import (
    CDF_SEVERE_MONOTONICITY_EPSILON,
    CDF_TOTAL_NEGATIVE_VARIATION_EPSILON,
    cdf_from_price_curve,
    direct_cdf_diagnostics,
    _minimal_direct_cdf_cleanup,
    raw_cdf_from_price_curve,
)
from oipd.pricing import black76_call_price, black_scholes_call_price


def _assert_raw_direct_cdf_matches_lognormal(
    strikes: np.ndarray,
    raw_direct_cdf: np.ndarray,
    analytic_cdf: np.ndarray,
    expected_quantiles: dict[float, float],
) -> None:
    """Assert unrepaired direct-CDF analytic and structural criteria.

    Args:
        strikes: Strike grid aligned with CDF values.
        raw_direct_cdf: Unclipped first-derivative CDF estimates.
        analytic_cdf: Analytic terminal lognormal CDF values.
        expected_quantiles: Analytic quantiles keyed by probability.
    """
    diagnostics = direct_cdf_diagnostics(raw_direct_cdf)
    full_domain_error = np.max(np.abs(raw_direct_cdf - analytic_cdf))
    central_mask = (analytic_cdf >= 0.01) & (analytic_cdf <= 0.99)
    central_error = np.max(
        np.abs(raw_direct_cdf[central_mask] - analytic_cdf[central_mask])
    )

    assert np.all(np.isfinite(raw_direct_cdf))
    assert full_domain_error <= 3e-3
    assert central_error <= 1e-3
    assert diagnostics["below_zero_count"] == 0
    assert diagnostics["above_one_count"] == 0
    assert diagnostics["negative_step_count"] == 0
    assert diagnostics["max_negative_step"] >= -1e-10

    grid_step = float(strikes[1] - strikes[0])
    for probability, expected_quantile in expected_quantiles.items():
        actual_quantile = float(np.interp(probability, raw_direct_cdf, strikes))
        tolerance = max(grid_step, 0.0025 * expected_quantile)
        assert actual_quantile == pytest.approx(expected_quantile, abs=tolerance)


def _assert_direct_cdf_matches_lognormal(
    strikes: np.ndarray,
    direct_cdf: np.ndarray,
    analytic_cdf: np.ndarray,
    expected_quantiles: dict[float, float],
) -> None:
    """Assert direct-CDF analytic and quantile accuracy criteria.

    Args:
        strikes: Strike grid aligned with CDF values.
        direct_cdf: Direct first-derivative CDF estimates.
        analytic_cdf: Analytic terminal lognormal CDF values.
        expected_quantiles: Analytic quantiles keyed by probability.
    """
    full_domain_error = np.max(np.abs(direct_cdf - analytic_cdf))
    central_mask = (analytic_cdf >= 0.01) & (analytic_cdf <= 0.99)
    central_error = np.max(
        np.abs(direct_cdf[central_mask] - analytic_cdf[central_mask])
    )

    assert full_domain_error <= 3e-3
    assert central_error <= 1e-3
    assert direct_cdf[0] >= 0.0
    assert direct_cdf[-1] <= 1.0
    assert np.all(np.diff(direct_cdf) >= -1e-6)
    assert np.all((0.0 <= direct_cdf) & (direct_cdf <= 1.0))

    grid_step = float(strikes[1] - strikes[0])
    for probability, expected_quantile in expected_quantiles.items():
        actual_quantile = float(np.interp(probability, direct_cdf, strikes))
        tolerance = max(grid_step, 0.0025 * expected_quantile)
        assert actual_quantile == pytest.approx(expected_quantile, abs=tolerance)


def test_cdf_from_price_curve_matches_black_scholes_analytic_cdf() -> None:
    """Direct first-derivative CDF should match analytic Black-Scholes CDF."""
    spot = 100.0
    risk_free_rate = 0.03
    dividend_yield = 0.01
    volatility = 0.22
    years_to_expiry = 1.25
    strikes = np.linspace(1.0, 350.0, 2001)
    call_prices = black_scholes_call_price(
        spot,
        strikes,
        volatility,
        years_to_expiry,
        risk_free_rate,
        dividend_yield,
    )

    cdf_result = cdf_from_price_curve(
        strikes,
        call_prices,
        risk_free_rate=risk_free_rate,
        time_to_expiry_years=years_to_expiry,
        reference_price=spot,
    )
    direct_cdf = cdf_result.cdf_values

    d2 = (
        np.log(spot / strikes)
        + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * years_to_expiry
    ) / (volatility * np.sqrt(years_to_expiry))
    analytic_cdf = norm.cdf(-d2)
    expected_quantiles = {
        probability: spot
        * np.exp(
            (risk_free_rate - dividend_yield - 0.5 * volatility**2) * years_to_expiry
            + volatility * np.sqrt(years_to_expiry) * norm.ppf(probability)
        )
        for probability in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)
    }

    _assert_direct_cdf_matches_lognormal(
        strikes,
        direct_cdf,
        analytic_cdf,
        expected_quantiles,
    )


def test_raw_cdf_from_price_curve_matches_black_scholes_analytic_cdf() -> None:
    """Raw direct CDF should match Black-Scholes before any repairs."""
    spot = 100.0
    risk_free_rate = 0.03
    dividend_yield = 0.01
    volatility = 0.22
    years_to_expiry = 1.25
    strikes = np.linspace(1.0, 350.0, 2001)
    call_prices = black_scholes_call_price(
        spot,
        strikes,
        volatility,
        years_to_expiry,
        risk_free_rate,
        dividend_yield,
    )

    _, raw_direct_cdf = raw_cdf_from_price_curve(
        strikes,
        call_prices,
        risk_free_rate=risk_free_rate,
        time_to_expiry_years=years_to_expiry,
    )

    d2 = (
        np.log(spot / strikes)
        + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * years_to_expiry
    ) / (volatility * np.sqrt(years_to_expiry))
    analytic_cdf = norm.cdf(-d2)
    expected_quantiles = {
        probability: spot
        * np.exp(
            (risk_free_rate - dividend_yield - 0.5 * volatility**2) * years_to_expiry
            + volatility * np.sqrt(years_to_expiry) * norm.ppf(probability)
        )
        for probability in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)
    }

    _assert_raw_direct_cdf_matches_lognormal(
        strikes,
        raw_direct_cdf,
        analytic_cdf,
        expected_quantiles,
    )


def test_cdf_from_price_curve_matches_black76_analytic_cdf() -> None:
    """Direct first-derivative CDF should match analytic Black-76 CDF."""
    forward = 100.0
    risk_free_rate = 0.03
    volatility = 0.22
    years_to_expiry = 1.25
    strikes = np.linspace(1.0, 350.0, 2001)
    call_prices = black76_call_price(
        forward,
        strikes,
        volatility,
        years_to_expiry,
        risk_free_rate,
    )

    cdf_result = cdf_from_price_curve(
        strikes,
        call_prices,
        risk_free_rate=risk_free_rate,
        time_to_expiry_years=years_to_expiry,
        reference_price=forward,
    )
    direct_cdf = cdf_result.cdf_values

    d2 = (np.log(forward / strikes) - 0.5 * volatility**2 * years_to_expiry) / (
        volatility * np.sqrt(years_to_expiry)
    )
    analytic_cdf = norm.cdf(-d2)
    expected_quantiles = {
        probability: forward
        * np.exp(
            -0.5 * volatility**2 * years_to_expiry
            + volatility * np.sqrt(years_to_expiry) * norm.ppf(probability)
        )
        for probability in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)
    }

    _assert_direct_cdf_matches_lognormal(
        strikes,
        direct_cdf,
        analytic_cdf,
        expected_quantiles,
    )


def test_raw_cdf_from_price_curve_matches_black76_analytic_cdf() -> None:
    """Raw direct CDF should match Black-76 before any repairs."""
    forward = 100.0
    risk_free_rate = 0.03
    volatility = 0.22
    years_to_expiry = 1.25
    strikes = np.linspace(1.0, 350.0, 2001)
    call_prices = black76_call_price(
        forward,
        strikes,
        volatility,
        years_to_expiry,
        risk_free_rate,
    )

    _, raw_direct_cdf = raw_cdf_from_price_curve(
        strikes,
        call_prices,
        risk_free_rate=risk_free_rate,
        time_to_expiry_years=years_to_expiry,
    )

    d2 = (np.log(forward / strikes) - 0.5 * volatility**2 * years_to_expiry) / (
        volatility * np.sqrt(years_to_expiry)
    )
    analytic_cdf = norm.cdf(-d2)
    expected_quantiles = {
        probability: forward
        * np.exp(
            -0.5 * volatility**2 * years_to_expiry
            + volatility * np.sqrt(years_to_expiry) * norm.ppf(probability)
        )
        for probability in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)
    }

    _assert_raw_direct_cdf_matches_lognormal(
        strikes,
        raw_direct_cdf,
        analytic_cdf,
        expected_quantiles,
    )


def _quadratic_call_prices_for_linear_raw_cdf(
    strikes: np.ndarray,
    *,
    slope: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Build call prices whose first derivative implies a linear raw CDF.

    Args:
        strikes: Uniform strike grid.
        slope: Slope of the target raw CDF as a function of strike.
        offset: Additive offset for the target raw CDF.

    Returns:
        np.ndarray: Synthetic call prices with derivative
        ``slope * strike + offset - 1``.
    """
    return 0.5 * slope * strikes**2 + (offset - 1.0) * strikes


def test_cdf_from_price_curve_applies_only_tiny_endpoint_cleanup() -> None:
    """Canonical direct CDF should snap only epsilon-level endpoint dust."""
    strikes = np.linspace(0.0, 4.0, 101)
    call_prices = _quadratic_call_prices_for_linear_raw_cdf(
        strikes,
        slope=0.25,
        offset=-3e-6,
    )

    cdf_result = cdf_from_price_curve(
        strikes,
        call_prices,
        risk_free_rate=0.0,
        time_to_expiry_years=1.0,
        reference_price=1.0,
    )

    assert cdf_result.raw_cdf_values[0] == pytest.approx(-3e-6, abs=1e-10)
    assert cdf_result.cdf_values[0] == 0.0
    assert cdf_result.cdf_values[-1] == 1.0
    assert cdf_result.diagnostics["cdf_left_endpoint_snapped"]
    assert cdf_result.diagnostics["cdf_right_endpoint_snapped"]
    assert cdf_result.diagnostics["cdf_cleanup_policy"] == "minimal_epsilon_cleanup"


def test_cdf_from_price_curve_does_not_force_finite_domain_to_one() -> None:
    """A finite right endpoint should keep its actual direct-CDF level."""
    strikes = np.linspace(0.0, 4.0, 101)
    call_prices = _quadratic_call_prices_for_linear_raw_cdf(
        strikes,
        slope=0.10,
    )

    cdf_result = cdf_from_price_curve(
        strikes,
        call_prices,
        risk_free_rate=0.0,
        time_to_expiry_years=1.0,
        reference_price=1.0,
    )

    assert cdf_result.raw_cdf_values[-1] == pytest.approx(0.4, abs=1e-10)
    assert cdf_result.cdf_values[-1] == pytest.approx(0.4, abs=1e-10)
    assert not cdf_result.diagnostics["cdf_right_endpoint_snapped"]


def test_cdf_from_price_curve_rejects_meaningful_lower_bound_violation() -> None:
    """Direct CDF should fail instead of clipping material negative mass."""
    strikes = np.linspace(0.0, 4.0, 101)
    call_prices = _quadratic_call_prices_for_linear_raw_cdf(
        strikes,
        slope=0.25,
        offset=-2e-5,
    )

    with pytest.raises(InvalidInputError, match="materially below zero"):
        cdf_from_price_curve(
            strikes,
            call_prices,
            risk_free_rate=0.0,
            time_to_expiry_years=1.0,
            reference_price=1.0,
        )


def test_cdf_from_price_curve_rejects_meaningful_upper_bound_violation() -> None:
    """Direct CDF should fail instead of capping material mass above one."""
    strikes = np.linspace(0.0, 4.0, 101)
    call_prices = _quadratic_call_prices_for_linear_raw_cdf(
        strikes,
        slope=0.25,
        offset=1.5e-3,
    )

    with pytest.raises(InvalidInputError, match="materially above one"):
        cdf_from_price_curve(
            strikes,
            call_prices,
            risk_free_rate=0.0,
            time_to_expiry_years=1.0,
            reference_price=1.0,
        )


def test_minimal_direct_cdf_cleanup_clips_small_monotone_upper_overshoot() -> None:
    """Small finite monotone upper-tail overshoots should be clipped visibly."""
    strikes = np.linspace(0.0, 4.0, 5)
    raw_cdf_values = np.array([0.0, 0.25, 0.50, 0.90, 1.0007045269])

    cleaned, diagnostics = _minimal_direct_cdf_cleanup(
        strikes,
        raw_cdf_values,
        reference_price=1.0,
    )

    assert cleaned[-1] == 1.0
    assert np.all((0.0 <= cleaned) & (cleaned <= 1.0))
    assert diagnostics["cdf_upper_tail_clip_applied"]
    assert diagnostics["cdf_upper_tail_clip_tolerance"] == pytest.approx(1e-3)
    assert diagnostics["cdf_upper_tail_max_excess"] == pytest.approx(0.0007045269)
    assert diagnostics["cdf_upper_tail_clip_count"] == 1


def test_minimal_direct_cdf_cleanup_warns_and_repairs_monotonicity_violation() -> None:
    """Warn policy should repair material downward CDF steps and continue."""
    strikes = np.linspace(0.0, 4.0, 5)
    raw_cdf_values = np.array([0.0, 0.5, 0.49995, 0.75, 1.0])

    with pytest.warns(UserWarning, match="monotonicity violation repaired"):
        cleaned, diagnostics = _minimal_direct_cdf_cleanup(
            strikes,
            raw_cdf_values,
            reference_price=1.0,
        )

    assert np.all(np.diff(cleaned) >= 0.0)
    assert cleaned[2] == pytest.approx(0.5)
    assert diagnostics["cdf_violation_policy"] == "warn"
    assert diagnostics["cdf_monotonicity_repair_applied"]
    assert diagnostics["cdf_monotonicity_repair_tolerance"] == pytest.approx(5e-6)
    assert diagnostics["cdf_monotonicity_severity"] == "material"
    assert diagnostics["raw_cdf_negative_step_count"] == 1
    assert diagnostics["raw_cdf_max_negative_step"] == pytest.approx(-5e-5)
    assert diagnostics["raw_cdf_total_negative_variation"] == pytest.approx(5e-5)
    assert diagnostics["raw_cdf_worst_step_strike"] == pytest.approx(2.0)
    assert diagnostics["cdf_total_negative_variation_tolerance"] == pytest.approx(
        CDF_TOTAL_NEGATIVE_VARIATION_EPSILON
    )


def test_cdf_from_price_curve_can_suppress_verbose_repair_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct CDF repair diagnostics remain available when warnings are suppressed."""
    import oipd.core.probability_density_conversion.rnd as rnd

    strikes = np.linspace(0.0, 4.0, 5)
    raw_cdf_values = np.array([0.0, 0.5, 0.49995, 0.75, 1.0])

    def fake_first_derivative(
        call_prices: np.ndarray,
        strike_values: np.ndarray,
    ) -> np.ndarray:
        """Return a synthetic first derivative that implies raw CDF damage."""
        return raw_cdf_values - 1.0

    monkeypatch.setattr(rnd, "finite_diff_first_derivative", fake_first_derivative)

    with pytest.warns(UserWarning, match="monotonicity violation repaired"):
        default_result = cdf_from_price_curve(
            strikes,
            np.zeros_like(strikes),
            risk_free_rate=0.0,
            time_to_expiry_years=1.0,
            reference_price=1.0,
        )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        suppressed_result = cdf_from_price_curve(
            strikes,
            np.zeros_like(strikes),
            risk_free_rate=0.0,
            time_to_expiry_years=1.0,
            reference_price=1.0,
            emit_warning=False,
        )

    assert recorded_warnings == []
    assert default_result.diagnostics["cdf_monotonicity_repair_applied"]
    assert suppressed_result.diagnostics["cdf_monotonicity_repair_applied"]
    assert suppressed_result.diagnostics["cdf_monotonicity_severity"] == "material"


def test_minimal_direct_cdf_cleanup_warns_on_cumulative_negative_variation() -> None:
    """Repeated sub-threshold downward steps should trigger warning repair."""
    strikes = np.linspace(0.0, 1.0, 80)
    raw_cdf_values = []
    current_cdf_value = 0.10
    for _ in range(40):
        raw_cdf_values.append(current_cdf_value)
        current_cdf_value += 1.0e-5
        raw_cdf_values.append(current_cdf_value)
        current_cdf_value -= 4.0e-6
    raw_cdf_values_array = np.asarray(raw_cdf_values, dtype=float)

    with pytest.warns(UserWarning, match="total_negative_variation"):
        cleaned, diagnostics = _minimal_direct_cdf_cleanup(
            strikes,
            raw_cdf_values_array,
            reference_price=1.0,
        )

    assert np.all(np.diff(cleaned) >= 0.0)
    assert diagnostics["raw_cdf_negative_step_count"] == 0
    assert diagnostics["raw_cdf_total_negative_variation"] > (
        CDF_TOTAL_NEGATIVE_VARIATION_EPSILON
    )
    assert diagnostics["cdf_monotonicity_repair_applied"]
    assert diagnostics["cdf_monotonicity_severity"] == "material"


def test_minimal_direct_cdf_cleanup_raise_rejects_material_violation() -> None:
    """Raise policy should fail on any material downward CDF step."""
    strikes = np.linspace(0.0, 4.0, 5)
    raw_cdf_values = np.array(
        [0.0, 0.5, 0.5 - (0.5 * CDF_SEVERE_MONOTONICITY_EPSILON), 0.75, 1.0]
    )

    with pytest.raises(
        InvalidInputError,
        match="policy='raise'.*max_negative_step=-5e-05",
    ):
        _minimal_direct_cdf_cleanup(
            strikes,
            raw_cdf_values,
            reference_price=1.0,
            cdf_violation_policy="raise",
        )


def test_minimal_direct_cdf_cleanup_rejects_nonfinite_under_warn_policy() -> None:
    """Non-finite direct CDF values should fail even when repair is requested."""
    strikes = np.linspace(0.0, 4.0, 5)
    raw_cdf_values = np.array([0.0, 0.25, np.nan, 0.75, 1.0])

    with pytest.raises(InvalidInputError, match="non-finite"):
        _minimal_direct_cdf_cleanup(
            strikes,
            raw_cdf_values,
            reference_price=1.0,
            cdf_violation_policy="warn",
        )


def test_minimal_direct_cdf_cleanup_rejects_monotonicity_violation() -> None:
    """Raise policy should fail instead of repairing severe downward steps."""
    strikes = np.linspace(0.0, 4.0, 5)
    raw_cdf_values = np.array(
        [0.0, 0.5, 0.5 - (2.0 * CDF_SEVERE_MONOTONICITY_EPSILON), 0.75, 1.0]
    )

    with pytest.raises(
        InvalidInputError,
        match="Severe.*policy='raise'.*worst_step_strike=2",
    ):
        _minimal_direct_cdf_cleanup(
            strikes,
            raw_cdf_values,
            reference_price=1.0,
            cdf_violation_policy="raise",
        )
