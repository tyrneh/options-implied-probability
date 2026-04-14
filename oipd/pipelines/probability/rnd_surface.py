"""Stateless helpers for VolSurface-derived probability estimation."""

from __future__ import annotations

from datetime import date
from typing import Any
import warnings

import numpy as np
import pandas as pd

from oipd.core.errors import InvalidInputError
from oipd.core.maturity import (
    SECONDS_PER_DAY,
    build_maturity_metadata,
    normalize_datetime_like,
    resolve_maturity,
)
from oipd.pipelines.probability.rnd_curve import build_density_results_frame
from oipd.pipelines.utils.surface_export import (
    resolve_surface_export_expiries,
    validate_export_domain,
)
from oipd.core.probability_density_conversion import (
    normalized_cdf_from_call_curve,
    pdf_and_cdf_from_normalized_cdf,
)
from oipd.core.utils import resolve_risk_free_rate
from oipd.market_inputs import Provenance, ResolvedMarket
from oipd.pricing import black76_call_price


def quantile_from_cdf(
    prices: np.ndarray,
    cdf_values: np.ndarray,
    q: float,
) -> float:
    """Interpolate the price level corresponding to a CDF quantile.

    Args:
        prices: Price grid aligned with ``cdf_values``.
        cdf_values: Cumulative probabilities aligned with ``prices``.
        q: Target quantile in ``[0, 1]``.

    Returns:
        float: Price level associated with the requested quantile.

    Raises:
        InvalidInputError: If no finite grid values are available.
    """
    finite_mask = np.isfinite(prices) & np.isfinite(cdf_values)
    if not finite_mask.any():
        raise InvalidInputError("CDF contains no finite values for quantile extraction")

    prices_clean = np.asarray(prices[finite_mask], dtype=float)
    cdf_clean = np.asarray(cdf_values[finite_mask], dtype=float)

    order = np.argsort(prices_clean)
    prices_sorted = prices_clean[order]
    cdf_sorted = cdf_clean[order]
    cdf_sorted = np.maximum.accumulate(cdf_sorted)

    cdf_min = float(cdf_sorted[0])
    cdf_max = float(cdf_sorted[-1])
    if cdf_max - cdf_min < 1e-6:
        return float(prices_sorted[-1])

    q_clamped = float(np.clip(q, cdf_min, cdf_max))
    return float(np.interp(q_clamped, cdf_sorted, prices_sorted))


def _expiry_from_year_fraction(
    valuation_timestamp: pd.Timestamp,
    t_years: float,
    *,
    days_per_year: float = 365.0,
) -> pd.Timestamp:
    """Convert a year-fraction maturity into an expiry timestamp.

    Args:
        valuation_timestamp: Canonical valuation timestamp.
        t_years: Time to expiry in years.
        days_per_year: Day-count denominator.

    Returns:
        pd.Timestamp: Expiry timestamp preserving intraday precision.
    """
    seconds_to_expiry = float(t_years) * days_per_year * SECONDS_PER_DAY
    return valuation_timestamp + pd.to_timedelta(seconds_to_expiry, unit="s")


def build_global_log_moneyness_grid(vol_surface: Any, *, points: int) -> np.ndarray:
    """Build a unified log-moneyness grid shared across all fitted expiries.

    Args:
        vol_surface: Fitted volatility surface interface object.
        points: Number of desired grid points.

    Returns:
        np.ndarray: Uniformly spaced log-moneyness grid.
    """
    k_min = np.inf
    k_max = -np.inf

    for expiry_timestamp in vol_surface.expiries:
        slice_data = vol_surface._model.get_slice(expiry_timestamp)
        metadata = slice_data.get("metadata", {})
        chain = slice_data.get("chain")
        forward_price = metadata.get("forward_price")

        if (
            chain is None
            or "strike" not in chain.columns
            or forward_price is None
            or float(forward_price) <= 0.0
        ):
            continue

        strikes = chain["strike"].to_numpy(dtype=float)
        valid_mask = np.isfinite(strikes) & (strikes > 0.0)
        if not np.any(valid_mask):
            continue

        log_moneyness = np.log(strikes[valid_mask] / float(forward_price))
        k_min = min(k_min, float(np.nanmin(log_moneyness)))
        k_max = max(k_max, float(np.nanmax(log_moneyness)))

    if not np.isfinite(k_min) or not np.isfinite(k_max):
        k_min, k_max = -1.25, 1.25
    elif np.isclose(k_min, k_max):
        k_min -= 0.25
        k_max += 0.25

    pad = 0.05 * (k_max - k_min)
    return np.linspace(k_min - pad, k_max + pad, points)


def resolve_surface_query_time(
    vol_surface: Any,
    t: float | str | date | pd.Timestamp,
) -> tuple[pd.Timestamp, float]:
    """Normalize maturity input and enforce strict supported-domain checks.

    Args:
        vol_surface: Fitted volatility surface interface object.
        t: Maturity as year-fraction float or date-like object.

    Returns:
        tuple[pd.Timestamp, float]: ``(expiry_timestamp, t_years)``.

    Raises:
        ValueError: If maturity is non-positive or beyond the calibrated horizon.
    """
    market = vol_surface._market
    valuation_timestamp = market.valuation_timestamp
    max_expiry = max(vol_surface.expiries)
    max_supported_t = resolve_maturity(
        max_expiry,
        valuation_timestamp,
        floor_at_zero=False,
    ).time_to_expiry_years

    if isinstance(t, (str, date, pd.Timestamp)):
        expiry_timestamp = normalize_datetime_like(t)
        resolved_maturity = resolve_maturity(
            expiry_timestamp,
            valuation_timestamp,
            floor_at_zero=False,
        )
        t_years = float(resolved_maturity.time_to_expiry_years)
        if t_years <= 0:
            raise ValueError(
                "Requested maturity must be strictly after valuation date."
            )
        if expiry_timestamp > max_expiry:
            raise ValueError(
                f"Expiry {expiry_timestamp.date()} is beyond the last fitted pillar "
                f"({max_expiry.date()}). Long-end extrapolation is not supported."
            )
        return resolved_maturity.expiry, t_years

    t_years = float(t)
    if t_years <= 0.0:
        raise ValueError("Requested maturity t must be strictly positive.")
    tolerance_years = 1e-12
    if t_years > max_supported_t + tolerance_years:
        raise ValueError(
            "Requested maturity exceeds the last fitted pillar. "
            "Long-end extrapolation is not supported."
        )

    expiry_timestamp = _expiry_from_year_fraction(valuation_timestamp, t_years)
    if expiry_timestamp > max_expiry:
        expiry_timestamp = max_expiry
    resolved_maturity = resolve_maturity(
        expiry_timestamp,
        valuation_timestamp,
        floor_at_zero=False,
    )
    return resolved_maturity.expiry, float(resolved_maturity.time_to_expiry_years)


def derive_surface_distribution_at_t(
    vol_surface: Any,
    t_years: float,
    *,
    log_moneyness_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive strike-grid PDF/CDF arrays for a given maturity.

    Args:
        vol_surface: Fitted volatility surface interface object.
        t_years: Time to maturity in years.
        log_moneyness_grid: Shared log-moneyness evaluation grid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(strikes, pdf, cdf)``.
    """
    market = vol_surface._market
    forward_price = float(vol_surface.forward_price(t_years))
    strike_grid = forward_price * np.exp(log_moneyness_grid)

    effective_rate = resolve_risk_free_rate(
        market.risk_free_rate, market.risk_free_rate_mode, t_years
    )
    interpolator = vol_surface._interpolator
    implied_vols = np.asarray(
        interpolator.implied_vol(strike_grid, t_years), dtype=float
    )
    call_prices = np.asarray(
        black76_call_price(
            forward_price,
            strike_grid,
            implied_vols,
            t_years,
            effective_rate,
        ),
        dtype=float,
    )

    cdf_values = normalized_cdf_from_call_curve(
        call_prices,
        strike_grid,
        log_moneyness_grid,
        effective_rate=effective_rate,
        time_to_expiry_years=t_years,
    )
    pdf_values, rebuilt_cdf_values = pdf_and_cdf_from_normalized_cdf(
        cdf_values,
        strike_grid,
        log_moneyness_grid,
    )
    return np.asarray(strike_grid, dtype=float), pdf_values, rebuilt_cdf_values


def build_interpolated_resolved_market(
    vol_surface: Any,
    t_years: float,
    *,
    expiry_timestamp: pd.Timestamp | None = None,
) -> ResolvedMarket:
    """Build a synthetic resolved market snapshot for maturity ``t_years``.

    Args:
        vol_surface: Fitted volatility surface interface object.
        t_years: Time to maturity in years.

    Returns:
        ResolvedMarket: Synthetic market snapshot aligned with ``t_years``.
    """
    market = vol_surface._market
    valuation_timestamp = market.valuation_timestamp
    if expiry_timestamp is None:
        expiry_timestamp = _expiry_from_year_fraction(valuation_timestamp, t_years)
    else:
        expiry_timestamp = normalize_datetime_like(expiry_timestamp)
    resolved_maturity = resolve_maturity(
        expiry_timestamp,
        valuation_timestamp,
        floor_at_zero=False,
    )
    canonical_t_years = float(resolved_maturity.time_to_expiry_years)

    return ResolvedMarket(
        risk_free_rate=market.risk_free_rate,
        underlying_price=float(vol_surface.forward_price(canonical_t_years)),
        valuation_date=valuation_timestamp,
        dividend_yield=market.dividend_yield,
        dividend_schedule=None,
        provenance=Provenance(price="user", dividends="none"),
        source_meta={
            "interpolated": True,
            "expiry": resolved_maturity.expiry,
            "time_to_expiry_years": canonical_t_years,
            "risk_free_rate_mode": market.risk_free_rate_mode,
        },
    )


def build_probcurve_metadata(
    vol_surface: Any,
    expiry_timestamp: pd.Timestamp,
    t_years: float,
) -> dict[str, Any]:
    """Build metadata payload for a ``ProbSurface.slice(...)`` result.

    Args:
        vol_surface: Fitted volatility surface interface object.
        expiry_timestamp: Requested expiry timestamp.
        t_years: Time to expiry in years.

    Returns:
        dict[str, Any]: Metadata dictionary for ``ProbCurve.from_arrays``.
    """
    market = vol_surface._market
    valuation_timestamp = market.valuation_timestamp
    resolved_maturity = resolve_maturity(
        expiry_timestamp,
        valuation_timestamp,
        floor_at_zero=False,
    )
    canonical_t_years = float(resolved_maturity.time_to_expiry_years)
    is_pillar = resolved_maturity.expiry in vol_surface.expiries
    return {
        "interpolated": not is_pillar,
        "method": "unified_surface_probability",
        **build_maturity_metadata(resolved_maturity),
        "forward_price": float(vol_surface.forward_price(canonical_t_years)),
        "at_money_vol": float(vol_surface.atm_vol(canonical_t_years)),
    }


class _InvalidFanSliceError(ValueError):
    """Internal error for sampled fan slices that should be skipped."""


def _build_fan_quantile_record(
    prob_surface: Any,
    expiry_timestamp: pd.Timestamp,
    *,
    pillar_expiries: set[pd.Timestamp],
    quantile_levels: tuple[float, ...],
    quantile_columns: list[str],
) -> dict[str, Any]:
    """Build one fan-summary record for a sampled expiry.

    Args:
        prob_surface: Probability surface interface object.
        expiry_timestamp: Expiry being summarized.
        pillar_expiries: Set of fitted pillar expiries.
        quantile_levels: Quantile levels to extract.
        quantile_columns: Output column names aligned with ``quantile_levels``.

    Returns:
        dict[str, Any]: One summary row for the requested expiry.

    Raises:
        InvalidInputError: If quantile extraction cannot be performed.
        _InvalidFanSliceError: If extracted quantiles are non-finite or unordered.
    """
    resolved_expiry = pd.Timestamp(expiry_timestamp)
    _, t_years = prob_surface._resolve_query_time(resolved_expiry)
    prices, _, cdf_values = prob_surface._distribution_arrays_for_t_years(
        t_years,
        expiry_timestamp=resolved_expiry,
    )

    quantile_values = np.asarray(
        [quantile_from_cdf(prices, cdf_values, level) for level in quantile_levels],
        dtype=float,
    )
    if not np.all(np.isfinite(quantile_values)):
        raise _InvalidFanSliceError(
            f"Fan quantiles must be finite for expiry {resolved_expiry}."
        )
    if np.any(np.diff(quantile_values) < -1e-8):
        raise _InvalidFanSliceError(
            f"Fan quantiles must be ordered for expiry {resolved_expiry}."
        )

    record: dict[str, Any] = {
        "expiry": resolved_expiry,
        "is_pillar": resolved_expiry in pillar_expiries,
    }
    record.update(dict(zip(quantile_columns, quantile_values, strict=True)))
    return record


def build_fan_quantile_summary_frame(prob_surface: Any) -> pd.DataFrame:
    """Build a daily fan-summary frame of quantiles across expiries.

    Args:
        prob_surface: Probability surface interface object.

    Returns:
        pd.DataFrame: Summary frame with columns ``expiry``, ``is_pillar``,
        ``p10``, ``p20``, ``p30``, ``p40``, ``p50``, ``p60``, ``p70``,
        ``p80``, and ``p90``.

    Raises:
        ValueError: If no valid sampled expiries remain after invalid slices
            are skipped.
    """
    export_expiries = resolve_surface_export_expiries(prob_surface, step_days=1)
    summary_columns = [
        "expiry",
        "is_pillar",
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
    ]
    if not export_expiries:
        return pd.DataFrame(columns=summary_columns)

    pillar_expiries = {pd.Timestamp(expiry) for expiry in prob_surface.expiries}
    quantile_levels = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    quantile_columns = [f"p{int(level * 100):02d}" for level in quantile_levels]

    records: list[dict[str, Any]] = []
    skipped_expiries: list[pd.Timestamp] = []
    for expiry_timestamp in export_expiries:
        try:
            record = _build_fan_quantile_record(
                prob_surface,
                pd.Timestamp(expiry_timestamp),
                pillar_expiries=pillar_expiries,
                quantile_levels=quantile_levels,
                quantile_columns=quantile_columns,
            )
        except (InvalidInputError, _InvalidFanSliceError):
            skipped_expiries.append(pd.Timestamp(expiry_timestamp))
            continue
        records.append(record)

    if not records:
        raise ValueError(
            "No valid probability slices available for fan summary generation"
        )

    if skipped_expiries:
        skipped_count = len(skipped_expiries)
        expiry_label = "expiry" if skipped_count == 1 else "expiries"
        warnings.warn(
            (
                f"Skipped {skipped_count} sampled {expiry_label} during fan summary "
                "generation due to invalid quantile data."
            ),
            UserWarning,
            stacklevel=2,
        )

    return pd.DataFrame.from_records(records, columns=summary_columns)


def build_surface_density_results_frame(
    prob_surface: Any,
    *,
    domain: tuple[float, float] | None = None,
    points: int = 200,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    step_days: int | None = 1,
) -> pd.DataFrame:
    """Build a long-format density export DataFrame for a probability surface.

    Args:
        prob_surface: Probability surface interface object.
        domain: Optional export domain as ``(min_price, max_price)``.
        points: Number of resampled grid points when ``domain`` is set.
            Ignored when ``domain`` is omitted and the native slice grids are
            returned unchanged.
        start: Optional lower expiry bound. If omitted, the export starts at
            the first fitted pillar expiry.
        end: Optional upper expiry bound. If omitted, the export ends at the
            last fitted pillar expiry.
        step_days: Calendar-day sampling interval. Defaults to ``1`` so the
            export includes a daily grid while still preserving all fitted
            pillar expiries. Use ``None`` to export fitted pillars only.

    Returns:
        Long DataFrame with columns ``expiry``, ``price``, ``pdf``, and ``cdf``.
    """
    validate_export_domain(domain)
    export_expiries = resolve_surface_export_expiries(
        prob_surface,
        start=start,
        end=end,
        step_days=step_days,
    )
    if not export_expiries:
        return pd.DataFrame(columns=["expiry", "price", "pdf", "cdf"])

    frames: list[pd.DataFrame] = []
    for expiry_timestamp in export_expiries:
        _, t_years = prob_surface._resolve_query_time(expiry_timestamp)
        prices, pdf_values, cdf_values = prob_surface._distribution_arrays_for_t_years(
            t_years,
            expiry_timestamp=expiry_timestamp,
        )
        frame = build_density_results_frame(
            prices,
            pdf_values,
            cdf_values,
            domain=domain,
            points=points,
        )
        frame.insert(0, "expiry", pd.Timestamp(expiry_timestamp))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "build_fan_quantile_summary_frame",
    "build_global_log_moneyness_grid",
    "build_interpolated_resolved_market",
    "build_surface_density_results_frame",
    "build_probcurve_metadata",
    "derive_surface_distribution_at_t",
    "resolve_surface_query_time",
]
