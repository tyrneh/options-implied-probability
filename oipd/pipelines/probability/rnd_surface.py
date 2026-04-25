"""Stateless helpers for VolSurface-derived probability estimation."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from oipd.core.errors import CalculationError, InvalidInputError
from oipd.core.maturity import (
    SECONDS_PER_DAY,
    normalize_datetime_like,
    resolve_maturity,
)
from oipd.market_inputs import ResolvedMarket
from oipd.pipelines.probability.rnd_curve import build_density_results_frame
from oipd.pipelines.probability.rnd_curve import derive_distribution_from_curve
from oipd.pipelines.utils.surface_export import (
    resolve_surface_export_expiries,
    validate_export_domain,
)


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


def derive_surface_slice_probability(
    vol_surface: Any,
    expiry: float | str | date | pd.Timestamp,
    *,
    points: int,
    cdf_violation_policy: str = "warn",
) -> tuple[ResolvedMarket, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Derive one surface-slice probability payload via the single-expiry path.

    Args:
        vol_surface: Fitted volatility surface interface object.
        expiry: Requested maturity as a year fraction or date-like object.
        points: Number of probability-grid points to request from the canonical
            single-expiry distribution pipeline.
        cdf_violation_policy: Direct-CDF monotonicity violation policy passed
            to the canonical single-expiry distribution pipeline.

    Returns:
        tuple[ResolvedMarket, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
            Resolved market, metadata, prices, PDF values, and CDF values for
            the requested slice.
    """
    resolved_expiry, _ = resolve_surface_query_time(vol_surface, expiry)
    vol_curve = vol_surface.slice(resolved_expiry)
    resolved_market = vol_curve.resolved_market
    prices, pdf_values, cdf_values, metadata = derive_distribution_from_curve(
        vol_curve._vol_curve,
        resolved_market,
        pricing_engine=vol_curve._pricing_engine,
        vol_metadata=vol_curve._metadata,
        points=points,
        cdf_violation_policy=cdf_violation_policy,
    )
    return resolved_market, metadata, prices, pdf_values, cdf_values


class _InvalidFanSliceError(ValueError):
    """Internal error for sampled fan slices that should be skipped."""


def _surface_probability_curve(prob_surface: Any, expiry: pd.Timestamp) -> Any:
    """Return a probability curve using the internal no-warning path when present.

    Args:
        prob_surface: Probability surface interface object.
        expiry: Expiry timestamp for the requested slice.

    Returns:
        Any: Probability curve for the requested expiry.
    """
    internal_slice = getattr(prob_surface, "_internal_slice", None)
    if internal_slice is not None:
        return internal_slice(expiry)
    return prob_surface.slice(expiry)


def _exception_chain_contains(
    exc: BaseException | None,
    exception_type: type[BaseException],
) -> bool:
    """Return whether an exception cause/context chain contains a type.

    Args:
        exc: Exception whose causal chain should be inspected.
        exception_type: Exception class to find.

    Returns:
        bool: ``True`` when ``exception_type`` appears in the chain.
    """
    visited: set[int] = set()
    current = exc
    while current is not None and id(current) not in visited:
        if isinstance(current, exception_type):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _is_strict_cdf_materialization_failure(
    prob_surface: Any,
    exc: BaseException,
) -> bool:
    """Return whether a fan error should propagate under strict CDF policy.

    Args:
        prob_surface: Probability surface interface object.
        exc: Exception raised while materializing or querying a sampled slice.

    Returns:
        bool: ``True`` when strict CDF policy caused materialization to fail.
    """
    native_spec = getattr(
        getattr(prob_surface, "_definition", None), "native_spec", None
    )
    cdf_policy = getattr(native_spec, "cdf_violation_policy", None)
    return (
        cdf_policy == "raise"
        and isinstance(exc, CalculationError)
        and _exception_chain_contains(exc.__cause__, InvalidInputError)
    )


def _summarize_fan_skip_reasons(skip_reasons: dict[str, int]) -> str:
    """Summarize invalid fan-slice reasons for aggregate warnings.

    Args:
        skip_reasons: Mapping from reason labels to skipped-slice counts.

    Returns:
        str: Compact reason summary suitable for warning or error messages.
    """
    ordered_reasons = sorted(skip_reasons.items(), key=lambda item: (-item[1], item[0]))
    rendered = [f"{count} x {reason}" for reason, count in ordered_reasons[:3]]
    remaining_count = len(ordered_reasons) - len(rendered)
    if remaining_count > 0:
        rendered.append(f"{remaining_count} more reason(s)")
    return "; ".join(rendered)


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
    curve = _surface_probability_curve(prob_surface, resolved_expiry)
    quantile_values = np.asarray(
        [curve.quantile(level) for level in quantile_levels],
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
        ``p80``, and ``p90``. If sampled expiries are skipped, compact skip
        facts are returned in ``frame.attrs["fan_skip_report"]`` for interface
        diagnostics.

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
    skip_reasons: dict[str, int] = {}
    for expiry_timestamp in export_expiries:
        try:
            record = _build_fan_quantile_record(
                prob_surface,
                pd.Timestamp(expiry_timestamp),
                pillar_expiries=pillar_expiries,
                quantile_levels=quantile_levels,
                quantile_columns=quantile_columns,
            )
        except (CalculationError, InvalidInputError, _InvalidFanSliceError) as exc:
            if _is_strict_cdf_materialization_failure(prob_surface, exc):
                raise
            skipped_expiries.append(pd.Timestamp(expiry_timestamp))
            reason = f"{type(exc).__name__}: {exc}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        records.append(record)

    if not records:
        reason_summary = _summarize_fan_skip_reasons(skip_reasons)
        raise ValueError(
            "No valid probability slices available for fan summary generation. "
            f"Reasons: {reason_summary}."
        )

    summary_frame = pd.DataFrame.from_records(records, columns=summary_columns)

    if skipped_expiries:
        skipped_count = len(skipped_expiries)
        reason_summary = _summarize_fan_skip_reasons(skip_reasons)
        summary_frame.attrs["fan_skip_report"] = {
            "skipped_count": skipped_count,
            "reason_summary": reason_summary,
            "skipped_expiries": [
                pd.Timestamp(expiry_timestamp).isoformat()
                for expiry_timestamp in skipped_expiries
            ],
        }

    return summary_frame


def build_surface_density_results_frame(
    prob_surface: Any,
    *,
    domain: tuple[float, float] | None = None,
    points: int = 200,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    step_days: int | None = 1,
    full_domain: bool = False,
) -> pd.DataFrame:
    """Build a long-format density export DataFrame for a probability surface.

    Args:
        prob_surface: Probability surface interface object.
        domain: Optional export domain as ``(min_price, max_price)``.
        points: Number of resampled grid points for compact or explicit-domain
            exports. Ignored when ``full_domain`` returns native arrays exactly.
        start: Optional lower expiry bound. If omitted, the export starts at
            the first fitted pillar expiry.
        end: Optional upper expiry bound. If omitted, the export ends at the
            last fitted pillar expiry.
        step_days: Calendar-day sampling interval. Defaults to ``1`` so the
            export includes a daily grid while still preserving all fitted
            pillar expiries. Use ``None`` to export fitted pillars only.
        full_domain: If ``True`` and ``domain`` is omitted, each slice exports
            its native full-domain arrays exactly.

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
        curve = _surface_probability_curve(
            prob_surface,
            pd.Timestamp(expiry_timestamp),
        )
        frame = curve.density_results(
            domain=domain,
            points=points,
            full_domain=full_domain,
        )
        frame.insert(0, "expiry", pd.Timestamp(expiry_timestamp))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "build_fan_quantile_summary_frame",
    "build_surface_density_results_frame",
    "derive_surface_slice_probability",
    "resolve_surface_query_time",
]
