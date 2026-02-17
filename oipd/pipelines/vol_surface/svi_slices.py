"""Pipeline for fitting independent volatility slices."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Mapping, Optional

import warnings
import pandas as pd

from oipd.core.errors import CalculationError
from oipd.market_inputs import (
    MarketInputs,
    resolve_market,
)
from oipd.pipelines.vol_curve import fit_vol_curve_internal
from oipd.pipelines.vol_surface.models import DiscreteSurface


def _format_expiry_list(expiry_strings: list[str]) -> str:
    """Format expiry strings for compact warning or error messages.

    Args:
        expiry_strings: Sorted expiry date strings.

    Returns:
        Compact comma-delimited representation for diagnostics.
    """
    if len(expiry_strings) > 4:
        return f"{', '.join(expiry_strings[:3])} ... {expiry_strings[-1]}"
    return ", ".join(expiry_strings)


def _summarize_failure_reasons(failures: list[dict[str, str]]) -> str:
    """Summarize grouped per-expiry failure reasons.

    Args:
        failures: Structured failure records with ``reason`` fields.

    Returns:
        Grouped reason summary with counts suitable for warning text.
    """
    reason_counter = Counter(record["reason"] for record in failures)
    ordered_reason_items = sorted(
        reason_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return "; ".join(f"{reason} ({count})" for reason, count in ordered_reason_items)


def fit_independent_slices(
    chain: pd.DataFrame,
    market: MarketInputs,
    *,
    column_mapping: Optional[Mapping[str, str]],
    method_options: Optional[Mapping[str, Any]],
    pricing_engine: str,
    price_method: str,
    max_staleness_days: int,
    solver: str,
    method: str,
    failure_policy: Literal["raise", "skip_warn"],
) -> DiscreteSurface:
    """Fit a volatility curve independently for each expiry in the chain.

    Args:
        chain: Option chain DataFrame containing multiple expiries.
        market: Base market inputs (valuation date, rates, underlying price).
        column_mapping: Optional mapping from user column names to OIPD standard names.
        method_options: Options for the fitting method.
        pricing_engine: Pricing engine ("black76" or "bs").
        price_method: Price selection strategy (``"mid"`` or ``"last"``).
        max_staleness_days: Maximum allowed quote age before filtering.
        solver: Implied-vol solver to use.
        method: Smile fitting method (e.g., ``"svi"``).
        failure_policy: Slice-level failure handling policy. ``"raise"`` fails fast
            on the first broken expiry, while ``"skip_warn"`` skips failures and
            emits an aggregate warning.

    Returns:
        SliceCollection: A collection of fitted slices.
    """
    if failure_policy not in {"raise", "skip_warn"}:
        raise ValueError(
            "failure_policy must be either 'raise' or 'skip_warn', "
            f"got {failure_policy!r}."
        )

    chain_input = chain.copy()
    if column_mapping:
        chain_input = chain_input.rename(columns=column_mapping)

    if "expiry" not in chain_input.columns:
        raise CalculationError("Expiry column 'expiry' not found in input data")

    expiries = pd.to_datetime(chain_input["expiry"], errors="coerce")
    if expiries.isna().any():
        raise CalculationError("Invalid expiry values encountered during parsing")
    expiries = expiries.dt.tz_localize(None)
    chain_input["expiry"] = expiries

    unique_expiries = sorted(expiries.unique())

    slices: dict[pd.Timestamp, dict[str, Any]] = {}
    resolved_markets: dict[pd.Timestamp, Any] = {}
    slice_chains: dict[pd.Timestamp, pd.DataFrame] = {}

    expiries_with_mid_fills: list[str] = []
    staleness_reports: list[dict[str, Any]] = []
    skipped_expiry_failures: list[dict[str, str]] = []

    for expiry_timestamp in unique_expiries:
        expiry_date = expiry_timestamp.date()
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        slice_df = chain_input[chain_input["expiry"] == expiry_timestamp].copy()

        # Resolve market parameters for this slice
        # For straightforward surfaces, parameters are constant across slices.
        try:
            resolved = resolve_market(market)
            vol_curve, metadata = fit_vol_curve_internal(
                slice_df,
                resolved,
                pricing_engine=pricing_engine,
                price_method=price_method,
                max_staleness_days=max_staleness_days,
                solver=solver,
                method=method,
                method_options=method_options,
                suppress_price_warning=True,
                suppress_staleness_warning=True,
            )
        except Exception as exc:
            exception_type = type(exc).__name__
            reason = f"{exception_type}: {exc}"
            if failure_policy == "raise":
                raise CalculationError(
                    f"Surface calibration failed for expiry {expiry_str}: {reason}. "
                    "To continue with partial calibration, set "
                    "failure_policy='skip_warn'."
                ) from exc

            skipped_expiry_failures.append(
                {
                    "expiry_str": expiry_str,
                    "exception_type": exception_type,
                    "reason": reason,
                }
            )
            continue

        if metadata.get("mid_price_filled"):
            expiries_with_mid_fills.append(expiry_str)

        if metadata.get("staleness_report"):
            report = metadata["staleness_report"]
            report["expiry_str"] = expiry_str
            staleness_reports.append(report)

        slices[expiry_timestamp] = {
            "curve": vol_curve,
            "metadata": metadata,
        }
        resolved_markets[expiry_timestamp] = resolved
        slice_chains[expiry_timestamp] = slice_df

    if skipped_expiry_failures:
        skipped_expiry_strings = sorted(
            record["expiry_str"] for record in skipped_expiry_failures
        )
        skipped_expiry_detail = _format_expiry_list(skipped_expiry_strings)
        reason_summary = _summarize_failure_reasons(skipped_expiry_failures)
        warnings.warn(
            f"Skipped {len(skipped_expiry_failures)} expiries during surface fit: "
            f"[{skipped_expiry_detail}]. Reasons: {reason_summary}.",
            UserWarning,
        )

    if expiries_with_mid_fills:
        count = len(expiries_with_mid_fills)
        # Show first 3 and last 1 if too many
        if count > 4:
            details = f"{', '.join(expiries_with_mid_fills[:3])} ... {expiries_with_mid_fills[-1]}"
        else:
            details = ", ".join(expiries_with_mid_fills)

        warnings.warn(
            f"Filled missing mid prices with last_price for {count} expiries: [{details}]",
            UserWarning,
        )

    if staleness_reports:
        total_removed = sum(r["removed_count"] for r in staleness_reports)
        affected_count = len(staleness_reports)

        # Try to find global min/max age
        min_ages = [
            r["min_age"]
            for r in staleness_reports
            if isinstance(r.get("min_age"), (int, float))
        ]
        max_ages = [
            r["max_age"]
            for r in staleness_reports
            if isinstance(r.get("max_age"), (int, float))
        ]

        age_info = ""
        if min_ages and max_ages:
            global_min = min(min_ages)
            global_max = max(max_ages)
            age_info = f" (most recent: {global_min} days, oldest: {global_max} days)"

        affected_expiries = [
            r["expiry_str"] for r in staleness_reports if "expiry_str" in r
        ]

        if len(affected_expiries) > 4:
            expiry_detail = (
                f"{', '.join(affected_expiries[:3])} ... {affected_expiries[-1]}"
            )
        else:
            expiry_detail = ", ".join(affected_expiries)

        warnings.warn(
            f"Filtered {total_removed} stale option rows across {affected_count} expiries: [{expiry_detail}]"
            f"{age_info} older than {max_staleness_days} days.",
            UserWarning,
        )

    if len(slices) < 2:
        failure_detail = ""
        if skipped_expiry_failures:
            skipped_expiry_strings = sorted(
                record["expiry_str"] for record in skipped_expiry_failures
            )
            skipped_expiry_detail = _format_expiry_list(skipped_expiry_strings)
            reason_summary = _summarize_failure_reasons(skipped_expiry_failures)
            failure_detail = (
                f" Skipped {len(skipped_expiry_failures)} expiries: "
                f"[{skipped_expiry_detail}]. Reasons: {reason_summary}."
            )

        raise CalculationError(
            "VolSurface.fit requires at least two successfully calibrated expiries. "
            f"Only {len(slices)} succeeded out of {len(unique_expiries)} input expiries."
            f"{failure_detail} Broaden data coverage (e.g., horizon or "
            "max_staleness_days), or use VolCurve for single-expiry analysis."
        )

    return DiscreteSurface(slices, resolved_markets, slice_chains)
