"""Put-call parity preprocessing utilities."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from oipd.core.maturity import resolve_maturity
from oipd.core.utils import (
    resolve_risk_free_rate,
)
from oipd.market_inputs import ResolvedMarket

_MODIFIED_Z_SCORE_CUTOFF = 3.5
_MODIFIED_Z_SCORE_SCALE = 0.6745
_DEFAULT_MAX_FORWARD_PAIRS = 5
_DEFAULT_MIN_LAST_LEG_VOLUME = 1.0
_DEFAULT_MAX_BID_ASK_RELATIVE_SPREAD = 0.25


def infer_forward_from_atm(
    options_df: pd.DataFrame,
    underlying_price: float,
    discount_factor: float,
    *,
    max_forward_pairs: int = _DEFAULT_MAX_FORWARD_PAIRS,
    min_last_leg_volume: Optional[float] = _DEFAULT_MIN_LAST_LEG_VOLUME,
    max_bid_ask_relative_spread: Optional[float] = _DEFAULT_MAX_BID_ASK_RELATIVE_SPREAD,
) -> float:
    """Infer the forward price through the parity helper for compatibility.

    This compatibility wrapper delegates to
    :func:`infer_forward_from_put_call_parity`, which uses all valid same-strike
    pairs with robust aggregation when enough pairs are available.

    Args:
        options_df: Option quotes containing either ``call_price`` and ``put_price``
            columns or rows distinguished by an ``option_type`` column.
        underlying_price: Observed underlying spot price used to rank strikes in
            diagnostics and low-pair compatibility paths.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.
        max_forward_pairs: Maximum number of nearest-ATM valid parity pairs to use.
        min_last_leg_volume: Minimum canonical lowercase ``volume`` required on
            each leg when ``last_price`` fallback is used. Missing ``volume`` skips
            the filter and is recorded in diagnostics.
        max_bid_ask_relative_spread: Maximum relative bid/ask spread accepted for
            bid/ask-mid forward inference.

    Returns:
        Forward price inferred from valid same-strike call-put pairs.

    Raises:
        ValueError: If ``discount_factor`` is not finite and strictly positive, the
            inputs do not contain a same-strike call-put pair, or the parity
            calculation cannot produce a positive forward price.
    """
    return infer_forward_from_put_call_parity(
        options_df=options_df,
        underlying_price=underlying_price,
        discount_factor=discount_factor,
        max_forward_pairs=max_forward_pairs,
        min_last_leg_volume=min_last_leg_volume,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    )


def infer_forward_from_put_call_parity(
    options_df: pd.DataFrame,
    underlying_price: float,
    discount_factor: float,
    *,
    max_forward_pairs: int = _DEFAULT_MAX_FORWARD_PAIRS,
    min_last_leg_volume: Optional[float] = _DEFAULT_MIN_LAST_LEG_VOLUME,
    max_bid_ask_relative_spread: Optional[float] = _DEFAULT_MAX_BID_ASK_RELATIVE_SPREAD,
) -> float:
    """Infer the forward price from same-strike put-call parity pairs.

    Three or more valid pairs use median/MAD outlier filtering. One- and two-pair
    cases still return a forward but emit a low-confidence warning.

    Args:
        options_df: Option quotes containing either explicit ``call_price`` and
            ``put_price`` columns or rows distinguished by an ``option_type`` column.
        underlying_price: Observed underlying spot price used to rank strikes in
            diagnostics and low-pair compatibility paths.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.
        max_forward_pairs: Maximum number of nearest-ATM valid parity pairs to use.
        min_last_leg_volume: Minimum canonical lowercase ``volume`` required on
            each leg when ``last_price`` fallback is used. Missing ``volume`` skips
            the filter and is recorded in diagnostics.
        max_bid_ask_relative_spread: Maximum relative bid/ask spread accepted for
            bid/ask-mid forward inference.

    Returns:
        Robust or low-confidence forward price inferred from valid same-strike
        call-put pairs.

    Raises:
        ValueError: If ``discount_factor`` is not finite and strictly positive, the
            inputs do not contain a same-strike call-put pair, or the parity
            calculation cannot produce a positive forward price.
    """
    forward_price, _ = _infer_forward_from_put_call_parity_with_report(
        options_df=options_df,
        underlying_price=underlying_price,
        discount_factor=discount_factor,
        max_forward_pairs=max_forward_pairs,
        min_last_leg_volume=min_last_leg_volume,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    )
    return forward_price


def _infer_forward_from_put_call_parity_with_report(
    options_df: pd.DataFrame,
    underlying_price: float,
    discount_factor: float,
    *,
    max_forward_pairs: int = _DEFAULT_MAX_FORWARD_PAIRS,
    min_last_leg_volume: Optional[float] = _DEFAULT_MIN_LAST_LEG_VOLUME,
    max_bid_ask_relative_spread: Optional[float] = _DEFAULT_MAX_BID_ASK_RELATIVE_SPREAD,
) -> tuple[float, dict[str, Any]]:
    """Infer the forward price and return parity diagnostics.

    Args:
        options_df: Option quotes containing either explicit ``call_price`` and
            ``put_price`` columns or rows distinguished by an ``option_type`` column.
        underlying_price: Observed underlying spot price used to rank strikes by ATM
            proximity for legacy low-pair inference.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.
        max_forward_pairs: Maximum number of nearest-ATM valid parity pairs to use.
        min_last_leg_volume: Minimum canonical lowercase ``volume`` required on
            each leg when ``last_price`` fallback is used. Missing ``volume`` skips
            the filter and is recorded in diagnostics.
        max_bid_ask_relative_spread: Maximum relative bid/ask spread accepted for
            bid/ask-mid forward inference.

    Returns:
        Tuple containing the inferred forward price and a diagnostics report.

    Raises:
        ValueError: If ``discount_factor`` is not finite and strictly positive, the
            inputs do not contain a same-strike call-put pair, or no valid pair can
            produce a positive forward price.
    """
    discount_factor = _validate_discount_factor(discount_factor)
    max_forward_pairs = _validate_max_forward_pairs(max_forward_pairs)
    min_last_leg_volume = _validate_min_last_leg_volume(min_last_leg_volume)
    max_bid_ask_relative_spread = _validate_max_bid_ask_relative_spread(
        max_bid_ask_relative_spread
    )
    candidate_pairs = _collect_forward_candidate_pairs(
        options_df=options_df,
        underlying_price=underlying_price,
    )

    if not candidate_pairs:
        raise ValueError("No strikes found with both call and put options.")

    valid_pairs, pair_diagnostics = _evaluate_forward_candidate_pairs(
        candidate_pairs=candidate_pairs,
        discount_factor=discount_factor,
        min_last_leg_volume=min_last_leg_volume,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    )

    if not valid_pairs:
        available_strikes = sorted(pair["strike"] for pair in candidate_pairs)
        raise ValueError(
            "No valid put-call parity pairs found "
            f"(0 valid pairs from {len(candidate_pairs)} candidate pairs). "
            "Check same-strike call/put prices for positive forward estimates and "
            "intrinsic-value consistency. "
            f"Available candidate strikes: {available_strikes}"
        )

    (
        forward_price,
        aggregation_report,
        selected_candidate_indexes,
        used_candidate_indexes,
        outlier_candidate_indexes,
    ) = _select_forward_from_valid_pairs(
        valid_pairs=valid_pairs,
        max_forward_pairs=max_forward_pairs,
    )
    parity_report = _build_parity_report(
        candidate_pairs=candidate_pairs,
        valid_pairs=valid_pairs,
        pair_diagnostics=pair_diagnostics,
        aggregation_report=aggregation_report,
        selected_candidate_indexes=selected_candidate_indexes,
        used_candidate_indexes=used_candidate_indexes,
        outlier_candidate_indexes=outlier_candidate_indexes,
        forward_price=forward_price,
        max_forward_pairs=max_forward_pairs,
        min_last_leg_volume=min_last_leg_volume,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    )
    return forward_price, parity_report


def _evaluate_forward_candidate_pairs(
    candidate_pairs: list[dict[str, Any]],
    discount_factor: float,
    min_last_leg_volume: Optional[float],
    max_bid_ask_relative_spread: Optional[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate candidate pairs and mark parity-valid observations.

    Args:
        candidate_pairs: Same-strike call-put candidates.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.
        min_last_leg_volume: Minimum per-leg volume for last-price fallback, or
            ``None`` to disable the volume floor.
        max_bid_ask_relative_spread: Maximum relative bid/ask spread for midquote
            inference, or ``None`` to disable the spread ceiling.

    Returns:
        Tuple containing valid pair records and pair-level diagnostics for every
        candidate.
    """
    valid_pairs: list[dict[str, Any]] = []
    pair_diagnostics: list[dict[str, Any]] = []

    for candidate_index, pair in enumerate(candidate_pairs):
        diagnostic = _build_pair_diagnostic(pair, candidate_index)

        try:
            price_pair = _select_forward_price_pair(
                pair=pair,
                min_last_leg_volume=min_last_leg_volume,
                max_bid_ask_relative_spread=max_bid_ask_relative_spread,
            )
            diagnostic.update(price_pair["diagnostics"])
            if not price_pair["valid"]:
                diagnostic["status"] = "invalid"
                diagnostic["excluded_reason"] = price_pair["excluded_reason"]
            else:
                priced_pair = dict(pair)
                priced_pair.update(price_pair["pair_fields"])
                forward_price = _calculate_pair_forward(priced_pair, discount_factor)
                diagnostic["forward_price"] = forward_price
                excluded_reason = _get_pair_exclusion_reason(
                    pair=priced_pair,
                    forward_price=forward_price,
                    discount_factor=discount_factor,
                )
                if excluded_reason is not None:
                    diagnostic["status"] = "invalid"
                    diagnostic["excluded_reason"] = excluded_reason
                else:
                    diagnostic["valid"] = True
                    diagnostic["status"] = "valid"
                    valid_pair = dict(priced_pair)
                    valid_pair["candidate_index"] = candidate_index
                    valid_pair["forward_price"] = forward_price
                    valid_pairs.append(valid_pair)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            diagnostic["status"] = "invalid"
            diagnostic["excluded_reason"] = type(exc).__name__

        pair_diagnostics.append(diagnostic)

    return valid_pairs, pair_diagnostics


def _build_pair_diagnostic(
    pair: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    """Build the default diagnostics record for one candidate pair.

    Args:
        pair: Same-strike call-put candidate.
        candidate_index: Stable position of the candidate in collection order.

    Returns:
        Pair-level diagnostics dictionary.
    """
    return {
        "candidate_index": candidate_index,
        "strike": float(pair["strike"]),
        "call_price": _optional_float(pair.get("call_price")),
        "put_price": _optional_float(pair.get("put_price")),
        "price_source": str(pair.get("price_source", "unavailable")),
        "distance_from_underlying": float(pair["distance_from_underlying"]),
        "forward_price": None,
        "absolute_deviation": None,
        "modified_z_score": None,
        "valid": False,
        "selected": False,
        "used": False,
        "outlier": False,
        "status": "invalid",
        "excluded_reason": None,
        "call_relative_spread": None,
        "put_relative_spread": None,
        "call_volume": None,
        "put_volume": None,
        "volume_filter_status": "not_applicable",
    }


def _calculate_pair_forward(
    pair: dict[str, Any],
    discount_factor: float,
) -> float:
    """Calculate a pair-implied forward price.

    Args:
        pair: Same-strike call-put candidate.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.

    Returns:
        Forward price implied by put-call parity.
    """
    return float(
        pair["strike"] + (pair["call_price"] - pair["put_price"]) / discount_factor
    )


def _get_pair_exclusion_reason(
    pair: dict[str, Any],
    forward_price: float,
    discount_factor: float,
) -> Optional[str]:
    """Return why a pair cannot be used for forward inference.

    Args:
        pair: Same-strike call-put candidate.
        forward_price: Forward price implied by the pair.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.

    Returns:
        Exclusion reason when invalid; otherwise ``None``.
    """
    if not np.isfinite(forward_price) or forward_price <= 0.0:
        return "non_positive_forward"

    strike = pair["strike"]
    call_price = pair["call_price"]
    put_price = pair["put_price"]
    intrinsic_call = max(0.0, discount_factor * (forward_price - strike))
    intrinsic_put = max(0.0, discount_factor * (strike - forward_price))

    if call_price < intrinsic_call:
        return "call_below_intrinsic"
    if put_price < intrinsic_put:
        return "put_below_intrinsic"
    return None


def _select_forward_from_valid_pairs(
    valid_pairs: list[dict[str, Any]],
    max_forward_pairs: int,
) -> tuple[float, dict[str, Any], set[int], set[int], set[int]]:
    """Select the final forward from valid pair observations.

    Args:
        valid_pairs: Pair records that passed parity validity checks.
        max_forward_pairs: Maximum number of nearest-ATM valid pairs to aggregate.

    Returns:
        Tuple containing the final forward, aggregation diagnostics, selected
        candidate indexes, used candidate indexes, and outlier candidate indexes.
    """
    selected_pairs = sorted(
        valid_pairs,
        key=lambda pair: (
            float(pair["distance_from_underlying"]),
            float(pair["strike"]),
            int(pair["candidate_index"]),
        ),
    )[:max_forward_pairs]
    selected_candidate_indexes = {
        int(pair["candidate_index"]) for pair in selected_pairs
    }

    if len(selected_pairs) >= 3:
        (
            forward_price,
            aggregation_report,
            used_candidate_indexes,
            outlier_candidate_indexes,
        ) = _select_robust_forward_from_valid_pairs(selected_pairs)
        return (
            forward_price,
            aggregation_report,
            selected_candidate_indexes,
            used_candidate_indexes,
            outlier_candidate_indexes,
        )

    confidence = "low_two_pairs" if len(selected_pairs) == 2 else "low_single_pair"
    pair_forwards = [pair["forward_price"] for pair in selected_pairs]
    final_forward = float(np.median(pair_forwards))
    used_candidate_indexes = {int(pair["candidate_index"]) for pair in selected_pairs}
    _warn_low_confidence_forward_inference(
        valid_pair_count=len(selected_pairs),
        confidence=confidence,
    )
    aggregation_report = {
        "confidence": confidence,
        "aggregation_method": "median_low_pair",
        "raw_median": final_forward,
        "mad": None,
        "fallback_tolerance": None,
        "modified_z_score_cutoff": None,
        "filter_failed_open": False,
        "pair_metrics": {
            int(pair["candidate_index"]): {
                "absolute_deviation": None,
                "modified_z_score": None,
                "retained": True,
            }
            for pair in selected_pairs
        },
    }
    return (
        final_forward,
        aggregation_report,
        selected_candidate_indexes,
        used_candidate_indexes,
        set(),
    )


def _warn_low_confidence_forward_inference(
    valid_pair_count: int,
    confidence: str,
) -> None:
    """Warn that forward inference is based on few valid pairs.

    Args:
        valid_pair_count: Number of valid same-strike pairs used for inference.
        confidence: Low-confidence label included in diagnostics.
    """
    pair_label = "pair" if valid_pair_count == 1 else "pairs"
    warnings.warn(
        "Put-call parity forward inference used "
        f"{valid_pair_count} valid {pair_label}; confidence={confidence}.",
        UserWarning,
    )


def _select_robust_forward_from_valid_pairs(
    valid_pairs: list[dict[str, Any]],
) -> tuple[float, dict[str, Any], set[int], set[int]]:
    """Select a forward using median/MAD outlier filtering.

    Args:
        valid_pairs: Three or more valid pair records.

    Returns:
        Tuple containing the final robust forward, aggregation diagnostics, used
        candidate indexes, and outlier candidate indexes.
    """
    pair_forwards = np.asarray(
        [pair["forward_price"] for pair in valid_pairs],
        dtype=float,
    )
    raw_median = float(np.median(pair_forwards))
    absolute_deviations = np.abs(pair_forwards - raw_median)
    mad = float(np.median(absolute_deviations))
    fallback_tolerance: Optional[float] = None

    if mad == 0.0:
        fallback_tolerance = max(0.01, 1e-6 * max(abs(raw_median), 1.0))
        retained_mask = absolute_deviations <= fallback_tolerance
        modified_z_scores = np.where(
            absolute_deviations <= fallback_tolerance,
            0.0,
            np.inf,
        )
    else:
        modified_z_scores = _MODIFIED_Z_SCORE_SCALE * absolute_deviations / mad
        retained_mask = modified_z_scores <= _MODIFIED_Z_SCORE_CUTOFF

    filter_failed_open = False
    if not bool(np.any(retained_mask)):
        retained_mask = np.ones(len(valid_pairs), dtype=bool)
        filter_failed_open = True

    used_pairs = [
        pair for pair, retained in zip(valid_pairs, retained_mask) if retained
    ]
    used_candidate_indexes = {int(pair["candidate_index"]) for pair in used_pairs}
    outlier_candidate_indexes = {
        int(pair["candidate_index"])
        for pair, retained in zip(valid_pairs, retained_mask)
        if not retained
    }
    final_forward = float(np.median([pair["forward_price"] for pair in used_pairs]))
    pair_metrics = {
        int(pair["candidate_index"]): {
            "absolute_deviation": float(absolute_deviation),
            "modified_z_score": (
                None if pd.isna(modified_z_score) else float(modified_z_score)
            ),
            "retained": bool(retained),
        }
        for pair, absolute_deviation, modified_z_score, retained in zip(
            valid_pairs,
            absolute_deviations,
            modified_z_scores,
            retained_mask,
        )
    }
    aggregation_report = {
        "confidence": "robust",
        "aggregation_method": "median_mad",
        "raw_median": raw_median,
        "mad": mad,
        "fallback_tolerance": fallback_tolerance,
        "modified_z_score_cutoff": _MODIFIED_Z_SCORE_CUTOFF,
        "filter_failed_open": filter_failed_open,
        "pair_metrics": pair_metrics,
    }

    return (
        final_forward,
        aggregation_report,
        used_candidate_indexes,
        outlier_candidate_indexes,
    )


def _build_parity_report(
    candidate_pairs: list[dict[str, Any]],
    valid_pairs: list[dict[str, Any]],
    pair_diagnostics: list[dict[str, Any]],
    aggregation_report: dict[str, Any],
    selected_candidate_indexes: set[int],
    used_candidate_indexes: set[int],
    outlier_candidate_indexes: set[int],
    forward_price: float,
    max_forward_pairs: int,
    min_last_leg_volume: Optional[float],
    max_bid_ask_relative_spread: Optional[float],
) -> dict[str, Any]:
    """Build the parity inference diagnostics report.

    Args:
        candidate_pairs: Same-strike call-put candidates considered for inference.
        valid_pairs: Candidate pairs that passed validity checks.
        pair_diagnostics: Pair-level diagnostics for all candidates.
        aggregation_report: Robust or low-confidence aggregation diagnostics.
        selected_candidate_indexes: Candidate indexes included in the nearest-ATM
            selection subset.
        used_candidate_indexes: Candidate indexes retained for the final forward.
        outlier_candidate_indexes: Candidate indexes excluded as outliers.
        forward_price: Final inferred forward price.
        max_forward_pairs: Configured nearest-ATM pair cap.
        min_last_leg_volume: Configured last-price volume floor.
        max_bid_ask_relative_spread: Configured bid/ask spread ceiling.

    Returns:
        Diagnostics report suitable for storing in ``DataFrame.attrs``.
    """
    pair_metrics = aggregation_report.get("pair_metrics", {})
    valid_candidate_indexes = {int(pair["candidate_index"]) for pair in valid_pairs}
    valid_not_selected_candidate_indexes = (
        valid_candidate_indexes - selected_candidate_indexes
    )
    report_pairs = []

    for diagnostic in pair_diagnostics:
        pair_report = dict(diagnostic)
        candidate_index = int(pair_report["candidate_index"])
        metrics = pair_metrics.get(candidate_index, {})
        pair_report["absolute_deviation"] = metrics.get("absolute_deviation")
        pair_report["modified_z_score"] = metrics.get("modified_z_score")
        pair_report["selected"] = candidate_index in selected_candidate_indexes

        if candidate_index in used_candidate_indexes:
            pair_report["used"] = True
            pair_report["status"] = "used"
            pair_report["excluded_reason"] = None
        elif candidate_index in outlier_candidate_indexes:
            pair_report["outlier"] = True
            pair_report["status"] = "outlier"
            pair_report["excluded_reason"] = "robust_outlier"
        elif candidate_index in valid_not_selected_candidate_indexes:
            pair_report["status"] = "valid_not_selected"
            pair_report["excluded_reason"] = "not_selected_nearest_atm_cap"
        else:
            pair_report["status"] = "invalid"

        report_pairs.append(pair_report)

    outlier_strikes = [pair["strike"] for pair in report_pairs if pair["outlier"]]
    used_strikes = [pair["strike"] for pair in report_pairs if pair["used"]]
    selected_strikes = [pair["strike"] for pair in report_pairs if pair["selected"]]
    valid_not_selected_strikes = [
        pair["strike"]
        for pair in report_pairs
        if pair["status"] == "valid_not_selected"
    ]

    return {
        "confidence": aggregation_report["confidence"],
        "aggregation_method": aggregation_report["aggregation_method"],
        "forward_price": float(forward_price),
        "candidate_pair_count": len(candidate_pairs),
        "valid_pair_count": len(valid_pairs),
        "selected_pair_count": len(selected_candidate_indexes),
        "pairs_used_count": len(used_candidate_indexes),
        "outlier_count": len(outlier_candidate_indexes),
        "valid_not_selected_count": len(valid_not_selected_candidate_indexes),
        "max_forward_pairs": max_forward_pairs,
        "min_last_leg_volume": min_last_leg_volume,
        "max_bid_ask_relative_spread": max_bid_ask_relative_spread,
        "selected_strikes": selected_strikes,
        "used_strikes": used_strikes,
        "outlier_strikes": outlier_strikes,
        "valid_not_selected_strikes": valid_not_selected_strikes,
        "quote_liquidity_confidence": _determine_quote_liquidity_confidence(
            report_pairs
        ),
        "raw_median": aggregation_report.get("raw_median"),
        "mad": aggregation_report.get("mad"),
        "fallback_tolerance": aggregation_report.get("fallback_tolerance"),
        "modified_z_score_cutoff": aggregation_report.get("modified_z_score_cutoff"),
        "filter_failed_open": aggregation_report.get("filter_failed_open", False),
        "pairs": report_pairs,
    }


def _determine_quote_liquidity_confidence(
    report_pairs: list[dict[str, Any]],
) -> str:
    """Summarize quote-liquidity confidence for selected forward pairs.

    Args:
        report_pairs: Pair-level diagnostics included in the parity report.

    Returns:
        ``"low"`` when selected last-price pairs lacked lowercase volume,
        ``"medium"`` when selected mids came from precomputed mids without bid/ask
        spreads, and ``"high"`` otherwise.
    """
    selected_pairs = [pair for pair in report_pairs if pair["selected"]]
    if any(
        pair["price_source"] == "last_price"
        and pair["volume_filter_status"] == "unavailable"
        for pair in selected_pairs
    ):
        return "low"
    if any(
        pair["price_source"] == "mid"
        and pair["call_relative_spread"] is None
        and pair["put_relative_spread"] is None
        for pair in selected_pairs
    ):
        return "medium"
    return "high"


def _validate_discount_factor(discount_factor: float) -> float:
    """Validate and normalize a discount factor.

    Args:
        discount_factor: Present-value discount factor to validate.

    Returns:
        Discount factor as a float when finite and strictly positive.

    Raises:
        ValueError: If ``discount_factor`` is not finite and strictly positive.
    """
    try:
        normalized_discount_factor = float(discount_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "discount_factor must be finite and strictly positive."
        ) from exc

    if not np.isfinite(normalized_discount_factor) or normalized_discount_factor <= 0.0:
        raise ValueError("discount_factor must be finite and strictly positive.")

    return normalized_discount_factor


def _validate_max_forward_pairs(max_forward_pairs: int) -> int:
    """Validate the nearest-ATM parity-pair cap.

    Args:
        max_forward_pairs: Maximum number of valid pairs to use for inference.

    Returns:
        Positive integer pair cap.

    Raises:
        ValueError: If ``max_forward_pairs`` is not a positive integer.
    """
    if not isinstance(max_forward_pairs, int) or isinstance(max_forward_pairs, bool):
        raise ValueError("max_forward_pairs must be a positive integer.")
    if max_forward_pairs < 1:
        raise ValueError("max_forward_pairs must be a positive integer.")
    return max_forward_pairs


def _validate_min_last_leg_volume(
    min_last_leg_volume: Optional[float],
) -> Optional[float]:
    """Validate the last-price fallback volume floor.

    Args:
        min_last_leg_volume: Minimum per-leg volume for last-price fallback, or
            ``None`` to disable the volume floor.

    Returns:
        Normalized non-negative volume floor or ``None``.

    Raises:
        ValueError: If the provided floor is negative or non-finite.
    """
    if min_last_leg_volume is None:
        return None
    if isinstance(min_last_leg_volume, bool):
        raise ValueError("min_last_leg_volume must be non-negative or None.")
    try:
        normalized_volume = float(min_last_leg_volume)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_last_leg_volume must be non-negative or None.") from exc
    if not np.isfinite(normalized_volume) or normalized_volume < 0.0:
        raise ValueError("min_last_leg_volume must be non-negative or None.")
    return normalized_volume


def _validate_max_bid_ask_relative_spread(
    max_bid_ask_relative_spread: Optional[float],
) -> Optional[float]:
    """Validate the bid/ask relative-spread ceiling.

    Args:
        max_bid_ask_relative_spread: Maximum relative spread for bid/ask mids, or
            ``None`` to disable the spread ceiling.

    Returns:
        Normalized positive spread ceiling or ``None``.

    Raises:
        ValueError: If the provided ceiling is non-positive or non-finite.
    """
    if max_bid_ask_relative_spread is None:
        return None
    if isinstance(max_bid_ask_relative_spread, bool):
        raise ValueError("max_bid_ask_relative_spread must be positive or None.")
    try:
        normalized_spread = float(max_bid_ask_relative_spread)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_bid_ask_relative_spread must be positive or None."
        ) from exc
    if not np.isfinite(normalized_spread) or normalized_spread <= 0.0:
        raise ValueError("max_bid_ask_relative_spread must be positive or None.")
    return normalized_spread


def _collect_forward_candidate_pairs(
    options_df: pd.DataFrame,
    underlying_price: float,
) -> list[dict[str, Any]]:
    """Collect viable same-strike call-put pairs for forward inference.

    Args:
        options_df: Option quotes in one of the supported parity formats.
        underlying_price: Observed underlying spot price used to rank candidate
            strikes by ATM proximity.

    Returns:
        Candidate call-put pairs with a strike, call price, put price, and ATM
        distance.

    Raises:
        ValueError: If the structure of ``options_df`` is not recognised.
    """
    if {"call_price", "put_price"}.issubset(options_df.columns):
        return _collect_wide_forward_candidate_pairs(
            options_df=options_df,
            underlying_price=underlying_price,
        )

    if _has_supported_row_price_columns(options_df):
        return _collect_row_forward_candidate_pairs(
            options_df=options_df,
            underlying_price=underlying_price,
        )

    raise ValueError(
        "Expected DataFrame with either (call_price, put_price) columns or "
        "option_type rows with bid/ask, mid, or last_price columns."
    )


def _has_supported_row_price_columns(options_df: pd.DataFrame) -> bool:
    """Check whether row-format quote data has a supported price source.

    Args:
        options_df: Option quote table to inspect.

    Returns:
        ``True`` when the table has an ``option_type`` column plus bid/ask, ``mid``,
        or ``last_price`` columns; otherwise ``False``.
    """
    if "option_type" not in options_df.columns:
        return False

    return (
        {"bid", "ask"}.issubset(options_df.columns)
        or "mid" in options_df.columns
        or "last_price" in options_df.columns
    )


def _collect_wide_forward_candidate_pairs(
    options_df: pd.DataFrame,
    underlying_price: float,
) -> list[dict[str, Any]]:
    """Collect forward-inference candidates from explicit call/put columns.

    Args:
        options_df: Option quotes containing explicit ``call_price`` and
            ``put_price`` columns.
        underlying_price: Observed underlying spot price used to rank candidate
            strikes by ATM proximity.

    Returns:
        Candidate call-put pairs for each row with a finite strike. Explicit price
        validation happens later so diagnostics can report invalid wide pairs.
    """
    candidate_pairs: list[dict[str, Any]] = []

    for _, row in options_df.iterrows():
        strike = _optional_float(row.get("strike"))
        if strike is None:
            continue

        candidate_pairs.append(
            _build_forward_candidate(
                strike=strike,
                call_price=_optional_float(row.get("call_price")),
                put_price=_optional_float(row.get("put_price")),
                price_source="explicit",
                underlying_price=underlying_price,
            )
        )

    return candidate_pairs


def _collect_row_forward_candidate_pairs(
    options_df: pd.DataFrame,
    underlying_price: float,
) -> list[dict[str, Any]]:
    """Collect forward-inference candidates from long call/put rows.

    Args:
        options_df: Option quotes containing an ``option_type`` column.
        underlying_price: Observed underlying spot price used to rank candidate
            strikes by ATM proximity.

    Returns:
        Candidate call-put pairs using both-leg bid/ask mids when available, and
        both-leg ``last_price`` otherwise.
    """
    candidate_pairs: list[dict[str, Any]] = []

    for strike in options_df["strike"].unique():
        strike_data = options_df[options_df["strike"] == strike]
        option_type = _normalize_option_type_series(strike_data["option_type"])
        call_data = strike_data[option_type == "C"]
        put_data = strike_data[option_type == "P"]

        if len(call_data) == 0 or len(put_data) == 0:
            continue

        candidate_pairs.append(
            _build_forward_candidate(
                strike=float(strike),
                underlying_price=underlying_price,
                call_row=call_data.iloc[0],
                put_row=put_data.iloc[0],
            )
        )

    return candidate_pairs


def _select_forward_price_pair(
    pair: dict[str, Any],
    min_last_leg_volume: Optional[float],
    max_bid_ask_relative_spread: Optional[float],
) -> dict[str, Any]:
    """Select coherent call and put prices for one strike.

    Args:
        pair: Same-strike candidate pair.
        min_last_leg_volume: Minimum per-leg volume for last-price fallback, or
            ``None`` to disable the floor.
        max_bid_ask_relative_spread: Maximum bid/ask relative spread accepted for
            midquote inference, or ``None`` to disable the ceiling.

    Returns:
        Price-selection result containing validity, selected prices, source, and
        diagnostics.
    """
    diagnostics = {
        "call_relative_spread": None,
        "put_relative_spread": None,
        "call_volume": None,
        "put_volume": None,
        "volume_filter_status": "not_applicable",
    }

    if pair.get("price_source") == "explicit":
        return _build_price_selection_result(
            valid=_has_valid_price_pair(pair.get("call_price"), pair.get("put_price")),
            call_price=_optional_float(pair.get("call_price")),
            put_price=_optional_float(pair.get("put_price")),
            price_source="explicit",
            diagnostics=diagnostics,
            excluded_reason="invalid_explicit_price",
        )

    call_row = pair.get("call_row")
    put_row = pair.get("put_row")
    bid_ask_selection = _select_bid_ask_mid_price_pair(
        call_row=call_row,
        put_row=put_row,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    )
    diagnostics.update(bid_ask_selection["diagnostics"])
    if bid_ask_selection["valid"]:
        return bid_ask_selection

    precomputed_mid_selection = _select_precomputed_mid_price_pair(
        call_row=call_row,
        put_row=put_row,
        bid_ask_selection=bid_ask_selection,
    )
    if precomputed_mid_selection["valid"]:
        precomputed_mid_selection["diagnostics"].update(diagnostics)
        return precomputed_mid_selection

    last_price_selection = _select_last_price_pair(
        call_row=call_row,
        put_row=put_row,
        min_last_leg_volume=min_last_leg_volume,
        base_diagnostics=diagnostics,
    )
    if last_price_selection["valid"]:
        return last_price_selection

    diagnostics.update(last_price_selection["diagnostics"])
    return _build_price_selection_result(
        valid=False,
        call_price=None,
        put_price=None,
        price_source="unavailable",
        diagnostics=diagnostics,
        excluded_reason=last_price_selection["excluded_reason"]
        or bid_ask_selection["excluded_reason"]
        or "no_coherent_price_source",
    )


def _select_bid_ask_mid_price_pair(
    call_row: Optional[pd.Series],
    put_row: Optional[pd.Series],
    max_bid_ask_relative_spread: Optional[float],
) -> dict[str, Any]:
    """Select both-leg bid/ask mids when quotes are coherent and tight enough.

    Args:
        call_row: Raw call quote row.
        put_row: Raw put quote row.
        max_bid_ask_relative_spread: Maximum allowed relative spread, or ``None``.

    Returns:
        Price-selection result for bid/ask mids.
    """
    call_mid = _calculate_bid_ask_mid_with_spread(
        call_row,
        max_bid_ask_relative_spread,
    )
    put_mid = _calculate_bid_ask_mid_with_spread(
        put_row,
        max_bid_ask_relative_spread,
    )
    diagnostics = {
        "call_relative_spread": call_mid["relative_spread"],
        "put_relative_spread": put_mid["relative_spread"],
        "call_volume": None,
        "put_volume": None,
        "volume_filter_status": "not_applicable",
    }

    if call_mid["valid"] and put_mid["valid"]:
        return _build_price_selection_result(
            valid=True,
            call_price=call_mid["mid"],
            put_price=put_mid["mid"],
            price_source="mid",
            diagnostics=diagnostics,
            excluded_reason=None,
        )

    excluded_reason = _combine_quote_rejection_reasons(
        call_mid["excluded_reason"],
        put_mid["excluded_reason"],
    )
    return _build_price_selection_result(
        valid=False,
        call_price=None,
        put_price=None,
        price_source="mid",
        diagnostics=diagnostics,
        excluded_reason=excluded_reason,
    )


def _select_precomputed_mid_price_pair(
    call_row: Optional[pd.Series],
    put_row: Optional[pd.Series],
    bid_ask_selection: dict[str, Any],
) -> dict[str, Any]:
    """Select a precomputed mid column only when bid/ask quotes are unavailable.

    Args:
        call_row: Raw call quote row.
        put_row: Raw put quote row.
        bid_ask_selection: Result from bid/ask mid validation.

    Returns:
        Price-selection result for precomputed mids.
    """
    if _has_finite_bid_ask_values(call_row) or _has_finite_bid_ask_values(put_row):
        return _build_price_selection_result(
            valid=False,
            call_price=None,
            put_price=None,
            price_source="mid",
            diagnostics=dict(bid_ask_selection["diagnostics"]),
            excluded_reason=bid_ask_selection["excluded_reason"],
        )

    call_mid = _extract_positive_value(call_row, "mid")
    put_mid = _extract_positive_value(put_row, "mid")
    diagnostics = {
        "call_relative_spread": None,
        "put_relative_spread": None,
        "call_volume": None,
        "put_volume": None,
        "volume_filter_status": "not_applicable",
    }
    return _build_price_selection_result(
        valid=_has_valid_price_pair(call_mid, put_mid),
        call_price=_optional_float(call_mid),
        put_price=_optional_float(put_mid),
        price_source="mid",
        diagnostics=diagnostics,
        excluded_reason="invalid_precomputed_mid_pair",
    )


def _select_last_price_pair(
    call_row: Optional[pd.Series],
    put_row: Optional[pd.Series],
    min_last_leg_volume: Optional[float],
    base_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Select both-leg last prices after validating lowercase volume.

    Args:
        call_row: Raw call quote row.
        put_row: Raw put quote row.
        min_last_leg_volume: Minimum per-leg volume, or ``None``.
        base_diagnostics: Diagnostics already collected from bid/ask validation.

    Returns:
        Price-selection result for last-price fallback.
    """
    call_last = _extract_last_price(call_row)
    put_last = _extract_last_price(put_row)
    diagnostics = dict(base_diagnostics)
    volume_diagnostics = _evaluate_last_price_volume_filter(
        call_row=call_row,
        put_row=put_row,
        min_last_leg_volume=min_last_leg_volume,
    )
    diagnostics.update(volume_diagnostics)

    if not _has_valid_price_pair(call_last, put_last):
        return _build_price_selection_result(
            valid=False,
            call_price=_optional_float(call_last),
            put_price=_optional_float(put_last),
            price_source="last_price",
            diagnostics=diagnostics,
            excluded_reason="invalid_last_price_pair",
        )

    if volume_diagnostics["volume_filter_status"] == "failed":
        return _build_price_selection_result(
            valid=False,
            call_price=_optional_float(call_last),
            put_price=_optional_float(put_last),
            price_source="last_price",
            diagnostics=diagnostics,
            excluded_reason="last_price_volume_below_minimum",
        )

    return _build_price_selection_result(
        valid=True,
        call_price=float(call_last),
        put_price=float(put_last),
        price_source="last_price",
        diagnostics=diagnostics,
        excluded_reason=None,
    )


def _build_price_selection_result(
    valid: bool,
    call_price: Optional[float],
    put_price: Optional[float],
    price_source: str,
    diagnostics: dict[str, Any],
    excluded_reason: Optional[str],
) -> dict[str, Any]:
    """Build a normalized quote-source selection result.

    Args:
        valid: Whether the selected source is usable.
        call_price: Selected call premium, if any.
        put_price: Selected put premium, if any.
        price_source: Source label for the selected or rejected price pair.
        diagnostics: Pair-level quote and liquidity diagnostics.
        excluded_reason: Reason used when ``valid`` is ``False``.

    Returns:
        Normalized selection dictionary.
    """
    pair_fields = {
        "call_price": call_price,
        "put_price": put_price,
        "price_source": price_source,
        "call_relative_spread": diagnostics.get("call_relative_spread"),
        "put_relative_spread": diagnostics.get("put_relative_spread"),
        "call_volume": diagnostics.get("call_volume"),
        "put_volume": diagnostics.get("put_volume"),
        "volume_filter_status": diagnostics.get(
            "volume_filter_status",
            "not_applicable",
        ),
    }
    selection_diagnostics = dict(pair_fields)
    return {
        "valid": valid,
        "pair_fields": pair_fields,
        "diagnostics": selection_diagnostics,
        "excluded_reason": None if valid else excluded_reason,
    }


def _calculate_bid_ask_mid_with_spread(
    option_row: Optional[pd.Series],
    max_bid_ask_relative_spread: Optional[float],
) -> dict[str, Any]:
    """Calculate a bid/ask mid and relative spread for one leg.

    Args:
        option_row: Option quote row.
        max_bid_ask_relative_spread: Maximum allowed relative spread, or ``None``.

    Returns:
        Dictionary containing validity, midpoint, relative spread, and reason.
    """
    if option_row is None or not _has_bid_ask_columns(option_row):
        return {
            "valid": False,
            "mid": None,
            "relative_spread": None,
            "excluded_reason": "bid_ask_unavailable",
        }

    try:
        bid = float(option_row["bid"])
        ask = float(option_row["ask"])
    except (TypeError, ValueError):
        return {
            "valid": False,
            "mid": None,
            "relative_spread": None,
            "excluded_reason": "invalid_bid_ask",
        }

    if not np.isfinite(bid) or not np.isfinite(ask) or bid <= 0.0 or ask <= bid:
        return {
            "valid": False,
            "mid": None,
            "relative_spread": None,
            "excluded_reason": "invalid_bid_ask",
        }

    mid = (bid + ask) / 2.0
    relative_spread = (ask - bid) / mid
    if (
        max_bid_ask_relative_spread is not None
        and relative_spread > max_bid_ask_relative_spread
    ):
        return {
            "valid": False,
            "mid": mid,
            "relative_spread": relative_spread,
            "excluded_reason": "bid_ask_spread_too_wide",
        }

    return {
        "valid": True,
        "mid": mid,
        "relative_spread": relative_spread,
        "excluded_reason": None,
    }


def _evaluate_last_price_volume_filter(
    call_row: Optional[pd.Series],
    put_row: Optional[pd.Series],
    min_last_leg_volume: Optional[float],
) -> dict[str, Any]:
    """Evaluate lowercase-volume liquidity for last-price fallback.

    Args:
        call_row: Raw call quote row.
        put_row: Raw put quote row.
        min_last_leg_volume: Minimum per-leg volume, or ``None``.

    Returns:
        Pair-level volume diagnostics.
    """
    call_volume = _extract_optional_lowercase_volume(call_row)
    put_volume = _extract_optional_lowercase_volume(put_row)
    observed_volumes = [
        volume for volume in (call_volume, put_volume) if volume is not None
    ]

    if min_last_leg_volume is None:
        volume_filter_status = "disabled"
    elif any(volume < min_last_leg_volume for volume in observed_volumes):
        volume_filter_status = "failed"
    elif call_volume is None or put_volume is None:
        volume_filter_status = "unavailable"
    elif call_volume >= min_last_leg_volume and put_volume >= min_last_leg_volume:
        volume_filter_status = "passed"
    else:
        volume_filter_status = "failed"

    return {
        "call_volume": call_volume,
        "put_volume": put_volume,
        "volume_filter_status": volume_filter_status,
    }


def _combine_quote_rejection_reasons(
    call_reason: Optional[str],
    put_reason: Optional[str],
) -> Optional[str]:
    """Combine per-leg quote rejection reasons into one pair-level reason.

    Args:
        call_reason: Call-leg rejection reason.
        put_reason: Put-leg rejection reason.

    Returns:
        Pair-level rejection reason.
    """
    if (
        call_reason == "bid_ask_spread_too_wide"
        or put_reason == "bid_ask_spread_too_wide"
    ):
        return "bid_ask_spread_too_wide"
    if call_reason == put_reason:
        return call_reason
    if call_reason is None:
        return put_reason
    if put_reason is None:
        return call_reason
    return "invalid_bid_ask_pair"


def _has_bid_ask_columns(option_row: Optional[pd.Series]) -> bool:
    """Check whether an option row carries explicit bid and ask columns.

    Args:
        option_row: Option quote row.

    Returns:
        ``True`` when both bid and ask columns are present.
    """
    return option_row is not None and {"bid", "ask"}.issubset(option_row.index)


def _has_finite_bid_ask_values(option_row: Optional[pd.Series]) -> bool:
    """Check whether an option row carries observable bid/ask values.

    Args:
        option_row: Option quote row.

    Returns:
        ``True`` when both bid and ask are present and finite. The values may still
        be invalid as quotes; this helper only distinguishes observable quotes from
        unavailable or NaN placeholders.
    """
    if option_row is None or not _has_bid_ask_columns(option_row):
        return False
    try:
        bid = float(option_row["bid"])
        ask = float(option_row["ask"])
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(bid) and np.isfinite(ask))


def _normalize_option_type_series(option_types: pd.Series) -> pd.Series:
    """Normalize option type labels to one-letter call/put markers.

    Args:
        option_types: Raw option type labels such as ``C``, ``P``, ``call``, or
            ``put``.

    Returns:
        Series containing uppercase first-letter markers.
    """
    return option_types.astype(str).str.strip().str.upper().str[0]


def _build_forward_candidate(
    strike: float,
    underlying_price: float,
    call_price: Optional[float] = None,
    put_price: Optional[float] = None,
    price_source: str = "unavailable",
    call_row: Optional[pd.Series] = None,
    put_row: Optional[pd.Series] = None,
) -> dict[str, Any]:
    """Build a forward-inference candidate record.

    Args:
        strike: Strike shared by the call and put quote.
        underlying_price: Observed underlying spot price used to rank candidate
            strikes by ATM proximity.
        call_price: Selected call premium.
        put_price: Selected put premium.
        price_source: Coherent source used for both prices.
        call_row: Optional raw call row for row-format quotes.
        put_row: Optional raw put row for row-format quotes.

    Returns:
        Candidate record used by the forward inference routine.
    """
    candidate = {
        "strike": strike,
        "call_price": call_price,
        "put_price": put_price,
        "price_source": price_source,
        "distance_from_underlying": abs(strike - underlying_price),
    }
    if call_row is not None:
        candidate["call_row"] = call_row
    if put_row is not None:
        candidate["put_row"] = put_row
    return candidate


def _has_valid_price_pair(first_price: float, second_price: float) -> bool:
    """Check whether two prices are both usable positive values.

    Args:
        first_price: First price to validate.
        second_price: Second price to validate.

    Returns:
        ``True`` when both inputs are finite positive prices; otherwise ``False``.
    """
    return _is_valid_positive_price(first_price) and _is_valid_positive_price(
        second_price
    )


def _optional_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float when possible.

    Args:
        value: Value to normalize.

    Returns:
        Finite float, or ``None`` when the value is missing or non-numeric.
    """
    try:
        if pd.notna(value) and np.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _is_valid_positive_price(price: float) -> bool:
    """Check whether a price is present and strictly positive.

    Args:
        price: Price-like value to validate.

    Returns:
        ``True`` when ``price`` can be interpreted as a positive finite float.
    """
    try:
        return bool(pd.notna(price) and np.isfinite(float(price)) and float(price) > 0)
    except (TypeError, ValueError):
        return False


def _extract_positive_value(option_row: pd.Series, column_name: str) -> float:
    """Extract a positive numeric value from a row column.

    Args:
        option_row: Row potentially containing ``column_name``.
        column_name: Name of the price column to extract.

    Returns:
        Positive value if available; otherwise ``np.nan``.
    """
    try:
        if column_name in option_row and _is_valid_positive_price(
            option_row[column_name]
        ):
            return float(option_row[column_name])
    except Exception:
        return float("nan")
    return float("nan")


def _extract_optional_lowercase_volume(
    option_row: Optional[pd.Series],
) -> Optional[float]:
    """Extract canonical lowercase volume for diagnostics.

    Args:
        option_row: Row potentially containing ``volume``.

    Returns:
        Finite lowercase volume, or ``None`` when unavailable. Legacy uppercase
        ``Volume`` is intentionally ignored.
    """
    if option_row is None:
        return None
    try:
        if "volume" in option_row and pd.notna(option_row["volume"]):
            volume = float(option_row["volume"])
            if np.isfinite(volume):
                return volume
    except (TypeError, ValueError):
        return None
    return None


def apply_put_call_parity_to_quotes(
    options_df: pd.DataFrame,
    forward_price: float,
    discount_factor: float,
) -> pd.DataFrame:
    """Convert quotes into one synthetic call per strike via parity.

    Args:
        options_df: Option quotes containing either separate call/put columns or
            rows distinguished by ``option_type`` and priced with bid/ask, ``mid``,
            or ``last_price`` columns.
        forward_price: Forward price inferred for the valuation date.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.

    Returns:
        Quote table normalised to a single synthetic call per strike.

    Raises:
        ValueError: If the structure of ``options_df`` is not recognised or no valid
            quotes remain once parity adjustments are applied.
    """
    results: list[dict[str, float | str]] = []

    if {"call_price", "put_price"}.issubset(options_df.columns):
        for _, row in options_df.iterrows():
            strike = float(row["strike"])
            call_price = row.get("call_price")
            call_data = row if pd.notna(call_price) and float(call_price) > 0 else None

            put_price = row.get("put_price")
            put_data = row if pd.notna(put_price) and float(put_price) > 0 else None

            _process_strike_prices(
                results=results,
                strike=strike,
                call_data=call_data,
                put_data=put_data,
                forward_price=forward_price,
                discount_factor=discount_factor,
                call_price_column="call_price",
                put_price_column="put_price",
            )
    elif _has_supported_row_price_columns(options_df):
        strikes = options_df["strike"].unique()
        for strike in strikes:
            strike_data = options_df[options_df["strike"] == strike]
            option_type = _normalize_option_type_series(strike_data["option_type"])
            call_data = strike_data[option_type == "C"]
            put_data = strike_data[option_type == "P"]

            call_row = call_data.iloc[0] if len(call_data) > 0 else None
            put_row = put_data.iloc[0] if len(put_data) > 0 else None

            _process_strike_prices(
                results=results,
                strike=float(strike),
                call_data=call_row,
                put_data=put_row,
                forward_price=forward_price,
                discount_factor=discount_factor,
            )
    else:
        raise ValueError(
            "Expected DataFrame with either (call_price, put_price) columns or "
            "option_type rows with bid/ask, mid, or last_price columns."
        )

    if not results:
        raise ValueError("No valid option prices after put-call parity conversion.")

    result_df = pd.DataFrame(results).sort_values("strike").reset_index(drop=True)
    result_df["F_used"] = forward_price
    result_df["DF_used"] = discount_factor

    original_cols = set(options_df.columns) - {
        "call_price",
        "put_price",
        "option_type",
        "last_price",
        "mid",
        "strike",
        "Volume",
        "volume",
        "bid",
        "ask",
    }
    for col in original_cols:
        if col in options_df.columns:
            result_df = result_df.merge(
                options_df[["strike", col]].drop_duplicates("strike"),
                on="strike",
                how="left",
            )

    return result_df.sort_values("strike").reset_index(drop=True)


def detect_parity_opportunity(options_df: pd.DataFrame) -> bool:
    """Check whether parity preprocessing is likely to help.

    Args:
        options_df: Option quotes in either supported format.

    Returns:
        ``True`` when at least one same-strike call-put pair with positive prices is
        present; otherwise ``False``.
    """
    min_last_leg_volume = _DEFAULT_MIN_LAST_LEG_VOLUME
    max_bid_ask_relative_spread = _DEFAULT_MAX_BID_ASK_RELATIVE_SPREAD
    return _has_parity_opportunity(
        options_df=options_df,
        min_last_leg_volume=min_last_leg_volume,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    )


def _has_parity_opportunity(
    options_df: pd.DataFrame,
    min_last_leg_volume: Optional[float],
    max_bid_ask_relative_spread: Optional[float],
) -> bool:
    """Check for at least one coherently priced same-strike call-put pair.

    Args:
        options_df: Option quotes in either supported format.
        min_last_leg_volume: Minimum last-price volume floor.
        max_bid_ask_relative_spread: Maximum bid/ask relative spread.

    Returns:
        ``True`` when a candidate can pass quote-source validation.
    """
    if options_df.empty:
        return False

    try:
        if {"call_price", "put_price"}.issubset(options_df.columns):
            candidate_pairs = _collect_wide_forward_candidate_pairs(
                options_df=options_df,
                underlying_price=0.0,
            )
            return any(
                _select_forward_price_pair(
                    pair=pair,
                    min_last_leg_volume=min_last_leg_volume,
                    max_bid_ask_relative_spread=max_bid_ask_relative_spread,
                )["valid"]
                for pair in candidate_pairs
            )

        if _has_supported_row_price_columns(options_df):
            candidate_pairs = _collect_row_forward_candidate_pairs(
                options_df=options_df,
                underlying_price=0.0,
            )
            for pair in candidate_pairs:
                price_pair = _select_forward_price_pair(
                    pair=pair,
                    min_last_leg_volume=min_last_leg_volume,
                    max_bid_ask_relative_spread=max_bid_ask_relative_spread,
                )
                if price_pair["valid"]:
                    return True
    except Exception:
        return False

    return False


def preprocess_with_parity(
    options_df: pd.DataFrame,
    underlying_price: float,
    discount_factor: float,
    *,
    max_forward_pairs: int = _DEFAULT_MAX_FORWARD_PAIRS,
    min_last_leg_volume: Optional[float] = _DEFAULT_MIN_LAST_LEG_VOLUME,
    max_bid_ask_relative_spread: Optional[float] = _DEFAULT_MAX_BID_ASK_RELATIVE_SPREAD,
) -> pd.DataFrame:
    """Produce parity-adjusted quotes when beneficial.

    Args:
        options_df: Option quotes in either supported format.
        underlying_price: Observed underlying spot price.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.
        max_forward_pairs: Maximum number of nearest-ATM valid parity pairs to use.
        min_last_leg_volume: Minimum canonical lowercase ``volume`` required on
            each leg when ``last_price`` fallback is used.
        max_bid_ask_relative_spread: Maximum relative bid/ask spread accepted for
            bid/ask-mid forward inference.

    Returns:
        Quote table containing usable price columns for downstream processing. When
        parity adjustments succeed the table also includes synthetic call quotes and
        records missing ``last_price`` values as ``NaN`` rather than manufacturing
        last-trade prices.

    Raises:
        ValueError: If the inputs do not contain usable price data.
    """
    max_forward_pairs = _validate_max_forward_pairs(max_forward_pairs)
    min_last_leg_volume = _validate_min_last_leg_volume(min_last_leg_volume)
    max_bid_ask_relative_spread = _validate_max_bid_ask_relative_spread(
        max_bid_ask_relative_spread
    )

    if not _has_parity_opportunity(
        options_df=options_df,
        min_last_leg_volume=min_last_leg_volume,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    ):
        result = options_df.copy()
        if {"bid", "ask"}.issubset(result.columns):
            calculated_mid = (result["bid"] + result["ask"]) / 2
            if "mid" in result.columns:
                result["mid"] = calculated_mid.where(
                    calculated_mid.notna(),
                    result["mid"],
                )
            else:
                result["mid"] = calculated_mid
            return result
        if "last_price" in options_df.columns:
            return result
        if "call_price" in options_df.columns:
            result["last_price"] = result["call_price"]
            return result
        raise ValueError("No usable price data found in options DataFrame.")

    try:
        forward_price, parity_report = _infer_forward_from_put_call_parity_with_report(
            options_df=options_df,
            underlying_price=underlying_price,
            discount_factor=discount_factor,
            max_forward_pairs=max_forward_pairs,
            min_last_leg_volume=min_last_leg_volume,
            max_bid_ask_relative_spread=max_bid_ask_relative_spread,
        )
        cleaned_df = apply_put_call_parity_to_quotes(
            options_df=options_df,
            forward_price=forward_price,
            discount_factor=discount_factor,
        )
        cleaned_df.attrs["parity_report"] = parity_report
        return cleaned_df
    except Exception as exc:
        if "No valid put-call parity pairs found" in str(exc):
            raise ValueError(
                "Put-call parity preprocessing could not infer a forward. " f"{exc}"
            ) from exc
        warnings.warn(
            "Put-call parity preprocessing failed: "
            f"{exc}. Returning original data without F_used.",
            UserWarning,
        )
        if "last_price" in options_df.columns:
            return options_df
        if "call_price" in options_df.columns:
            result = options_df.copy()
            result["last_price"] = result["call_price"]
            return result
        raise ValueError("No usable price data found in options DataFrame.")


def apply_put_call_parity(
    options_data: pd.DataFrame,
    spot: float,
    resolved_market: ResolvedMarket,
    *,
    max_forward_pairs: int = _DEFAULT_MAX_FORWARD_PAIRS,
    min_last_leg_volume: Optional[float] = _DEFAULT_MIN_LAST_LEG_VOLUME,
    max_bid_ask_relative_spread: Optional[float] = _DEFAULT_MAX_BID_ASK_RELATIVE_SPREAD,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """Apply parity preprocessing and infer a forward price when possible.

    Args:
        options_data: Option chain quotes, expected to include either bid/ask pairs
            or explicit call/put columns.
        spot: Observed spot price of the underlying asset.
        resolved_market: Fully resolved market snapshot providing risk-free rate
            and time-to-expiry.
        max_forward_pairs: Maximum number of nearest-ATM valid parity pairs to use.
        min_last_leg_volume: Minimum canonical lowercase ``volume`` required on
            each leg when ``last_price`` fallback is used.
        max_bid_ask_relative_spread: Maximum relative bid/ask spread accepted for
            bid/ask-mid forward inference.

    Returns:
        Tuple containing the parity-adjusted option quotes and an optional forward
        price inferred during preprocessing. ``None`` is returned when a forward
        could not be inferred reliably.

    Raises:
        ValueError: Propagated from :func:`preprocess_with_parity` when the provided
            data cannot support parity adjustments.
    """
    # Calculate time to expiry from the DataFrame's expiry column.
    # We assume the DataFrame contains data for a single expiry (standard for parity checks).
    if "expiry" not in options_data.columns:
        raise ValueError(
            "Options data must contain an 'expiry' column for parity adjustments."
        )

    expiry_val = options_data["expiry"].iloc[0]
    resolved_maturity = resolve_maturity(
        expiry_val,
        resolved_market.valuation_timestamp,
        floor_at_zero=False,
    )
    years_to_expiry = resolved_maturity.time_to_expiry_years
    if years_to_expiry <= 0:
        raise ValueError(
            "Expiry must be strictly after valuation_date for parity adjustments."
        )
    rate_mode = resolved_market.source_meta["risk_free_rate_mode"]
    effective_r = resolve_risk_free_rate(
        resolved_market.risk_free_rate, rate_mode, years_to_expiry
    )
    discount_factor = float(np.exp(-effective_r * years_to_expiry))
    processed = preprocess_with_parity(
        options_data,
        spot,
        discount_factor,
        max_forward_pairs=max_forward_pairs,
        min_last_leg_volume=min_last_leg_volume,
        max_bid_ask_relative_spread=max_bid_ask_relative_spread,
    )
    forward_price: Optional[float] = None
    if "F_used" in processed.columns:
        try:
            forward_price = float(processed["F_used"].iloc[0])
        except Exception:
            forward_price = None
    return processed, forward_price


def _calculate_mid_price(option_row: pd.Series) -> float:
    """Calculate or extract the mid price for a given option quote.

    Args:
        option_row: Row potentially containing ``bid``/``ask`` or a precomputed
            ``mid`` column.

    Returns:
        Bid/ask midpoint when both quotes are positive, then a positive precomputed
        ``mid`` value when available; otherwise ``np.nan``.
    """
    if option_row is None:
        return float("nan")

    try:
        if {"bid", "ask"}.issubset(option_row.index):
            bid = option_row["bid"]
            ask = option_row["ask"]
            if (
                pd.notna(bid)
                and pd.notna(ask)
                and float(bid) > 0
                and float(ask) > float(bid)
            ):
                return float(bid + ask) / 2.0
        if "mid" in option_row and _is_valid_positive_price(option_row["mid"]):
            return float(option_row["mid"])
    except Exception:
        return float("nan")
    return float("nan")


def _extract_bid(option_row: pd.Series) -> float:
    """Extract the bid price from an option row.

    Args:
        option_row: Row potentially containing a ``bid`` column.

    Returns:
        Bid price if present and positive; otherwise ``np.nan``.
    """
    if option_row is None:
        return float("nan")

    try:
        if (
            "bid" in option_row
            and pd.notna(option_row["bid"])
            and float(option_row["bid"]) > 0
        ):
            return float(option_row["bid"])
    except Exception:
        return float("nan")
    return float("nan")


def _extract_ask(option_row: pd.Series) -> float:
    """Extract the ask price from an option row.

    Args:
        option_row: Row potentially containing an ``ask`` column.

    Returns:
        Ask price if present and positive; otherwise ``np.nan``.
    """
    if option_row is None:
        return float("nan")

    try:
        if (
            "ask" in option_row
            and pd.notna(option_row["ask"])
            and float(option_row["ask"]) > 0
        ):
            return float(option_row["ask"])
    except Exception:
        return float("nan")
    return float("nan")


def _extract_last_price(option_row: pd.Series) -> float:
    """Extract the last traded price from an option row.

    Args:
        option_row: Row potentially containing a ``last_price`` column.

    Returns:
        Last traded price if available and positive; otherwise ``np.nan``.
    """
    if option_row is None:
        return float("nan")

    try:
        if (
            "last_price" in option_row
            and pd.notna(option_row["last_price"])
            and float(option_row["last_price"]) > 0
        ):
            return float(option_row["last_price"])
    except Exception:
        return float("nan")
    return float("nan")


def _extract_last_or_explicit_price(
    option_row: pd.Series,
    explicit_price_column: Optional[str],
) -> float:
    """Extract a wide-format explicit price before generic last price.

    Args:
        option_row: Row potentially containing ``last_price`` or an explicit price
            column such as ``call_price`` or ``put_price``.
        explicit_price_column: Optional explicit price column to use before
            generic ``last_price``.

    Returns:
        Explicit leg price when present and positive, then last traded price when
        available; otherwise ``np.nan``.
    """
    if explicit_price_column is not None:
        explicit_price = _extract_positive_value(option_row, explicit_price_column)
        if _is_valid_positive_price(explicit_price):
            return explicit_price

    last_price = _extract_last_price(option_row)
    if _is_valid_positive_price(last_price):
        return last_price

    return float("nan")


def _extract_volume(option_row: pd.Series) -> float:
    """Extract volume from an option row.

    Args:
        option_row: Row potentially containing the canonical ``volume`` column.

    Returns:
        Volume if present; otherwise ``np.nan``.
    """
    if option_row is None:
        return float("nan")
    try:
        if "volume" in option_row and pd.notna(option_row["volume"]):
            return float(option_row["volume"])
    except Exception:
        return float("nan")
    return float("nan")


def _process_strike_prices(
    results: list[dict[str, float | str]],
    strike: float,
    call_data: Optional[pd.Series],
    put_data: Optional[pd.Series],
    forward_price: float,
    discount_factor: float,
    call_price_column: Optional[str] = None,
    put_price_column: Optional[str] = None,
) -> None:
    """Generate the parity-adjusted quote for a single strike.

    Args:
        results: Mutable collection storing parity-adjusted quotes.
        strike: Strike currently being processed.
        call_data: Data row containing call quote information for ``strike``.
        put_data: Data row containing put quote information for ``strike``.
        forward_price: Forward price inferred for the valuation.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.
        call_price_column: Optional explicit wide-format call price column.
        put_price_column: Optional explicit wide-format put price column.
    """
    call_mid = _calculate_mid_price(call_data)
    put_mid = _calculate_mid_price(put_data)
    call_last = _extract_last_or_explicit_price(call_data, call_price_column)
    put_last = _extract_last_or_explicit_price(put_data, put_price_column)
    call_bid = _extract_bid(call_data)
    call_ask = _extract_ask(call_data)
    put_bid = _extract_bid(put_data)
    put_ask = _extract_ask(put_data)

    if strike > forward_price:
        final_mid = (
            call_mid if pd.notna(call_mid) and float(call_mid) > 0 else float("nan")
        )
        final_last = (
            call_last if pd.notna(call_last) and float(call_last) > 0 else float("nan")
        )
        final_bid = (
            call_bid if pd.notna(call_bid) and float(call_bid) > 0 else float("nan")
        )
        final_ask = (
            call_ask if pd.notna(call_ask) and float(call_ask) > 0 else float("nan")
        )
        source = "call"
        volume = _extract_volume(call_data) if call_data is not None else float("nan")
    else:
        final_mid = (
            _convert_put_to_call(put_mid, strike, forward_price, discount_factor)
            if pd.notna(put_mid) and float(put_mid) > 0
            else float("nan")
        )
        final_last = (
            _convert_put_to_call(put_last, strike, forward_price, discount_factor)
            if pd.notna(put_last) and float(put_last) > 0
            else float("nan")
        )
        final_bid = (
            _convert_put_to_call(put_bid, strike, forward_price, discount_factor)
            if pd.notna(put_bid) and float(put_bid) > 0
            else float("nan")
        )
        final_ask = (
            _convert_put_to_call(put_ask, strike, forward_price, discount_factor)
            if pd.notna(put_ask) and float(put_ask) > 0
            else float("nan")
        )
        source = "put_converted"
        volume = _extract_volume(put_data) if put_data is not None else float("nan")

    if pd.isna(final_mid) and pd.isna(final_last):
        return

    intrinsic_value = max(0.0, discount_factor * (forward_price - strike))
    if pd.notna(final_mid):
        final_mid = max(final_mid, intrinsic_value)
    if pd.notna(final_last):
        final_last = max(final_last, intrinsic_value)
    if pd.notna(final_bid):
        final_bid = max(final_bid, intrinsic_value)
    if pd.notna(final_ask):
        final_ask = max(final_ask, intrinsic_value)

    results.append(
        {
            "strike": strike,
            "mid": final_mid,
            "last_price": final_last,
            "bid": final_bid,
            "ask": final_ask,
            "source": source,
            "volume": volume,
        }
    )


def _convert_put_to_call(
    put_price: float,
    strike: float,
    forward_price: float,
    discount_factor: float,
) -> float:
    """Convert a put price into a synthetic call price.

    Args:
        put_price: Observed put premium.
        strike: Strike corresponding to the quote.
        forward_price: Forward price inferred for the valuation.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.

    Returns:
        Synthetic call premium respecting intrinsic value bounds.
    """
    put_forward = put_price / discount_factor
    call_forward = put_forward + (forward_price - strike)
    call_price = call_forward * discount_factor
    return max(0.0, call_price)


__all__ = [
    "apply_put_call_parity",
    "apply_put_call_parity_to_quotes",
    "detect_parity_opportunity",
    "infer_forward_from_atm",
    "infer_forward_from_put_call_parity",
    "preprocess_with_parity",
]
