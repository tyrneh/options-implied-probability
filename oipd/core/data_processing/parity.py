"""Put-call parity preprocessing utilities."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from oipd.market_inputs import ResolvedMarket
from oipd.core.utils import calculate_days_to_expiry


def infer_forward_from_atm(
    options_df: pd.DataFrame,
    underlying_price: float,
    discount_factor: float,
) -> float:
    """Infer the forward price using the closest ATM call-put pair.

    Args:
        options_df: Option quotes containing either ``call_price`` and ``put_price``
            columns or rows distinguished by an ``option_type`` column.
        underlying_price: Observed underlying spot price used to find the ATM strike.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.

    Returns:
        Forward price inferred from the earliest viable call-put pair near ATM.

    Raises:
        ValueError: If the inputs do not contain a same-strike call-put pair or the
            parity calculation cannot produce a positive forward price.
    """
    candidate_pairs: list[dict[str, float]] = []

    if {"call_price", "put_price"}.issubset(options_df.columns):
        for _, row in options_df.iterrows():
            if (
                pd.notna(row["call_price"])
                and pd.notna(row["put_price"])
                and row["call_price"] > 0
                and row["put_price"] > 0
            ):
                candidate_pairs.append(
                    {
                        "strike": float(row["strike"]),
                        "call_price": float(row["call_price"]),
                        "put_price": float(row["put_price"]),
                        "distance_from_underlying": abs(
                            float(row["strike"]) - underlying_price
                        ),
                    }
                )
    elif {"last_price", "option_type"}.issubset(options_df.columns):
        strikes = options_df["strike"].unique()
        for strike in strikes:
            strike_data = options_df[options_df["strike"] == strike]
            call_data = strike_data[strike_data["option_type"] == "C"]
            put_data = strike_data[strike_data["option_type"] == "P"]

            if len(call_data) == 0 or len(put_data) == 0:
                continue

            call_price = _calculate_mid_price(call_data.iloc[0])
            if pd.isna(call_price):
                call_price = _extract_last_price(call_data.iloc[0])

            put_price = _calculate_mid_price(put_data.iloc[0])
            if pd.isna(put_price):
                put_price = _extract_last_price(put_data.iloc[0])

            if (
                pd.notna(call_price)
                and pd.notna(put_price)
                and float(call_price) > 0
                and float(put_price) > 0
            ):
                candidate_pairs.append(
                    {
                        "strike": float(strike),
                        "call_price": float(call_price),
                        "put_price": float(put_price),
                        "distance_from_underlying": abs(
                            float(strike) - underlying_price
                        ),
                    }
                )
    else:
        raise ValueError(
            "Expected DataFrame with either (call_price, put_price) columns or "
            "(last_price, option_type) columns."
        )

    if not candidate_pairs:
        raise ValueError("No strikes found with both call and put options.")

    candidate_pairs.sort(key=lambda item: item["distance_from_underlying"])

    for pair in candidate_pairs:
        try:
            strike = pair["strike"]
            call_price = pair["call_price"]
            put_price = pair["put_price"]
            forward_price = strike + (call_price - put_price) / discount_factor

            intrinsic_call = max(0.0, discount_factor * (forward_price - strike))
            intrinsic_put = max(0.0, discount_factor * (strike - forward_price))
            if call_price < intrinsic_call or put_price < intrinsic_put:
                continue

            if forward_price > 0:
                return forward_price
        except (KeyError, TypeError, ValueError):
            continue

    available_strikes = sorted(pair["strike"] for pair in candidate_pairs)
    raise ValueError(
        f"Could not infer a forward price near {underlying_price:.2f}. "
        f"Available strikes with both options: {available_strikes}"
    )


def apply_put_call_parity_to_quotes(
    options_df: pd.DataFrame,
    forward_price: float,
    discount_factor: float,
) -> pd.DataFrame:
    """Convert quotes into one synthetic call per strike via parity.

    Args:
        options_df: Option quotes containing either separate call/put columns or an
            ``option_type`` column describing calls and puts.
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
            )
    elif {"last_price", "option_type"}.issubset(options_df.columns):
        strikes = options_df["strike"].unique()
        for strike in strikes:
            strike_data = options_df[options_df["strike"] == strike]
            call_data = strike_data[strike_data["option_type"] == "C"]
            put_data = strike_data[strike_data["option_type"] == "P"]

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
            "(last_price, option_type) columns."
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
    if options_df.empty:
        return False

    try:
        if {"call_price", "put_price"}.issubset(options_df.columns):
            has_both_count = 0
            for _, row in options_df.iterrows():
                if (
                    pd.notna(row["call_price"])
                    and pd.notna(row["put_price"])
                    and float(row["call_price"]) > 0
                    and float(row["put_price"]) > 0
                ):
                    has_both_count += 1
            return has_both_count >= 1
        if {"last_price", "option_type"}.issubset(options_df.columns):
            strikes = options_df["strike"].unique()
            has_both_count = 0
            for strike in strikes:
                strike_data = options_df[options_df["strike"] == strike]
                call_data = strike_data[strike_data["option_type"] == "C"]
                put_data = strike_data[strike_data["option_type"] == "P"]

                if len(call_data) == 0 or len(put_data) == 0:
                    continue

                call_price = _calculate_mid_price(call_data.iloc[0])
                if pd.isna(call_price):
                    call_price = _extract_last_price(call_data.iloc[0])

                put_price = _calculate_mid_price(put_data.iloc[0])
                if pd.isna(put_price):
                    put_price = _extract_last_price(put_data.iloc[0])

                if (
                    pd.notna(call_price)
                    and pd.notna(put_price)
                    and float(call_price) > 0
                    and float(put_price) > 0
                ):
                    has_both_count += 1
            return has_both_count >= 1
    except Exception:
        return False

    return False


def preprocess_with_parity(
    options_df: pd.DataFrame,
    underlying_price: float,
    discount_factor: float,
) -> pd.DataFrame:
    """Produce parity-adjusted quotes when beneficial.

    Args:
        options_df: Option quotes in either supported format.
        underlying_price: Observed underlying spot price.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.

    Returns:
        Quote table containing at least a ``last_price`` column for downstream
        processing. When parity adjustments succeed the table also includes
        synthetic call quotes.

    Raises:
        ValueError: If the inputs do not contain usable price data.
    """
    if not detect_parity_opportunity(options_df):
        result = options_df.copy()
        if "last_price" in options_df.columns:
            if {"bid", "ask"}.issubset(result.columns):
                result["mid"] = (result["bid"] + result["ask"]) / 2
            return result
        if "call_price" in options_df.columns:
            result["last_price"] = result["call_price"]
            return result
        raise ValueError("No usable price data found in options DataFrame.")

    try:
        forward_price = infer_forward_from_atm(
            options_df=options_df,
            underlying_price=underlying_price,
            discount_factor=discount_factor,
        )
        cleaned_df = apply_put_call_parity_to_quotes(
            options_df=options_df,
            forward_price=forward_price,
            discount_factor=discount_factor,
        )
        return cleaned_df
    except Exception as exc:
        warnings.warn(
            f"Put-call parity preprocessing failed: {exc}. Using original data.",
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
) -> Tuple[pd.DataFrame, Optional[float]]:
    """Apply parity preprocessing and infer a forward price when possible.

    Args:
        options_data: Option chain quotes, expected to include either bid/ask pairs
            or explicit call/put columns.
        spot: Observed spot price of the underlying asset.
        resolved_market: Fully resolved market snapshot providing risk-free rate
            and time-to-expiry.

    Returns:
        Tuple containing the parity-adjusted option quotes and an optional forward
        price inferred during preprocessing. ``None`` is returned when a forward
        could not be inferred reliably.

    Raises:
        ValueError: Propagated from :func:`preprocess_with_parity` when the provided
            data cannot support parity adjustments.
    """
    # Calculate time to expiry from the DataFrame's expiry column
    # We assume the DataFrame contains data for a single expiry (standard for parity checks)
    if "expiry" not in options_data.columns:
        raise ValueError("Options data must contain an 'expiry' column for parity adjustments.")
    
    expiry_val = options_data["expiry"].iloc[0]
    days_to_expiry = calculate_days_to_expiry(expiry_val, resolved_market.valuation_date)

    discount_factor = float(
        np.exp(-resolved_market.risk_free_rate * days_to_expiry / 365.0)
    )
    processed = preprocess_with_parity(options_data, spot, discount_factor)
    forward_price: Optional[float] = None
    if "F_used" in processed.columns:
        try:
            forward_price = float(processed["F_used"].iloc[0])
        except Exception:
            forward_price = None
    return processed, forward_price


def _calculate_mid_price(option_row: pd.Series) -> float:
    """Calculate the mid price for a given option quote.

    Args:
        option_row: Row containing at least ``bid`` and ``ask`` columns.

    Returns:
        Mid price when both quotes are positive; otherwise ``np.nan``.
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


def _extract_volume(option_row: pd.Series) -> float:
    """Extract volume from an option row, ignoring column casing.

    Args:
        option_row: Row potentially containing ``Volume`` or ``volume`` columns.

    Returns:
        Volume if present; otherwise ``np.nan``.
    """
    if option_row is None:
        return float("nan")
    try:
        for key in ("Volume", "volume"):
            if key in option_row and pd.notna(option_row[key]):
                return float(option_row[key])
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
) -> None:
    """Generate the parity-adjusted quote for a single strike.

    Args:
        results: Mutable collection storing parity-adjusted quotes.
        strike: Strike currently being processed.
        call_data: Data row containing call quote information for ``strike``.
        put_data: Data row containing put quote information for ``strike``.
        forward_price: Forward price inferred for the valuation.
        discount_factor: Present-value discount factor :math:`\\exp(-rT)`.
    """
    call_mid = _calculate_mid_price(call_data)
    put_mid = _calculate_mid_price(put_data)
    call_last = _extract_last_price(call_data)
    put_last = _extract_last_price(put_data)
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
            "Volume": volume,
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
    "preprocess_with_parity",
]
