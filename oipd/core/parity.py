"""Put-call parity preprocessing for OIPD.

This module provides internal functions to preprocess options data using put-call parity.
The functionality is based on the Aït-Sahalia & Lo approach:
1. Infer forward price from ATM call-put pair
2. Use OTM options only (calls above forward, puts below forward)
3. Convert puts to synthetic calls via put-call parity

This preprocessing happens automatically and transparently when both puts and calls
are available, improving the quality of the risk-neutral density estimation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import warnings


def infer_forward_from_atm(
    options_df: pd.DataFrame, underlying_price: float, discount_factor: float
) -> float:
    """
    Infer forward price from ATM call-put pair using put-call parity.

    Finds the strike closest to spot that has both a call and put available,
    then applies: F = K_atm + (C - P) / DF

    Parameters
    ----------
    options_df : pd.DataFrame
        Options data with columns: strike, call_price, put_price
        (or strike, last_price, option_type)
    underlying_price : float
        Current underlying price (used to find ATM strike)
    discount_factor : float
        exp(-r * T) for present value calculations

    Returns
    -------
    float
        Inferred forward price

    Raises
    ------
    ValueError
        If no same-strike call-put pair can be found
    """
    # Find call-put pairs using the same logic as apply_put_call_parity
    candidate_pairs = []

    if "call_price" in options_df.columns and "put_price" in options_df.columns:
        # Format 1: Already have call_price/put_price columns
        for _, row in options_df.iterrows():
            if (
                pd.notna(row["call_price"])
                and pd.notna(row["put_price"])
                and row["call_price"] > 0
                and row["put_price"] > 0
            ):
                candidate_pairs.append(
                    {
                        "strike": row["strike"],
                        "call_price": row["call_price"],
                        "put_price": row["put_price"],
                        "distance_from_underlying": abs(row["strike"] - underlying_price),
                    }
                )

    elif "last_price" in options_df.columns and "option_type" in options_df.columns:
        # Format 2: Need to find matching call-put pairs
        strikes = options_df["strike"].unique()

        for strike in strikes:
            strike_data = options_df[options_df["strike"] == strike]
            call_data = strike_data[strike_data["option_type"] == "C"]
            put_data = strike_data[strike_data["option_type"] == "P"]

            if len(call_data) > 0 and len(put_data) > 0:
                # Try to get best price for each - prefer mid over last
                call_price = _calculate_mid_price(call_data.iloc[0])
                if pd.isna(call_price):
                    call_price = _extract_last_price(call_data.iloc[0])

                put_price = _calculate_mid_price(put_data.iloc[0])
                if pd.isna(put_price):
                    put_price = _extract_last_price(put_data.iloc[0])

                if (
                    pd.notna(call_price)
                    and pd.notna(put_price)
                    and call_price > 0
                    and put_price > 0
                ):
                    candidate_pairs.append(
                        {
                            "strike": strike,
                            "call_price": call_price,
                            "put_price": put_price,
                            "distance_from_underlying": abs(strike - underlying_price),
                        }
                    )
    else:
        raise ValueError(
            "Expected DataFrame with either (call_price, put_price) columns "
            "or (last_price, option_type) columns"
        )

    if not candidate_pairs:
        raise ValueError("No strikes found with both call and put options")

    # Sort by distance from current underlying price to find ATM
    candidate_pairs.sort(key=lambda x: x["distance_from_underlying"])

    # Try strikes starting from closest to the current price
    for pair in candidate_pairs:
        try:
            k_atm = pair["strike"]
            call_price = pair["call_price"]
            put_price = pair["put_price"]

            # Calculate forward using put-call parity: F = K + (C - P) / DF
            forward_price = k_atm + (call_price - put_price) / discount_factor

            # Bounds check: option prices should not violate intrinsic values
            intrinsic_call = max(0.0, discount_factor * (forward_price - k_atm))
            intrinsic_put = max(0.0, discount_factor * (k_atm - forward_price))
            if call_price < intrinsic_call or put_price < intrinsic_put:
                continue

            if forward_price > 0:
                return forward_price

        except (KeyError, TypeError, ValueError):
            # Skip this strike if data is bad
            continue

    # If we get here, no good ATM pair was found
    available_strikes = sorted([pair["strike"] for pair in candidate_pairs])
    raise ValueError(
        f"Could not find a clean ATM call-put pair near price {underlying_price:.2f}. "
        f"Available strikes with both options: {available_strikes}"
    )


def apply_put_call_parity(
    options_df: pd.DataFrame, forward_price: float, discount_factor: float
) -> pd.DataFrame:
    """
    Apply put-call parity to produce one clean call price per strike.

    For each unique strike in the union of calls and puts:
      * If K > F: use the call price when available; otherwise convert the put
      * If K ≤ F: use the put converted via parity; fallback to call when put missing

    Parameters
    ----------
    options_df : pd.DataFrame
        Options data with either separate call/put columns or option_type format
    forward_price : float
        Inferred forward price
    discount_factor : float
        exp(-r * T)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: strike, mid, last_price, source, F_used, DF_used
    """
    results = []

    # Handle both input formats directly
    if "call_price" in options_df.columns and "put_price" in options_df.columns:
        # Format 1: Separate call_price/put_price columns
        for _, row in options_df.iterrows():
            strike = row["strike"]

            # For this format, we create synthetic call/put data from the row
            # Check if this row has call data
            call_price = row.get("call_price")
            call_data = row if pd.notna(call_price) and call_price > 0 else None

            # Check if this row has put data
            put_price = row.get("put_price")
            put_data = row if pd.notna(put_price) and put_price > 0 else None

            _process_strike_prices(
                results, strike, call_data, put_data, forward_price, discount_factor
            )

    elif "last_price" in options_df.columns and "option_type" in options_df.columns:
        # Format 2: option_type format - get prices by strike
        strikes = options_df["strike"].unique()

        for strike in strikes:
            strike_data = options_df[options_df["strike"] == strike]

            # Extract call and put data for this strike
            call_data = strike_data[strike_data["option_type"] == "C"]
            put_data = strike_data[strike_data["option_type"] == "P"]

            # Pass the actual data objects (or None) to preserve volume info
            call_row = call_data.iloc[0] if len(call_data) > 0 else None
            put_row = put_data.iloc[0] if len(put_data) > 0 else None

            _process_strike_prices(
                results, strike, call_row, put_row, forward_price, discount_factor
            )
    else:
        raise ValueError(
            "Expected DataFrame with either (call_price, put_price) columns "
            "or (last_price, option_type) columns"
        )

    if not results:
        raise ValueError("No valid option prices after put-call parity conversion")

    result_df = pd.DataFrame(results).sort_values("strike").reset_index(drop=True)
    result_df["F_used"] = forward_price
    result_df["DF_used"] = discount_factor

    # Preserve original columns that might be needed downstream
    original_cols = set(options_df.columns) - {
        "call_price",
        "put_price",
        "option_type",
        "last_price",
        "mid",
        "strike",
        "Volume",
        "volume",
    }
    for col in original_cols:
        if col in options_df.columns:
            result_df = result_df.merge(
                options_df[["strike", col]].drop_duplicates("strike"),
                on="strike",
                how="left",
            )

    return result_df.sort_values("strike").reset_index(drop=True)


def _calculate_mid_price(option_row) -> float:
    """Calculate mid price from bid/ask if available."""
    if option_row is None:
        return np.nan

    try:
        if "bid" in option_row and "ask" in option_row:
            bid, ask = option_row["bid"], option_row["ask"]
            if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > bid:
                return (bid + ask) / 2.0
    except Exception:
        pass
    return np.nan


def _extract_last_price(option_row) -> float:
    """Extract last_price if available."""
    if option_row is None:
        return np.nan

    try:
        if (
            "last_price" in option_row
            and pd.notna(option_row["last_price"])
            and option_row["last_price"] > 0
        ):
            return float(option_row["last_price"])
    except Exception:
        pass
    return np.nan


def _extract_volume(option_row) -> float:
    """Extract volume regardless of column casing."""
    if option_row is None:
        return np.nan
    try:
        for key in ("Volume", "volume"):
            if key in option_row and pd.notna(option_row[key]):
                return float(option_row[key])
    except Exception:
        pass
    return np.nan


def _process_strike_prices(
    results, strike, call_data, put_data, forward_price, discount_factor
):
    """Process a single strike's prices through put-call parity logic."""

    # Extract prices
    call_mid = _calculate_mid_price(call_data)
    put_mid = _calculate_mid_price(put_data)
    call_last = _extract_last_price(call_data)
    put_last = _extract_last_price(put_data)

    if strike > forward_price:
        # OTM call region - use ONLY original calls (never convert puts)
        final_mid = call_mid if pd.notna(call_mid) and call_mid > 0 else np.nan
        final_last = call_last if pd.notna(call_last) and call_last > 0 else np.nan
        source = "call"
        # Use call's volume for OTM strikes
        volume = _extract_volume(call_data) if call_data is not None else np.nan
    else:
        # ITM call region - use ONLY synthetic calls from puts (never use original ITM calls)
        final_mid = (
            _convert_put_to_call(put_mid, strike, forward_price, discount_factor)
            if pd.notna(put_mid) and put_mid > 0
            else np.nan
        )
        final_last = (
            _convert_put_to_call(put_last, strike, forward_price, discount_factor)
            if pd.notna(put_last) and put_last > 0
            else np.nan
        )
        source = "put_converted"
        # Use put's volume for ITM strikes (synthetic calls from puts)
        volume = _extract_volume(put_data) if put_data is not None else np.nan

    # Skip strikes where we have no usable prices
    if pd.isna(final_mid) and pd.isna(final_last):
        return

    # Apply intrinsic value bounds
    intrinsic_value = max(0.0, discount_factor * (forward_price - strike))
    if pd.notna(final_mid):
        final_mid = max(final_mid, intrinsic_value)
    if pd.notna(final_last):
        final_last = max(final_last, intrinsic_value)

    results.append(
        {
            "strike": strike,
            "mid": final_mid,
            "last_price": final_last,
            "source": source,
            "Volume": volume,
        }
    )


def _convert_put_to_call(
    put_price: float, strike: float, forward_price: float, discount_factor: float
) -> float:
    """
    Convert put price to synthetic call price using put-call parity.

    Uses the forward-space calculation for numerical stability:
    C_f = P_f + (F - K), then C = C_f * DF

    Parameters
    ----------
    put_price : float
        Put option price
    strike : float
        Strike price
    forward_price : float
        Forward price of underlying
    discount_factor : float
        exp(-r * T) discount factor

    Returns
    -------
    float
        Synthetic call price
    """
    # Convert to forward space for stability
    put_forward = put_price / discount_factor
    call_forward = put_forward + (forward_price - strike)

    # Convert back to present value
    call_price = call_forward * discount_factor

    return max(0.0, call_price)  # Ensure non-negative


def detect_parity_opportunity(options_df: pd.DataFrame) -> bool:
    """
    Detect if put-call parity preprocessing would be beneficial.

    Returns True if both calls and puts are available with reasonable coverage.

    Parameters
    ----------
    options_df : pd.DataFrame
        Options data with either separate call_price/put_price columns
        or option_type indicators

    Returns
    -------
    bool
        True if parity preprocessing should be applied
    """
    if options_df.empty:
        return False

    try:
        # Check coverage using the same logic as other functions
        if "call_price" in options_df.columns and "put_price" in options_df.columns:
            # Format 1: Check for both calls and puts
            has_both_count = 0
            for _, row in options_df.iterrows():
                if (
                    pd.notna(row["call_price"])
                    and pd.notna(row["put_price"])
                    and row["call_price"] > 0
                    and row["put_price"] > 0
                ):
                    has_both_count += 1
            return has_both_count >= 1

        elif "last_price" in options_df.columns and "option_type" in options_df.columns:
            # Format 2: Check for strikes with both calls and puts
            strikes = options_df["strike"].unique()
            has_both_count = 0

            for strike in strikes:
                strike_data = options_df[options_df["strike"] == strike]
                call_data = strike_data[strike_data["option_type"] == "C"]
                put_data = strike_data[strike_data["option_type"] == "P"]

                if len(call_data) > 0 and len(put_data) > 0:
                    # Check if we can extract prices from both
                    call_price = _calculate_mid_price(call_data.iloc[0])
                    if pd.isna(call_price):
                        call_price = _extract_last_price(call_data.iloc[0])

                    put_price = _calculate_mid_price(put_data.iloc[0])
                    if pd.isna(put_price):
                        put_price = _extract_last_price(put_data.iloc[0])

                    if (
                        pd.notna(call_price)
                        and pd.notna(put_price)
                        and call_price > 0
                        and put_price > 0
                    ):
                        has_both_count += 1

            return has_both_count >= 1
        else:
            return False  # Can't detect both option types

    except Exception:
        # If any error in detection, assume no parity opportunity
        return False


def preprocess_with_parity(
    options_df: pd.DataFrame, underlying_price: float, discount_factor: float
) -> pd.DataFrame:
    """
    Convenience function to apply put-call parity preprocessing if beneficial.

    This is the main entry point for estimator.py to use.

    Parameters
    ----------
    options_df : pd.DataFrame
        Options data (various formats supported)
    underlying_price : float
        Current underlying price
    discount_factor : float
        exp(-r * T) for discounting

    Returns
    -------
    pd.DataFrame
        Cleaned options data with 'last_price' column ready for pipeline
    """
    # Check if parity preprocessing would be beneficial
    if not detect_parity_opportunity(options_df):
        # No benefit - return original data (ensure it has 'last_price' column)
        result = options_df.copy()
        if "last_price" in options_df.columns:
            # Add mid column from bid/ask if available
            if "bid" in result.columns and "ask" in result.columns:
                result["mid"] = (result["bid"] + result["ask"]) / 2
            return result
        elif "call_price" in options_df.columns:
            # Rename call_price to last_price for pipeline compatibility
            result["last_price"] = result["call_price"]
            return result
        else:
            raise ValueError("No usable price data found in options DataFrame")

    try:
        # Apply parity preprocessing
        forward_price = infer_forward_from_atm(options_df, underlying_price, discount_factor)
        cleaned_df = apply_put_call_parity(options_df, forward_price, discount_factor)
        return cleaned_df

    except Exception as e:
        # If parity processing fails, fall back to original data
        warnings.warn(
            f"Put-call parity preprocessing failed: {e}. Using original data.",
            UserWarning,
        )
        if "last_price" in options_df.columns:
            return options_df
        elif "call_price" in options_df.columns:
            result = options_df.copy()
            result["last_price"] = result["call_price"]
            return result
        else:
            raise ValueError("No usable price data found in options DataFrame")


__all__ = [
    "infer_forward_from_atm",
    "apply_put_call_parity",
    "detect_parity_opportunity",
    "preprocess_with_parity",
]
