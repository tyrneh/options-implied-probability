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
from typing import Tuple, Optional


def infer_forward_from_atm(
    options_df: pd.DataFrame,
    spot_price: float,
    discount_factor: float
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
    spot_price : float
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
    # Standardize input format
    if 'call_price' in options_df.columns and 'put_price' in options_df.columns:
        df = options_df.copy()
    elif 'last_price' in options_df.columns and 'option_type' in options_df.columns:
        df = _pivot_option_types(options_df)
    else:
        raise ValueError(
            "Expected DataFrame with either (call_price, put_price) columns "
            "or (last_price, option_type) columns"
        )

    # Find strikes that have both calls and puts
    has_both = df.dropna(subset=['call_price', 'put_price'])
    if has_both.empty:
        raise ValueError("No strikes found with both call and put options")

    # Sort by distance from spot price to find ATM
    has_both = has_both.copy()
    has_both['distance_from_spot'] = abs(has_both['strike'] - spot_price)
    has_both = has_both.sort_values('distance_from_spot')

    # Try strikes starting from closest to spot
    for _, row in has_both.iterrows():
        try:
            k_atm = row['strike']
            call_price = row['call_price']
            put_price = row['put_price']

            # Basic sanity check - prices should be positive
            if call_price <= 0 or put_price <= 0:
                continue

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
    available_strikes = sorted(has_both['strike'].tolist())
    raise ValueError(
        f"Could not find a clean ATM call-put pair near spot {spot_price:.2f}. "
        f"Available strikes with both options: {available_strikes}"
    )


def apply_put_call_parity(
    options_df: pd.DataFrame,
    forward_price: float,
    discount_factor: float
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
        DataFrame with columns: strike, last_price, source, F_used, DF_used
    """
    # Detect input format and standardize
    if 'call_price' in options_df.columns and 'put_price' in options_df.columns:
        df = options_df.copy()
    elif 'last_price' in options_df.columns and 'option_type' in options_df.columns:
        df = _pivot_option_types(options_df)
    else:
        raise ValueError(
            "Expected DataFrame with either (call_price, put_price) columns "
            "or (last_price, option_type) columns"
        )

    results = []

    for _, row in df.iterrows():
        strike = row['strike']
        call_price = row.get('call_price')
        put_price = row.get('put_price')

        if strike > forward_price:
            if pd.notna(call_price) and call_price > 0:
                final_price = call_price
                source = "call"
            elif pd.notna(put_price) and put_price > 0:
                final_price = _convert_put_to_call(put_price, strike, forward_price, discount_factor)
                source = "put_converted"
            else:
                continue
        else:
            if pd.notna(put_price) and put_price > 0:
                final_price = _convert_put_to_call(put_price, strike, forward_price, discount_factor)
                source = "put_converted"
            elif pd.notna(call_price) and call_price > 0:
                final_price = call_price
                source = "call"
            else:
                continue

        intrinsic_value = max(0.0, discount_factor * (forward_price - strike))
        final_price = max(final_price, intrinsic_value)

        results.append(
            {
                'strike': strike,
                'last_price': final_price,
                'source': source,
            }
        )

    if not results:
        raise ValueError("No valid option prices after put-call parity conversion")

    result_df = pd.DataFrame(results).sort_values('strike').reset_index(drop=True)
    result_df['F_used'] = forward_price
    result_df['DF_used'] = discount_factor

    # Preserve original columns that might be needed downstream
    original_cols = set(options_df.columns) - {'call_price', 'put_price', 'option_type', 'last_price', 'strike'}
    for col in original_cols:
        if col in options_df.columns:
            result_df = result_df.merge(
                options_df[['strike', col]].drop_duplicates('strike'),
                on='strike',
                how='left'
            )

    return result_df.sort_values('strike').reset_index(drop=True)


def _pivot_option_types(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert option_type format to call_price/put_price format.
    
    Input: DataFrame with columns [strike, option_type] and price fields such as
    bid/ask, mid, or last_price
    Output: DataFrame with columns [strike, call_price, put_price]
    """
    # Compute a unified price column per row
    df = options_df.copy()
    df['price'] = np.nan
    for idx, row in df.iterrows():
        try:
            df.at[idx, 'price'] = _get_option_price(row)
        except Exception:
            df.at[idx, 'price'] = np.nan
    df = df[['strike', 'price', 'option_type']]
    
    try:
        # Pivot to get separate call and put price columns
        pivoted = df.pivot_table(
            index='strike', 
            columns='option_type',
            values='price',
            aggfunc='first'  # Take first value if duplicates
        ).reset_index()
        
        # Handle potential MultiIndex columns
        if isinstance(pivoted.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            pivoted.columns = [col[-1] if col[-1] != '' else col[0] for col in pivoted.columns]
        
        # Rename columns (handle cases where C or P might not exist)
        column_rename = {}
        if 'C' in pivoted.columns:
            column_rename['C'] = 'call_price'
        if 'P' in pivoted.columns:
            column_rename['P'] = 'put_price'
            
        pivoted = pivoted.rename(columns=column_rename)
        
        # Ensure we have the columns we expect
        if 'call_price' not in pivoted.columns:
            pivoted['call_price'] = np.nan
        if 'put_price' not in pivoted.columns:
            pivoted['put_price'] = np.nan
            
        return pivoted
        
    except Exception as e:
        # If pivot fails, try alternative approach using groupby
        try:
            calls = df[df['option_type'] == 'C'][['strike', 'price']].copy()
            calls = calls.rename(columns={'price': 'call_price'})

            puts = df[df['option_type'] == 'P'][['strike', 'price']].copy()
            puts = puts.rename(columns={'price': 'put_price'})
            
            # Merge on strike
            result = pd.merge(calls, puts, on='strike', how='outer')
            
            # Fill missing columns
            if 'call_price' not in result.columns:
                result['call_price'] = np.nan
            if 'put_price' not in result.columns:
                result['put_price'] = np.nan
                
            return result.sort_values('strike').reset_index(drop=True)
            
        except Exception as e2:
            raise ValueError(f"Failed to pivot option types: {e}. Fallback also failed: {e2}")


def _get_option_price(option_row: pd.Series) -> float:
    """Extract a price from a quote row.

    Preference order: explicit ``mid`` column, bid/ask midpoint, then ``last_price``.
    Raises ValueError if no usable price is found.
    """
    # Explicit mid column
    if 'mid' in option_row and pd.notna(option_row['mid']) and option_row['mid'] > 0:
        return float(option_row['mid'])

    # Compute mid from bid/ask
    if 'bid' in option_row and 'ask' in option_row:
        bid = option_row['bid']
        ask = option_row['ask']
        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > bid:
            return (bid + ask) / 2.0

    # Fallback to last_price
    if 'last_price' in option_row:
        last_price = option_row['last_price']
        if pd.notna(last_price) and last_price > 0:
            return float(last_price)

    raise ValueError(f"No valid price found in option row: {option_row.to_dict()}")


def _convert_put_to_call(
    put_price: float, 
    strike: float, 
    forward_price: float, 
    discount_factor: float
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
        # Standardize format to check coverage
        if 'call_price' in options_df.columns and 'put_price' in options_df.columns:
            df = options_df.copy()
        elif 'last_price' in options_df.columns and 'option_type' in options_df.columns:
            df = _pivot_option_types(options_df)
        else:
            return False  # Can't detect both option types
        
        # Check if there's meaningful data
        has_calls = df['call_price'].notna().any()
        has_puts = df['put_price'].notna().any()
        
        if not (has_calls and has_puts):
            return False
        
        # Check for strikes with both calls and puts
        has_both = df.dropna(subset=['call_price', 'put_price'])

        # Need at least one common strike for forward inference
        return len(has_both) >= 1
        
    except Exception:
        # If any error in detection, assume no parity opportunity
        return False


def preprocess_with_parity(
    options_df: pd.DataFrame,
    spot_price: float,
    discount_factor: float
) -> pd.DataFrame:
    """
    Convenience function to apply put-call parity preprocessing if beneficial.
    
    This is the main entry point for estimator.py to use.
    
    Parameters
    ----------
    options_df : pd.DataFrame
        Options data (various formats supported)
    spot_price : float
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
        if 'last_price' in options_df.columns:
            return options_df
        elif 'call_price' in options_df.columns:
            # Rename call_price to last_price for pipeline compatibility
            result = options_df.copy()
            result['last_price'] = result['call_price']
            return result
        else:
            raise ValueError("No usable price data found in options DataFrame")
    
    try:
        # Apply parity preprocessing
        forward_price = infer_forward_from_atm(options_df, spot_price, discount_factor)
        cleaned_df = apply_put_call_parity(options_df, forward_price, discount_factor)
        return cleaned_df
        
    except Exception as e:
        # If parity processing fails, fall back to original data
        warnings.warn(
            f"Put-call parity preprocessing failed: {e}. Using original data.", 
            UserWarning
        )
        if 'last_price' in options_df.columns:
            return options_df
        elif 'call_price' in options_df.columns:
            result = options_df.copy()
            result['last_price'] = result['call_price']
            return result
        else:
            raise ValueError("No usable price data found in options DataFrame")


__all__ = [
    "infer_forward_from_atm",
    "apply_put_call_parity", 
    "detect_parity_opportunity",
    "preprocess_with_parity"
]