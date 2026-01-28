"""Unified data loading interface for OIPD."""

from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, Optional, Union, Tuple, List
import pandas as pd
import warnings

from oipd.data_access.readers import CSVReader, DataFrameReader
from oipd.data_access.vendors import get_reader, get_adapter
from oipd.market_inputs import VendorSnapshot


def from_csv(
    path: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load option data from a CSV file.

    Args:
        path: Path to the CSV file.
        column_mapping: Optional mapping from CSV columns to OIPD standard names.
            Standard names: 'strike', 'last_price', 'bid', 'ask', 'volume',
            'open_interest', 'expiry', 'option_type'.

    Returns:
        pd.DataFrame: Cleaned and normalized DataFrame.
    """
    reader = CSVReader()
    return reader.read(path, column_mapping=column_mapping)


def from_dataframe(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load option data from an existing DataFrame.

    Args:
        df: Input DataFrame.
        column_mapping: Optional mapping from input columns to OIPD standard names.

    Returns:
        pd.DataFrame: Cleaned and normalized DataFrame.
    """
    reader = DataFrameReader()
    return reader.read(df, column_mapping=column_mapping)


def list_expiry_dates(
    ticker: str,
    vendor: str = "yfinance",
    **vendor_kwargs: Any,
) -> list[str]:
    """
    List available option expiry dates for a ticker from a vendor.

    Args:
        ticker: Ticker symbol (e.g., "AAPL").
        vendor: Vendor name (default: "yfinance").
        **vendor_kwargs: Vendor-specific keyword arguments forwarded to the
            vendor implementation.

    Returns:
        list[str]: Available expiry dates as strings (YYYY-MM-DD).

    Raises:
        NotImplementedError: If the vendor does not support expiry listing.
    """
    reader_cls = get_reader(vendor)
    list_method = getattr(reader_cls, "list_expiry_dates", None)
    if list_method is None:
        raise NotImplementedError(
            f"Vendor '{vendor}' does not support listing expiry dates."
        )

    try:
        expiry_dates = list_method(ticker, **vendor_kwargs)
    except TypeError:
        reader = reader_cls()
        expiry_dates = reader.list_expiry_dates(ticker, **vendor_kwargs)

    return list(expiry_dates)


def _normalize_expiries(
    expiries: Union[str, date, List[str], List[date]]
) -> List[date]:
    """
    Normalize flexible expiry inputs into a strict list of date objects.
    
    The Funnel Pattern: Loose inputs in -> Strict list of dates out.
    """
    if isinstance(expiries, (str, date)):
        expiries = [expiries]  # type: ignore
    
    normalized: List[date] = []
    for exp in expiries:
        if isinstance(exp, str):
            normalized.append(date.fromisoformat(exp))
        elif isinstance(exp, datetime):
            normalized.append(exp.date())
        elif isinstance(exp, date):
            normalized.append(exp)
        else:
            raise ValueError(f"Invalid expiry type: {type(exp)}. Expected str or date.")
    return normalized


def _filter_expiries_by_horizon(
    available_expiries: List[str], horizon: str
) -> List[date]:
    """
    Filter expiry dates based on a horizon string (e.g., "3m", "1y").
    """
    if not horizon:
        raise ValueError("Horizon cannot be empty")

    unit = horizon[-1].lower()
    try:
        value = int(horizon[:-1])
    except ValueError:
        raise ValueError(f"Invalid horizon format '{horizon}'. Expected e.g. '3m', '1y'.")

    now = datetime.now().date()
    
    if unit == "w":
        cutoff = now + pd.DateOffset(weeks=value)
    elif unit == "m":
        cutoff = now + pd.DateOffset(months=value)
    elif unit == "y":
        cutoff = now + pd.DateOffset(years=value)
    else:
        raise ValueError(f"Unknown horizon unit '{unit}'. use 'w', 'm', or 'y'.")

    # Convert cutoff to date object (DateOffset returns dynamic timestamp behavior)
    # pd.Timestamp(now) + DateOffset results in Timestamp
    cutoff_date = (pd.Timestamp(now) + pd.DateOffset(weeks=value if unit=='w' else 0, 
                                                     months=value if unit=='m' else 0,
                                                     years=value if unit=='y' else 0)).date()

    filtered = []
    for exp_str in available_expiries:
        d = date.fromisoformat(exp_str)
        if now <= d <= cutoff_date:
            filtered.append(d)
            
    return filtered


def fetch_chain(
    ticker: str,
    expiries: Optional[Union[str, date, List[str], List[date]]] = None,
    horizon: Optional[str] = None,
    vendor: str = "yfinance",
    column_mapping: Optional[Dict[str, str]] = None,
    cache_enabled: bool = True,
    cache_ttl_minutes: int = 15,
) -> Tuple[pd.DataFrame, VendorSnapshot]:
    """
    Fetch option chain(s) as a DataFrame and a snapshot of the underlying's price and retrieval time.

    Args:
        ticker: Symbol (e.g. "AAPL").
        expiries: One or more expiry dates. Can be a single string ("2025-01-01"),
            a date object, or a list of them.
        horizon: Time horizon string (e.g. "3m") to automatically select expiries.
                 (Not fully implemented, placeholder)
        vendor: Data provider alias. Currently only "yfinance" is supported.
        column_mapping: Optional dict to map vendor columns to standard names.
        cache_enabled: Whether to cache the network response (default: True).
        cache_ttl_minutes: How long to keep cached data in minutes (default: 15).

    Returns:
        (DataFrame, VendorSnapshot)
    """
    # 1. Validation (Exclusive arguments)
    if expiries is not None and horizon is not None:
        raise ValueError("Ambiguous request: specify 'expiries' OR 'horizon', not both.")
    if expiries is None and horizon is None:
        raise ValueError("Must specify either 'expiries' or 'horizon'.")
    
    # 2. Get adapter
    adapter = get_adapter(vendor)
    
    # 3. Determine target dates
    if horizon is not None:
        # Fetch all available expiries from vendor
        all_expiry_strs = adapter.list_expiry_dates(ticker)
        if not all_expiry_strs:
            raise ValueError(f"No expiries found for {ticker}")
            
        target_dates = _filter_expiries_by_horizon(all_expiry_strs, horizon)
        
        if not target_dates:
             raise ValueError(f"No expiries found within horizon '{horizon}' for {ticker}")

    else:
        # Normalize expiries (Funnel Pattern)
        target_dates = _normalize_expiries(expiries)  # type: ignore

    # 4. Delegate to adapter
    chain, snapshot = adapter.fetch_chain(
        ticker,
        target_dates,
        cache_enabled=cache_enabled,
        cache_ttl_minutes=cache_ttl_minutes,
    )

    # 5. Apply column mapping if provided
    if column_mapping:
        chain = chain.rename(columns=column_mapping)

    return chain, snapshot