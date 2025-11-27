"""Unified data loading interface for OIPD."""

from __future__ import annotations

from typing import Dict, Optional, Union

import pandas as pd

from oipd.data_access.readers import CSVReader, DataFrameReader
from oipd.data_access.vendors import get_reader


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


def from_ticker(
    ticker: str,
    expiry: Optional[str] = None,
    vendor: str = "yfinance",
    column_mapping: Optional[Dict[str, str]] = None,
    cache_enabled: bool = True,
    cache_ttl_minutes: int = 15,
) -> pd.DataFrame:
    """
    Load option data for a ticker from a vendor (e.g., Yahoo Finance).

    Args:
        ticker: Ticker symbol (e.g., "AAPL").
        expiry: Optional specific expiry date (YYYY-MM-DD). If None, may return all expiries
            or default behavior depending on vendor.
        vendor: Vendor name (default: "yfinance").
        column_mapping: Optional column mapping overrides.
        cache_enabled: Whether to cache the result.
        cache_ttl_minutes: Cache duration in minutes.

    Returns:
        pd.DataFrame: Cleaned and normalized DataFrame.
    """
    reader_cls = get_reader(vendor)
    try:
        reader = reader_cls(
            cache_enabled=cache_enabled,
            cache_ttl_minutes=cache_ttl_minutes,
        )
    except TypeError:
        # Fallback for readers that don't support cache args
        reader = reader_cls()

    # Construct the query string expected by the reader
    # yfinance reader expects "TICKER" or "TICKER:EXPIRY"
    query = ticker
    if expiry:
        query = f"{ticker}:{expiry}"

    return reader.read(query, column_mapping=column_mapping)
