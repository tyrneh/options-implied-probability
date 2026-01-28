"""Vendor adapter protocol for data access layer."""

from __future__ import annotations

from datetime import date
from typing import List, Protocol, Tuple, runtime_checkable

import pandas as pd

from oipd.market_inputs import VendorSnapshot


@runtime_checkable
class VendorAdapter(Protocol):
    """Protocol defining the contract for all vendor adapters.

    Each vendor (yfinance, alpaca, binance, etc.) must implement this interface.
    This allows `sources.fetch_chain` to remain vendor-agnostic.
    """

    def fetch_chain(
        self,
        ticker: str,
        expiries: List[date],
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 15,
    ) -> Tuple[pd.DataFrame, VendorSnapshot]:
        """Fetch option chain(s) for a ticker.

        Args:
            ticker: Symbol (e.g. "AAPL").
            expiries: List of expiry dates to fetch.
            cache_enabled: Whether to cache network responses.
            cache_ttl_minutes: Cache TTL in minutes.

        Returns:
            Tuple of (DataFrame with 'expiry' column, VendorSnapshot).
        """
        ...

    def list_expiry_dates(self, ticker: str) -> List[str]:
        """List available expiry dates for a ticker.

        Args:
            ticker: Symbol (e.g. "AAPL").

        Returns:
            List of expiry date strings in YYYY-MM-DD format.
        """
        ...


__all__ = ["VendorAdapter"]
