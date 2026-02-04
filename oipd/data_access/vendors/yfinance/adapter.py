"""YFinance adapter implementing VendorAdapter protocol."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import List, Optional, Tuple

import pandas as pd

from oipd.data_access.vendors.yfinance.reader import Reader, YFinanceError
from oipd.market_inputs import VendorSnapshot


class YFinanceAdapter:
    """Adapter for fetching option chains from Yahoo Finance.

    Implements the VendorAdapter protocol. Handles multi-expiry fetching,
    DataFrame concatenation, and snapshot construction.
    """

    def fetch_chain(
        self,
        ticker: str,
        expiries: List[date],
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 15,
    ) -> Tuple[pd.DataFrame, VendorSnapshot]:
        """Fetch option chain(s) for multiple expiries.

        Args:
            ticker: Symbol (e.g. "AAPL").
            expiries: List of expiry dates to fetch.
            cache_enabled: Whether to cache network responses.
            cache_ttl_minutes: Cache TTL in minutes.

        Returns:
            Tuple of (DataFrame with 'expiry' column, VendorSnapshot).

        Raises:
            ValueError: If no data found for any requested expiry.
        """
        reader = Reader(
            cache_enabled=cache_enabled,
            cache_ttl_minutes=cache_ttl_minutes,
        )

        chains: List[pd.DataFrame] = []
        primary_snapshot: Optional[VendorSnapshot] = None

        for d in expiries:
            expiry_str = d.isoformat()
            query = f"{ticker}:{expiry_str}"

            try:
                df = reader.read(query)
            except (YFinanceError, Exception) as e:
                warnings.warn(f"Failed to fetch expiry {expiry_str}: {e}")
                continue

            if df.empty:
                continue

            # Ensure expiry column exists
            if "expiry" not in df.columns:
                df["expiry"] = pd.Timestamp(d)

            chains.append(df)

            # Capture snapshot from first valid fetch
            if primary_snapshot is None:
                primary_snapshot = self._build_snapshot(df, ticker)

        if not chains:
            raise ValueError(f"No data found for requested expiries: {expiries}")

        # Concatenate all chains
        final_chain = pd.concat(chains, ignore_index=True)

        # Final strict guarantee
        if "expiry" not in final_chain.columns:
            raise ValueError("Critical: Fetched data is missing 'expiry' column.")

        if primary_snapshot is None:
            raise RuntimeError("Snapshot creation failed.")

        return final_chain, primary_snapshot

    def list_expiry_dates(self, ticker: str) -> List[str]:
        """List available expiry dates for a ticker.

        Args:
            ticker: Symbol (e.g. "AAPL").

        Returns:
            List of expiry date strings in YYYY-MM-DD format.
        """
        return Reader.list_expiry_dates(ticker)

    def _build_snapshot(self, df: pd.DataFrame, ticker: str) -> VendorSnapshot:
        """Build VendorSnapshot from DataFrame attributes."""
        asof_val = df.attrs.get("asof")

        if isinstance(asof_val, str):
            try:
                asof_val = datetime.fromisoformat(asof_val)
            except ValueError:
                asof_val = datetime.now()
        elif not isinstance(asof_val, datetime):
            asof_val = datetime.now()

        df.attrs["asof"] = asof_val

        return VendorSnapshot(
            asof=asof_val,
            vendor="yfinance",
            underlying_price=df.attrs.get("underlying_price"),
            dividend_yield=df.attrs.get("dividend_yield"),
            dividend_schedule=df.attrs.get("dividend_schedule"),
        )


__all__ = ["YFinanceAdapter"]
