from __future__ import annotations

"""YFinance data connector for *oipd*.

Implements :class:`Reader`, compatible with the generic *vendor* registry.
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np

from oipd.io.reader import AbstractReader

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class YFinanceError(Exception):
    """Exception raised when yfinance operations fail"""

    pass


# -----------------------------------------------------------------------------
# Simple file-based cache (unchanged)
# -----------------------------------------------------------------------------


class _YFinanceCache:
    def __init__(self, cache_dir: str = ".yfinance_cache", ttl_minutes: int = 15):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(minutes=ttl_minutes)

    def _path(self, ticker: str, expiry: str) -> Path:
        return self.cache_dir / f"{ticker}_{expiry}.pkl"

    def get(self, ticker: str, expiry: str) -> Optional[Tuple[pd.DataFrame, float]]:
        p = self._path(ticker, expiry)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if datetime.now() - data["timestamp"] < self.ttl:
                return data["options_data"], data["underlying_price"]
        except Exception:
            p.unlink(missing_ok=True)
        return None

    def set(self, ticker: str, expiry: str, df: pd.DataFrame, price: float):
        p = self._path(ticker, expiry)
        try:
            with open(p, "wb") as f:
                pickle.dump(
                    {
                        "timestamp": datetime.now(),
                        "options_data": df,
                        "underlying_price": price,
                    },
                    f,
                )
        except Exception:
            pass

    def clear(self):
        for fp in self.cache_dir.glob("*.pkl"):
            fp.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Reader implementation
# -----------------------------------------------------------------------------


class Reader(AbstractReader):
    """Fetch call-option chains from *yfinance* and return a validated DataFrame."""

    def __init__(self, cache_enabled: bool = True, cache_ttl_minutes: int = 15):
        self._cache = (
            _YFinanceCache(ttl_minutes=cache_ttl_minutes) if cache_enabled else None
        )
        self._yf = None

    # ---------- helper -------------------------------------------------------
    def _yf_mod(self):
        if self._yf is None:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise ImportError("Install with: pip install oipd[yfinance]") from exc
            self._yf = yf
        return self._yf

    # ---------- public -------------------------------------------------------
    def load(
        self,
    ) -> pd.DataFrame:  # pragma: no cover – Unused in this context
        raise NotImplementedError("Use read(ticker_expiry, …) via DataSource wrapper")

    # --- interface required by AbstractReader -------------------------------
    def _ingest_data(self, ticker_expiry: str) -> pd.DataFrame:  # noqa: D401
        """Download options for *ticker:expiry* or serve from cache."""
        try:
            ticker, expiry = ticker_expiry.split(":", 1)
        except ValueError as exc:
            raise ValueError("Expect 'TICKER:YYYY-MM-DD'") from exc

        # cache first
        if self._cache:
            hit = self._cache.get(ticker, expiry)
            if hit is not None:
                df, price = hit
                df.attrs["underlying_price"] = price

                # Apply column mapping to cached data
                column_mapping = {
                    "lastPrice": "last_price",
                    "lastTradeDate": "last_trade_date",
                }
                for yf_name, std_name in column_mapping.items():
                    if yf_name in df.columns:
                        df = df.rename(columns={yf_name: std_name})

                return df

        yf = self._yf_mod()
        try:
            tk = yf.Ticker(ticker)
        except Exception as exc:
            raise YFinanceError(f"yfinance failed for ticker {ticker}: {exc}") from exc

        # current price -------------------------------------------------------
        price = tk.info.get("currentPrice") or tk.info.get("regularMarketPrice")
        if price is None:
            hist = tk.history(period="1d")
            if hist.empty:
                raise YFinanceError("Unable to fetch current price")
            price = float(hist["Close"].iloc[-1])

        # option chain --------------------------------------------------------
        try:
            chain = tk.option_chain(expiry)
            calls_df = chain.calls
            puts_df = chain.puts

            if calls_df.empty and puts_df.empty:
                raise YFinanceError("No options data returned")
            elif calls_df.empty:
                raise YFinanceError("No call data returned")
        except Exception as exc:
            raise YFinanceError(f"Failed to fetch chain: {exc}") from exc

        # Map yfinance column names to our standard names for both DataFrames
        column_mapping = {
            "lastPrice": "last_price",
            "lastTradeDate": "last_trade_date",
        }

        # Apply mapping for calls
        for yf_name, std_name in column_mapping.items():
            if yf_name in calls_df.columns:
                calls_df = calls_df.rename(columns={yf_name: std_name})

        # Apply mapping for puts if available
        if not puts_df.empty:
            for yf_name, std_name in column_mapping.items():
                if yf_name in puts_df.columns:
                    puts_df = puts_df.rename(columns={yf_name: std_name})

        # Combine calls and puts into single DataFrame with option_type column
        final_df = self._combine_options_data(calls_df, puts_df)

        # Set metadata on final DataFrame
        final_df.attrs["underlying_price"] = price
        dividend_yield = self._extract_dividend_yield(tk.info)
        final_df.attrs["dividend_yield"] = dividend_yield
        final_df.attrs["dividend_schedule"] = (
            None  # yfinance cannot infer forward schedule
        )

        if self._cache:
            self._cache.set(ticker, expiry, final_df.copy(), price)

        return final_df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply yfinance-specific transformations.

        Arguments:
            df: The validated DataFrame

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If insufficient data points

        Note:
            Common validation (NaN checks, negative prices, etc.) is handled
            by the base class _validate_data() method.
        """
        # Ensure we have enough data points for meaningful analysis
        if len(df) < 5:
            raise ValueError("Need at least 5 strikes for a meaningful smile")
        return df.reset_index(drop=True)

    # ---------- options data helpers ----------------------------------------
    def _combine_options_data(
        self, calls_df: pd.DataFrame, puts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine calls and puts into standardized format with option_type column.

        Converts yfinance's separate calls/puts DataFrames into the standardized format:
        columns: [strike, last_price, option_type, bid, ask, ...]
        where option_type is 'C' for calls, 'P' for puts

        Parameters
        ----------
        calls_df : pd.DataFrame
            Call options data from yfinance
        puts_df : pd.DataFrame
            Put options data from yfinance

        Returns
        -------
        pd.DataFrame
            Combined options data in standardized format
        """
        combined_data = []

        # Add calls with option_type = 'C'
        if not calls_df.empty:
            calls_copy = calls_df.copy()
            calls_copy["option_type"] = "C"
            combined_data.append(calls_copy)

        # Add puts with option_type = 'P'
        if not puts_df.empty:
            puts_copy = puts_df.copy()
            puts_copy["option_type"] = "P"
            combined_data.append(puts_copy)

        if not combined_data:
            raise YFinanceError("No options data available")

        # Combine and sort by strike
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df = combined_df.sort_values(["strike", "option_type"]).reset_index(
            drop=True
        )

        return combined_df

    # ---------- dividend helpers -------------------------------------------
    def _extract_dividend_yield(self, info) -> Optional[float]:
        dy = info.get("dividendYield")
        if dy is None:
            return None
        try:
            # yfinance returns dividend yield as percentage (e.g., 1.21 for 1.21%)
            # but the library expects decimal format (e.g., 0.0121 for 1.21%)
            return float(dy) / 100.0
        except Exception:
            return None

    # ---------- misc helpers -------------------------------------------------
    @classmethod
    def list_expiry_dates(cls, ticker: str) -> list[str]:
        mod = cls()
        yf = mod._yf_mod()
        try:
            return list(yf.Ticker(ticker).options)
        except Exception as exc:
            raise YFinanceError(f"Failed to list expiries for {ticker}: {exc}") from exc


__all__ = ["Reader", "YFinanceError"]
