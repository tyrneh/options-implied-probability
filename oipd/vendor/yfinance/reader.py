from __future__ import annotations

"""YFinance data connector for *oipd*.

Implements :class:`Reader`, compatible with the generic *vendor* registry.
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd

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
                return data["options_data"], data["current_price"]
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
                        "current_price": price,
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
    def load(self) -> pd.DataFrame:  # pragma: no cover – Unused in this context
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
                df.attrs["current_price"] = price
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
            if calls_df.empty:
                raise YFinanceError("No call data returned")
        except Exception as exc:
            raise YFinanceError(f"Failed to fetch chain: {exc}") from exc

        calls_df.attrs["current_price"] = price

        if self._cache:
            self._cache.set(ticker, expiry, calls_df.copy(), price)

        return calls_df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if (df["last_price"] < 0).any():
            raise ValueError("Negative option prices detected")
        if (df["strike"] <= 0).any():
            raise ValueError("Non-positive strikes detected")
        if df[["strike", "last_price"]].isna().any().any():
            raise ValueError("NaN values in critical columns")
        if len(df) < 5:
            raise ValueError("Need at least 5 strikes for a meaningful smile")
        return df.sort_values("strike").reset_index(drop=True)

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
