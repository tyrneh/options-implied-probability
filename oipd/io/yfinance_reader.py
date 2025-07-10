import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from pathlib import Path

from pandas import DataFrame
from oipd.io.reader import AbstractReader


class YFinanceError(Exception):
    """Exception raised when yfinance operations fail"""

    pass


class YFinanceCache:
    """Simple file-based cache for yfinance data to avoid rate limiting"""

    def __init__(self, cache_dir: str = ".yfinance_cache", ttl_minutes: int = 15):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(minutes=ttl_minutes)

    def _get_cache_path(self, ticker: str, expiry: str) -> Path:
        """Get cache file path for ticker and expiry combination"""
        return self.cache_dir / f"{ticker}_{expiry}.pkl"

    def get(self, ticker: str, expiry: str) -> Optional[Tuple[DataFrame, float]]:
        """Get cached options data and current price if available and not expired"""
        cache_path = self._get_cache_path(ticker, expiry)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            # Check if cache is still valid
            if datetime.now() - cached_data["timestamp"] < self.ttl:
                return cached_data["options_data"], cached_data["current_price"]

        except (pickle.PickleError, KeyError, OSError):
            # If cache is corrupted, remove it
            cache_path.unlink(missing_ok=True)

        return None

    def set(
        self, ticker: str, expiry: str, options_data: DataFrame, current_price: float
    ):
        """Cache options data and current price"""
        cache_path = self._get_cache_path(ticker, expiry)

        cached_data = {
            "timestamp": datetime.now(),
            "options_data": options_data,
            "current_price": current_price,
            "ticker": ticker,
            "expiry": expiry,
        }

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)
        except (pickle.PickleError, OSError):
            # If we can't cache, just continue without caching
            pass

    def clear(self):
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)


class YFinanceReader(AbstractReader):
    """Reader implementation for fetching options data from yfinance."""

    def __init__(self, cache_enabled: bool = True, cache_ttl_minutes: int = 15):
        self.cache_enabled = cache_enabled
        self.cache = (
            YFinanceCache(ttl_minutes=cache_ttl_minutes) if cache_enabled else None
        )
        self._yf = None

    def _get_yfinance(self):
        """Lazy import and cache yfinance module"""
        if self._yf is None:
            try:
                import yfinance as yf

                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is required for ticker-based data fetching. "
                    "Install with: pip install yfinance"
                )
        return self._yf

    def _ingest_data(self, ticker_expiry: str) -> DataFrame:
        """Ingest options data from yfinance for the given ticker and expiry

        Arguments:
            ticker_expiry: String in format "TICKER:EXPIRY" (e.g., "AAPL:2025-01-17")

        Returns:
            DataFrame containing options data with current_price as metadata

        Raises:
            YFinanceError: If data fetching fails
            ValueError: If ticker_expiry format is invalid
        """
        try:
            ticker, expiry = ticker_expiry.split(":", 1)
        except ValueError:
            raise ValueError(
                "ticker_expiry must be in format 'TICKER:EXPIRY' (e.g., 'AAPL:2025-01-17')"
            )

        # Try to get from cache first
        if self.cache_enabled and self.cache:
            cached_result = self.cache.get(ticker, expiry)
            if cached_result:
                options_data, current_price = cached_result
                # Store current price as metadata in the DataFrame
                options_data.attrs["current_price"] = current_price
                return options_data

        # Fetch from yfinance
        yf = self._get_yfinance()

        try:
            yf_ticker = yf.Ticker(ticker)

            # Get current price
            try:
                # Try to get from info first
                info = yf_ticker.info
                current_price = info.get("currentPrice") or info.get(
                    "regularMarketPrice"
                )

                if current_price is None:
                    # Fallback to recent history
                    hist = yf_ticker.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist["Close"].iloc[-1])
                    else:
                        raise YFinanceError(
                            "Could not fetch current price from yfinance"
                        )

            except Exception as e:
                raise YFinanceError(f"Failed to fetch current price for {ticker}: {e}")

            # Get options data
            try:
                options_chain = yf_ticker.option_chain(expiry)
                calls_df = options_chain.calls

                if calls_df.empty:
                    raise YFinanceError(
                        f"No options data available for {ticker} expiry {expiry}"
                    )

            except Exception as e:
                raise YFinanceError(
                    f"Failed to fetch options data for {ticker} expiry {expiry}: {e}"
                )

            # Store current price as metadata
            calls_df.attrs["current_price"] = current_price

            # Cache the result
            if self.cache_enabled and self.cache:
                self.cache.set(ticker, expiry, calls_df.copy(), current_price)

            return calls_df

        except Exception as e:
            if isinstance(e, YFinanceError):
                raise
            raise YFinanceError(f"Unexpected error fetching data for {ticker}: {e}")

    def _transform_data(self, cleaned_data: DataFrame) -> DataFrame:
        """Apply validation and transformation to the yfinance DataFrame.

        Arguments:
            cleaned_data: The cleaned DataFrame

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If data contains invalid values
        """
        # Check for negative prices
        if (cleaned_data["last_price"] < 0).any():
            raise ValueError("Options data contains negative prices")

        # Check for zero or negative strikes
        if (cleaned_data["strike"] <= 0).any():
            raise ValueError("Options data contains non-positive strike prices")

        # Check for NaN values in critical columns
        critical_columns = ["strike", "last_price"]
        for col in critical_columns:
            if cleaned_data[col].isna().any():
                raise ValueError(f"Options data contains NaN values in column '{col}'")

        # Sort by strike price for consistency
        cleaned_data = cleaned_data.sort_values("strike")

        # Ensure we have enough data points
        if len(cleaned_data) < 5:
            raise ValueError(
                f"Insufficient options data: need at least 5 strike prices, got {len(cleaned_data)}"
            )

        return cleaned_data.reset_index(drop=True)

    @classmethod
    def list_expiry_dates(cls, ticker: str) -> list[str]:
        """List available expiry dates for a given ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL", "SPY")

        Returns
        -------
        list[str]
            List of available expiry dates in YYYY-MM-DD format
        """
        reader = cls()
        yf = reader._get_yfinance()

        try:
            yf_ticker = yf.Ticker(ticker)
            expiry_dates = yf_ticker.options
            return list(expiry_dates)
        except Exception as e:
            raise YFinanceError(
                f"Failed to fetch expiry dates for ticker '{ticker}': {e}"
            )
