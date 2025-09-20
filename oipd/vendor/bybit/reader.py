from __future__ import annotations

"""Bybit data connector for *oipd*.

Implements :class:`Reader`, compatible with the generic *vendor* registry.
"""

import pickle
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oipd.io.reader import AbstractReader

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class BybitError(Exception):
    """Exception raised when Bybit operations fail"""

    pass


# -----------------------------------------------------------------------------
# Simple file-based cache
# -----------------------------------------------------------------------------


class _BybitCache:
    def __init__(self, cache_dir: str = ".bybit_cache", ttl_minutes: int = 15):
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
    """Fetch crypto options data from *Bybit* and return a validated DataFrame."""

    def __init__(self, cache_enabled: bool = True, cache_ttl_minutes: int = 15):
        self._cache = (
            _BybitCache(ttl_minutes=cache_ttl_minutes) if cache_enabled else None
        )
        self._session = None

    # ---------- helper -------------------------------------------------------
    def _get_session(self):
        """Get HTTP session for API calls with retry logic"""
        if self._session is None:
            self._session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=5,  # Total number of retries
                status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
                backoff_factor=1,  # Wait time between retries
                raise_on_status=False
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
            
            # Set headers
            self._session.headers.update({
                'Content-Type': 'application/json',
                'User-Agent': 'OIPD/1.0'
            })
            
        return self._session
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request to Bybit demo API with error handling and retries"""
        base_url = "https://api.bybit.com"  # Using live API (not testnet)
        url = f"{base_url}{endpoint}"
        
        session = self._get_session()
        
        for attempt in range(3):  # Additional retry logic on top of requests retry
            try:
                response = session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("retCode") == 0:
                        return data
                    else:
                        raise BybitError(f"API error: {data.get('retMsg', 'Unknown error')}")
                else:
                    raise BybitError(f"HTTP error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < 2:  # Only retry on connection errors
                    print(f"Connection attempt {attempt + 1} failed, retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                raise BybitError(f"Connection failed after {attempt + 1} attempts: {e}")
            except requests.exceptions.Timeout as e:
                if attempt < 2:
                    print(f"Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                raise BybitError(f"Request timeout after {attempt + 1} attempts: {e}")
            except Exception as e:
                raise BybitError(f"Unexpected error: {e}")
                
        raise BybitError("Max retries exceeded")
    
    def get_tickers(self, category: str, symbol: str = None, baseCoin: str = None) -> Dict:
        """Get ticker information"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        if baseCoin:
            params["baseCoin"] = baseCoin
            
        return self._make_api_request("/v5/market/tickers", params)
    
    def get_instruments_info(self, category: str, baseCoin: str = None, limit: int = 1000) -> Dict:
        """Get instruments information"""
        params = {"category": category, "limit": limit}
        if baseCoin:
            params["baseCoin"] = baseCoin
            
        return self._make_api_request("/v5/market/instruments-info", params)
    
    def get_kline(self, category: str, symbol: str, interval: str, start: int = None, end: int = None, limit: int = 200) -> Dict:
        """Get kline/candlestick data"""
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return self._make_api_request("/v5/market/kline", params)

    def _ingest_data(self, ticker_expiry: str) -> pd.DataFrame:
        """Download options for *ticker:expiry* or serve from cache."""
        try:
            ticker, expiry = ticker_expiry.split(":", 1)
        except ValueError as exc:
            raise ValueError("Expect 'TICKER:YYYY-MM-DD'") from exc

        # Check cache first
        if self._cache:
            hit = self._cache.get(ticker, expiry)
            if hit is not None:
                df, price = hit
                df.attrs["underlying_price"] = price
                return df

        try:
            # Convert ticker format (e.g., BTC -> BTC for Bybit)
            base_coin = ticker.upper()
            
            # Get underlying price from spot ticker
            spot_ticker = f"{base_coin}USDT"
            try:
                spot_data = self.get_tickers(category="spot", symbol=spot_ticker)
                underlying_price = float(spot_data["result"]["list"][0]["lastPrice"])
            except (KeyError, IndexError, ValueError) as e:
                raise BybitError(f"Could not parse underlying price for {spot_ticker}: {e}")

            # Get options instruments for the expiry date
            # Bybit options use format like "BTC-30DEC22-18000-C"
            instruments_data = self.get_instruments_info(
                category="option",
                baseCoin=base_coin,
                limit=1000
            )
            
            if instruments_data["retCode"] != 0:
                raise BybitError(f"Failed to get instruments: {instruments_data['retMsg']}")

            # Filter instruments by expiry date
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            options_symbols = []
            
            for instrument in instruments_data["result"]["list"]:
                symbol = instrument["symbol"]
                try:
                    # Parse Bybit option symbol format: BTC-30DEC22-18000-C
                    parts = symbol.split("-")
                    if len(parts) >= 4 and parts[0] == base_coin:
                        # Parse expiry from symbol (e.g., 30DEC22)
                        expiry_str = parts[1]
                        instrument_expiry = self._parse_bybit_expiry(expiry_str)
                        if instrument_expiry == expiry_date:
                            options_symbols.append(symbol)
                except Exception:
                    # Skip malformed symbols
                    continue

            if not options_symbols:
                raise BybitError(f"No options found for {ticker} expiring on {expiry}")

            # Get ticker data for all relevant options
            options_data = []
            
            # Process in batches to avoid rate limits
            batch_size = 50
            for i in range(0, len(options_symbols), batch_size):
                batch_symbols = options_symbols[i:i + batch_size]
                
                for symbol in batch_symbols:
                    try:
                        ticker_data = self.get_tickers(category="option", symbol=symbol)
                        
                        if ticker_data["retCode"] == 0 and ticker_data["result"]["list"]:
                            option_info = ticker_data["result"]["list"][0]
                            
                            # Parse symbol to get strike and option type
                            try:
                                parts = symbol.split("-")
                                strike = float(parts[2])
                                option_type = "C" if parts[3] == "C" else "P"
                            except (IndexError, ValueError):
                                continue  # Skip malformed symbols
                            
                            # Extract relevant data
                            last_price = float(option_info.get("lastPrice", 0))
                            bid_price = float(option_info.get("bid1Price", 0))
                            ask_price = float(option_info.get("ask1Price", 0))
                            
                            # Skip if no meaningful price data
                            if last_price <= 0 and bid_price <= 0 and ask_price <= 0:
                                continue
                                
                            options_data.append({
                                "strike": strike,
                                "last_price": last_price,
                                "option_type": option_type,
                                "bid": bid_price if bid_price > 0 else np.nan,
                                "ask": ask_price if ask_price > 0 else np.nan,
                                "symbol": symbol,
                                "volume": float(option_info.get("volume24h", 0)),
                                "open_interest": float(option_info.get("openInterest", 0)),
                            })
                    except Exception as e:
                        # Log error but continue with other symbols
                        print(f"Warning: Failed to get data for {symbol}: {e}")
                        continue

            if not options_data:
                raise BybitError(f"No valid option data found for {ticker} on {expiry}")

            # Create DataFrame
            df = pd.DataFrame(options_data)
            
            # Set metadata
            df.attrs["underlying_price"] = underlying_price
            df.attrs["dividend_yield"] = 0.0  # Crypto doesn't have dividends
            df.attrs["dividend_schedule"] = None

            # Cache the result
            if self._cache:
                self._cache.set(ticker, expiry, df.copy(), underlying_price)

            return df

        except Exception as exc:
            if isinstance(exc, BybitError):
                raise
            raise BybitError(f"Failed to fetch data from Bybit: {exc}") from exc

    def _parse_bybit_expiry(self, expiry_str: str) -> date:
        """Parse Bybit expiry format like '30DEC22' to date object"""
        try:
            # Handle formats like "30DEC22" or "30DEC2022"
            if len(expiry_str) == 7:  # "30DEC22"
                day = int(expiry_str[:2])
                month_str = expiry_str[2:5]
                year = int("20" + expiry_str[5:])
            elif len(expiry_str) == 9:  # "30DEC2022"
                day = int(expiry_str[:2])
                month_str = expiry_str[2:5]
                year = int(expiry_str[5:])
            else:
                raise ValueError(f"Unexpected expiry format: {expiry_str}")
            
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
            }
            
            month = month_map[month_str]
            return date(year, month, day)
            
        except (ValueError, KeyError) as e:
            raise ValueError(f"Could not parse expiry '{expiry_str}': {e}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter Bybit options data"""
        if df.empty:
            raise BybitError("No options data available")

        # Remove rows with no price data
        df = df[
            (df["last_price"] > 0) | 
            (df["bid"].notna() & (df["bid"] > 0)) | 
            (df["ask"].notna() & (df["ask"] > 0))
        ].copy()

        if df.empty:
            raise BybitError("No options with valid price data")

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the cleaned data"""
        # Check required columns exist
        required_cols = ["strike", "last_price", "option_type"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise BybitError(f"Missing required columns: {missing_cols}")

        # Validate option types
        valid_types = {"C", "P"}
        invalid_types = set(df["option_type"].unique()) - valid_types
        if invalid_types:
            raise BybitError(f"Invalid option types found: {invalid_types}")

        # Check for negative strikes or prices
        if (df["strike"] <= 0).any():
            raise BybitError("Found negative or zero strike prices")

        if (df["last_price"] < 0).any():
            raise BybitError("Found negative option prices")

        return df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Bybit-specific transformations"""
        # Ensure we have enough data points for meaningful analysis
        if len(df) < 5:
            raise BybitError("Need at least 5 strikes for a meaningful smile")
        
        # Sort by strike and option type for consistent processing
        df = df.sort_values(["strike", "option_type"]).reset_index(drop=True)
        
        return df

    def load(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError("Use read(ticker_expiry, …) via DataSource wrapper")

    @classmethod
    def list_expiry_dates(cls, ticker: str) -> List[str]:
        """
        List available expiry dates for a given crypto ticker.

        Parameters
        ----------
        ticker : str
            Crypto ticker symbol (e.g., "BTC", "ETH")

        Returns
        -------
        List[str]
            List of available expiry dates in YYYY-MM-DD format
        """
        reader = cls()
        
        try:
            base_coin = ticker.upper()
            
            # Get all option instruments for this base coin
            instruments_data = reader.get_instruments_info(
                category="option",
                baseCoin=base_coin,
                limit=1000
            )
            
            if instruments_data["retCode"] != 0:
                raise BybitError(f"Failed to get instruments: {instruments_data['retMsg']}")

            expiry_dates = set()
            
            for instrument in instruments_data["result"]["list"]:
                symbol = instrument["symbol"]
                try:
                    # Parse Bybit option symbol format: BTC-30DEC22-18000-C
                    parts = symbol.split("-")
                    if len(parts) >= 4 and parts[0] == base_coin:
                        expiry_str = parts[1]
                        expiry_date = reader._parse_bybit_expiry(expiry_str)
                        expiry_dates.add(expiry_date.strftime("%Y-%m-%d"))
                except Exception:
                    # Skip malformed symbols
                    continue

            return sorted(list(expiry_dates))

        except Exception as exc:
            if isinstance(exc, BybitError):
                raise
            raise BybitError(f"Failed to list expiry dates for {ticker}: {exc}") from exc


__all__ = ["Reader", "BybitError"]
