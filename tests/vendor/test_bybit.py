"""Tests for Bybit vendor integration."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import date

from oipd.vendor.bybit.reader import Reader, BybitError


class TestBybitReader:
    """Test suite for Bybit options data reader."""
    
    def test_init(self):
        """Test reader initialization."""
        reader = Reader()
        assert reader._cache is not None
        assert reader._session is None
        
        # Test cache disabled
        reader_no_cache = Reader(cache_enabled=False)
        assert reader_no_cache._cache is None

    def test_parse_bybit_expiry(self):
        """Test parsing Bybit expiry format."""
        reader = Reader()
        
        # Test short format
        result = reader._parse_bybit_expiry("30DEC22")
        expected = date(2022, 12, 30)
        assert result == expected
        
        # Test long format  
        result = reader._parse_bybit_expiry("30DEC2022")
        expected = date(2022, 12, 30)
        assert result == expected
        
        # Test different months
        result = reader._parse_bybit_expiry("15JAN23")
        expected = date(2023, 1, 15)
        assert result == expected
        
        # Test invalid format
        with pytest.raises(ValueError):
            reader._parse_bybit_expiry("INVALID")

    @patch('pybit.unified_trading.HTTP')
    def test_ingest_data_no_options(self, mock_http):
        """Test handling when no options are available."""
        reader = Reader(cache_enabled=False)
        
        # Mock API responses
        mock_session = Mock()
        mock_http.return_value = mock_session
        
        # Mock spot ticker response
        mock_session.get_tickers.return_value = {
            "retCode": 0,
            "result": {"list": [{"lastPrice": "50000.0"}]}
        }
        
        # Mock empty instruments response
        mock_session.get_instruments_info.return_value = {
            "retCode": 0,
            "result": {"list": []}
        }
        
        with pytest.raises(BybitError, match="No options found"):
            reader._ingest_data("BTC:2025-12-26")

    @patch('pybit.unified_trading.HTTP')
    def test_ingest_data_success(self, mock_http):
        """Test successful data ingestion."""
        reader = Reader(cache_enabled=False)
        
        # Mock API responses
        mock_session = Mock()
        mock_http.return_value = mock_session
        
        # Mock spot ticker response
        mock_session.get_tickers.side_effect = [
            # Spot price response
            {
                "retCode": 0,
                "result": {"list": [{"lastPrice": "50000.0"}]}
            },
            # Option ticker responses
            {
                "retCode": 0,
                "result": {"list": [{
                    "lastPrice": "1000.0",
                    "bid1Price": "950.0",
                    "ask1Price": "1050.0",
                    "volume24h": "10.5",
                    "openInterest": "100.0"
                }]}
            }
        ]
        
        # Mock instruments response
        mock_session.get_instruments_info.return_value = {
            "retCode": 0,
            "result": {
                "list": [{
                    "symbol": "BTC-26DEC25-50000-C"
                }]
            }
        }
        
        result = reader._ingest_data("BTC:2025-12-26")
        
        assert isinstance(result, pd.DataFrame)
        assert result.attrs["underlying_price"] == 50000.0
        assert result.attrs["dividend_yield"] == 0.0
        assert result.attrs["dividend_schedule"] is None
        
        # Check DataFrame structure
        expected_columns = ["strike", "last_price", "option_type", "bid", "ask", "symbol", "volume", "open_interest"]
        for col in expected_columns:
            assert col in result.columns

    def test_clean_data_empty(self):
        """Test cleaning empty data."""
        reader = Reader()
        empty_df = pd.DataFrame()
        
        with pytest.raises(BybitError, match="No options data available"):
            reader._clean_data(empty_df)

    def test_clean_data_no_prices(self):
        """Test cleaning data with no valid prices."""
        reader = Reader()
        df = pd.DataFrame({
            "strike": [100, 110],
            "last_price": [0, 0],
            "bid": [0, float('nan')],
            "ask": [0, float('nan')],
            "option_type": ["C", "C"]
        })
        
        with pytest.raises(BybitError, match="No options with valid price data"):
            reader._clean_data(df)

    def test_validate_data_missing_columns(self):
        """Test validation with missing required columns."""
        reader = Reader()
        df = pd.DataFrame({
            "strike": [100],
            "last_price": [10]
            # Missing option_type
        })
        
        with pytest.raises(BybitError, match="Missing required columns"):
            reader._validate_data(df)

    def test_validate_data_invalid_option_types(self):
        """Test validation with invalid option types."""
        reader = Reader()
        df = pd.DataFrame({
            "strike": [100],
            "last_price": [10],
            "option_type": ["X"]  # Invalid
        })
        
        with pytest.raises(BybitError, match="Invalid option types found"):
            reader._validate_data(df)

    def test_validate_data_negative_strikes(self):
        """Test validation with negative strikes."""
        reader = Reader()
        df = pd.DataFrame({
            "strike": [-100],
            "last_price": [10],
            "option_type": ["C"]
        })
        
        with pytest.raises(BybitError, match="Found negative or zero strike prices"):
            reader._validate_data(df)

    def test_transform_data_insufficient_strikes(self):
        """Test transformation with insufficient data points."""
        reader = Reader()
        df = pd.DataFrame({
            "strike": [100, 110],
            "last_price": [10, 15],
            "option_type": ["C", "C"]
        })
        
        with pytest.raises(BybitError, match="Need at least 5 strikes"):
            reader._transform_data(df)

    def test_transform_data_success(self):
        """Test successful data transformation."""
        reader = Reader()
        df = pd.DataFrame({
            "strike": [90, 100, 110, 120, 130],
            "last_price": [15, 10, 6, 3, 1],
            "option_type": ["C", "C", "C", "C", "C"]
        })
        
        result = reader._transform_data(df)
        
        assert len(result) == 5
        assert result.index.tolist() == [0, 1, 2, 3, 4]  # Reset index
        
    @patch('pybit.unified_trading.HTTP')
    def test_list_expiry_dates(self, mock_http):
        """Test listing available expiry dates."""
        # Mock API response
        mock_session = Mock()
        mock_http.return_value = mock_session
        
        mock_session.get_instruments_info.return_value = {
            "retCode": 0,
            "result": {
                "list": [
                    {"symbol": "BTC-30DEC22-50000-C"},
                    {"symbol": "BTC-30DEC22-55000-P"},
                    {"symbol": "BTC-31JAN23-50000-C"},
                    {"symbol": "ETH-30DEC22-3000-C"},  # Different base coin, should be filtered
                    {"symbol": "INVALID-FORMAT"},       # Invalid format, should be skipped
                ]
            }
        }
        
        result = Reader.list_expiry_dates("BTC")
        
        expected = ["2022-12-30", "2023-01-31"]
        assert result == expected
        
        # Verify the API was called correctly
        mock_session.get_instruments_info.assert_called_once_with(
            category="option",
            baseCoin="BTC",
            limit=1000
        )

    @patch('pybit.unified_trading.HTTP')
    def test_list_expiry_dates_api_error(self, mock_http):
        """Test handling API errors when listing expiry dates."""
        mock_session = Mock()
        mock_http.return_value = mock_session
        
        mock_session.get_instruments_info.return_value = {
            "retCode": 10001,
            "retMsg": "API error"
        }
        
        with pytest.raises(BybitError, match="Failed to get instruments"):
            Reader.list_expiry_dates("BTC")


if __name__ == "__main__":
    pytest.main([__file__])
