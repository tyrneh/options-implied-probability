"""
Test Skeleton for Sources Data Access
======================================
Tests for the unified data loading interface (sources module).

Based on first-principles analysis of oipd.data_access.sources
"""

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock


# =============================================================================
# sources.from_csv() Tests
# =============================================================================

class TestFromCsv:
    """Tests for sources.from_csv() function."""

    def test_from_csv_returns_dataframe(self, tmp_path):
        """from_csv() returns a DataFrame."""
        from oipd.data_access import sources
        
        # Create temp CSV
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({
            "strike": [100.0, 110.0, 120.0, 130.0, 140.0],
            "last_price": [10.0, 8.0, 6.0, 4.0, 2.0],
            "option_type": ["C"] * 5,
        }).to_csv(csv_path, index=False)
        
        df = sources.from_csv(str(csv_path))
        assert isinstance(df, pd.DataFrame)

    def test_from_csv_with_column_mapping(self, tmp_path):
        """from_csv() applies column mapping."""
        from oipd.data_access import sources
        
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({
            "K": [100.0, 110.0, 120.0, 130.0, 140.0],
            "Price": [10.0, 8.0, 6.0, 4.0, 2.0],
            "Type": ["C"] * 5,
        }).to_csv(csv_path, index=False)
        
        mapping = {"K": "strike", "Price": "last_price", "Type": "option_type"}
        df = sources.from_csv(str(csv_path), column_mapping=mapping)
        
        assert "strike" in df.columns
        assert "last_price" in df.columns


# =============================================================================
# sources.from_dataframe() Tests
# =============================================================================

class TestFromDataframe:
    """Tests for sources.from_dataframe() function."""

    def test_from_dataframe_returns_dataframe(self):
        """from_dataframe() returns a DataFrame."""
        from oipd.data_access import sources
        
        raw = pd.DataFrame({
            "strike": [100.0, 110.0, 120.0, 130.0, 140.0],
            "last_price": [10.0, 8.0, 6.0, 4.0, 2.0],
            "option_type": ["C"] * 5,
        })
        
        df = sources.from_dataframe(raw)
        assert isinstance(df, pd.DataFrame)

    def test_from_dataframe_normalizes_option_type(self):
        """from_dataframe() normalizes option_type to C/P."""
        from oipd.data_access import sources
        
        raw = pd.DataFrame({
            "strike": [100.0, 110.0, 120.0, 130.0, 140.0],
            "last_price": [10.0, 8.0, 6.0, 4.0, 2.0],
            "option_type": ["call"] * 5,
        })
        
        df = sources.from_dataframe(raw)
        assert df["option_type"].iloc[0] in ["C", "P", "call", "put"]


# =============================================================================
# sources.fetch_chain() Tests
# =============================================================================

class TestFetchChain:
    """Tests for sources.fetch_chain() function."""

    @patch("oipd.data_access.sources.get_adapter")
    def test_fetch_chain_returns_tuple(self, mock_get_adapter):
        """fetch_chain() returns (DataFrame, VendorSnapshot)."""
        from oipd.data_access import sources
        from oipd.market_inputs import VendorSnapshot
        
        # Setup mock
        mock_adapter = MagicMock()
        mock_adapter.fetch_chain.return_value = (
            pd.DataFrame({"strike": [100.0]}),
            VendorSnapshot(
                vendor="yfinance",
                asof=pd.Timestamp.now(),
                underlying_price=100.0,
            )
        )
        mock_get_adapter.return_value = mock_adapter
        
        chain, snapshot = sources.fetch_chain("AAPL", expiries="2025-03-21")
        
        assert isinstance(chain, pd.DataFrame)
        assert isinstance(snapshot, VendorSnapshot)

    def test_fetch_chain_requires_expiries_or_horizon(self):
        """fetch_chain() raises if neither expiries nor horizon provided."""
        from oipd.data_access import sources
        
        with pytest.raises(ValueError, match="Must specify either"):
            sources.fetch_chain("AAPL")

    def test_fetch_chain_rejects_both_expiries_and_horizon(self):
        """fetch_chain() raises if both expiries and horizon provided."""
        from oipd.data_access import sources
        
        with pytest.raises(ValueError, match="Ambiguous request"):
            sources.fetch_chain("AAPL", expiries="2025-03-21", horizon="3m")


# =============================================================================
# sources.list_expiry_dates() Tests
# =============================================================================

class TestListExpiryDates:
    """Tests for sources.list_expiry_dates() function."""

    @patch("oipd.data_access.sources.get_reader")
    def test_list_expiry_dates_returns_list(self, mock_get_reader):
        """list_expiry_dates() returns a list of date strings."""
        from oipd.data_access import sources
        
        mock_reader = MagicMock()
        mock_reader.list_expiry_dates.return_value = ["2025-01-17", "2025-02-21"]
        mock_get_reader.return_value = mock_reader
        
        expiries = sources.list_expiry_dates("AAPL")
        
        assert isinstance(expiries, list)
        assert len(expiries) == 2
