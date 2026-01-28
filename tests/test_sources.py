import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from oipd.data_access import sources


def test_from_csv():
    # 1. Path to test data
    data_path = os.path.join(os.path.dirname(__file__), "../data/AAPL_data.csv")

    # 2. Load
    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
        "expiration": "expiration",
    }

    df = sources.from_csv(data_path, column_mapping=column_mapping)

    # 3. Verify
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "strike" in df.columns
    assert "last_price" in df.columns
    # Check normalization (option_type should be lowercase 'call'/'put')
    assert df["option_type"].isin(["C", "P"]).all()


def test_from_dataframe():
    # 1. Create dummy dataframe
    raw_data = {
        "K": [100, 110, 120, 130, 140],
        "Price": [10, 5, 2, 1, 0.5],
        "Type": ["C", "P", "C", "P", "C"],
        "Expiry": ["2025-01-01"] * 5,
    }
    df_raw = pd.DataFrame(raw_data)

    # 2. Load
    mapping = {
        "K": "strike",
        "Price": "last_price",
        "Type": "option_type",
        "Expiry": "expiration",
    }

    df = sources.from_dataframe(df_raw, column_mapping=mapping)

    # 3. Verify
    assert len(df) == 5
    assert "strike" in df.columns
    assert df.iloc[0]["strike"] == 100
    # Check normalization of option types (C -> C)
    assert df.iloc[0]["option_type"] == "C"


@patch("oipd.data_access.sources.get_reader")
def test_from_ticker(mock_get_reader):
    # 1. Setup Mock
    mock_reader_instance = MagicMock()
    mock_reader_instance.read.return_value = pd.DataFrame(
        {"strike": [100], "last_price": [10]}
    )

    mock_reader_cls = MagicMock(return_value=mock_reader_instance)
    mock_get_reader.return_value = mock_reader_cls

    # 2. Call
    df, snapshot = sources.from_ticker("AAPL", expiry="2025-01-01")

    # 3. Verify
    mock_get_reader.assert_called_with("yfinance")
    mock_reader_instance.read.assert_called()
    call_args = mock_reader_instance.read.call_args
    assert call_args[0][0] == "AAPL:2025-01-01"
    assert not df.empty
    assert snapshot.vendor == "yfinance"


@patch("oipd.data_access.sources.get_reader")
def test_list_expiry_dates(mock_get_reader):
    class DummyReader:
        @classmethod
        def list_expiry_dates(cls, ticker):
            return ["2025-01-01", "2025-02-01"]

    mock_get_reader.return_value = DummyReader

    expiries = sources.list_expiry_dates("AAPL")

    mock_get_reader.assert_called_with("yfinance")
    assert expiries == ["2025-01-01", "2025-02-01"]


if __name__ == "__main__":
    test_from_csv()
    test_from_dataframe()
    test_from_ticker()
    test_list_expiry_dates()
    print("\nTest Passed!")
