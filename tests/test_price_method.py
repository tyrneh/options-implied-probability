import numpy as np
import pandas as pd
import pytest
import warnings
from datetime import date

from oipd.estimator import RND, ModelParams
from oipd.market_inputs import MarketInputs
from oipd.core.pdf import _calculate_price, CalculationError
from oipd.io.csv_reader import CSVReader
from oipd.io.dataframe_reader import DataFrameReader


class TestPriceMethodCalculation:
    """Test the price calculation logic for last vs mid price methods."""

    def test_calculate_price_last_method(self):
        """Test that 'last' method uses last_price column."""
        # Create test data
        options_data = pd.DataFrame({
            "strike": [90, 95, 100, 105, 110],
            "last_price": [12.0, 8.0, 5.0, 3.0, 1.5],
            "bid": [11.5, 7.5, 4.5, 2.5, 1.0],
            "ask": [12.5, 8.5, 5.5, 3.5, 2.0],
        })

        result = _calculate_price(options_data, "last")

        # Should use last_price values
        expected_prices = [12.0, 8.0, 5.0, 3.0, 1.5]
        assert list(result["price"]) == expected_prices

    def test_calculate_price_mid_method(self):
        """Test that 'mid' method averages bid and ask."""
        options_data = pd.DataFrame({
            "strike": [90, 95, 100, 105, 110],
            "last_price": [12.0, 8.0, 5.0, 3.0, 1.5],
            "bid": [11.0, 7.0, 4.0, 2.0, 1.0],
            "ask": [13.0, 9.0, 6.0, 4.0, 2.0],
        })

        result = _calculate_price(options_data, "mid")

        # Should use (bid + ask) / 2
        expected_prices = [12.0, 8.0, 5.0, 3.0, 1.5]
        assert list(result["price"]) == expected_prices

    def test_calculate_price_filters_negative(self):
        """Test that negative prices are filtered out."""
        options_data = pd.DataFrame({
            "strike": [90, 95, 100, 105, 110],
            "last_price": [12.0, -1.0, 5.0, 3.0, 1.5],  # Negative price
            "bid": [11.0, 7.0, 4.0, 2.0, 1.0],
            "ask": [13.0, 9.0, 6.0, 4.0, 2.0],
        })

        result = _calculate_price(options_data, "last")

        # Should exclude the row with negative price
        assert len(result) == 4
        assert 95 not in result["strike"].values


class TestModelParamsIntegration:
    """Test the integration of price_method with ModelParams."""

    def test_model_params_default_last(self):
        """Test that ModelParams defaults to 'last' price method."""
        model = ModelParams()
        assert model.price_method == "last"

    def test_model_params_explicit_mid(self):
        """Test that ModelParams accepts 'mid' price method."""
        model = ModelParams(price_method="mid")
        assert model.price_method == "mid"


class TestEndToEndPriceMethod:
    """Test price_method functionality end-to-end."""

    @pytest.fixture
    def sample_options_data(self):
        """Create sample options data for testing."""
        return pd.DataFrame({
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],  # Different from mid-price
            "bid": [11.0, 7.0, 4.0, 2.0, 1.0],
            "ask": [13.0, 9.0, 6.0, 4.0, 2.0],  # Mid would be [12.0, 8.0, 5.0, 3.0, 1.5]
        })

    @pytest.fixture
    def market_inputs(self):
        """Create market inputs for testing."""
        return MarketInputs(
            valuation_date=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
            risk_free_rate_mode="continuous",
        )

    def test_rnd_from_dataframe_last_price(self, sample_options_data, market_inputs):
        """Test RND calculation using last price method."""
        model = ModelParams(price_method="last", pricing_engine="bs")
        
        result = RND.from_dataframe(sample_options_data, market_inputs, model=model)
        
        # Should complete without error
        assert result.prices is not None
        assert result.pdf is not None
        assert result.cdf is not None

    def test_rnd_from_dataframe_mid_price(self, sample_options_data, market_inputs):
        """Test RND calculation using mid price method."""
        model = ModelParams(price_method="mid", pricing_engine="bs")
        
        result = RND.from_dataframe(sample_options_data, market_inputs, model=model)
        
        # Should complete without error
        assert result.prices is not None
        assert result.pdf is not None
        assert result.cdf is not None

    def test_price_methods_give_different_results(self, sample_options_data, market_inputs):
        """Test that last and mid price methods produce different results."""
        model_last = ModelParams(price_method="last", pricing_engine="bs")
        model_mid = ModelParams(price_method="mid", pricing_engine="bs")
        
        result_last = RND.from_dataframe(sample_options_data, market_inputs, model=model_last)
        result_mid = RND.from_dataframe(sample_options_data, market_inputs, model=model_mid)
        
        # Results should be different (unless bid-ask spread is zero, which it isn't in our test data)
        assert not np.array_equal(result_last.pdf, result_mid.pdf)

    def test_backward_compatibility_default_last(self, sample_options_data, market_inputs):
        """Test that default behavior (no explicit price_method) uses last price."""
        # No explicit model - should default to last price
        result_default = RND.from_dataframe(
            sample_options_data,
            market_inputs,
            model=ModelParams(pricing_engine="bs"),
        )

        # Explicit last price
        model_last = ModelParams(price_method="last", pricing_engine="bs")
        result_explicit_last = RND.from_dataframe(
            sample_options_data, market_inputs, model=model_last
        )
        
        # Results should be identical
        assert np.array_equal(result_default.pdf, result_explicit_last.pdf)
        assert np.array_equal(result_default.cdf, result_explicit_last.cdf)


class TestRobustDataHandling:
    """Test handling of missing data and data quality issues."""

    def test_dataframe_with_missing_bid_ask_columns(self):
        """Test DataFrame with missing bid/ask columns."""
        df = pd.DataFrame({
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            # No bid/ask columns
        })
        
        reader = DataFrameReader()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = reader.read(df)
            
            # Should warn about missing optional columns
            assert len(w) == 1
            assert "Optional columns not present" in str(w[0].message)
            assert "bid" in str(w[0].message) and "ask" in str(w[0].message)
        
        # Should have NaN columns added
        assert "bid" in result.columns
        assert "ask" in result.columns
        assert result["bid"].isna().all()
        assert result["ask"].isna().all()

    def test_csv_with_dashes_in_bid_ask(self):
        """Test CSV data with dashes representing missing bid/ask."""
        df = pd.DataFrame({
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            "bid": ["–", "7.0", "4.0", "–", "1.0"],  # Some dashes
            "ask": ["13.0", "–", "6.0", "4.0", "2.0"],  # Some dashes
        })
        
        reader = DataFrameReader()
        result = reader.read(df)
        
        # Dashes should be converted to NaN
        assert pd.isna(result.iloc[0]["bid"])  # First dash
        assert pd.isna(result.iloc[1]["ask"])   # Second dash
        assert pd.isna(result.iloc[3]["bid"])   # Fourth dash
        
        # Valid numbers should be preserved
        assert result.iloc[1]["bid"] == 7.0
        assert result.iloc[0]["ask"] == 13.0

    def test_comma_separated_numbers(self):
        """Test handling of comma-separated numbers."""
        df = pd.DataFrame({
            "strike": ["1,000", "1,050", "1,100", "1,150", "1,200"],
            "last_price": ["12.50", "8.25", "5.10", "3.00", "1.50"],
            "bid": ["12.00", "8.00", "5.00", "2.75", "1.25"],
            "ask": ["13.00", "8.50", "5.20", "3.25", "1.75"],
        })
        
        reader = DataFrameReader()
        result = reader.read(df)
        
        # Comma should be removed and converted to float
        assert result.iloc[0]["strike"] == 1000.0
        assert result.iloc[1]["strike"] == 1050.0
        assert result.iloc[2]["strike"] == 1100.0
        assert result.iloc[3]["strike"] == 1150.0
        assert result.iloc[4]["strike"] == 1200.0

    def test_mid_price_with_missing_data_fallback(self):
        """Test mid-price calculation with missing bid/ask data."""
        df = pd.DataFrame({
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            "bid": [12.0, np.nan, 4.0, np.nan, 1.0],  # Some missing
            "ask": [13.0, 9.0, np.nan, 4.0, 2.0],      # Some missing
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _calculate_price(df, "mid")
            
            # Should warn about fallback
            assert len(w) == 1
            assert "Using last_price for" in str(w[0].message)
            assert "missing bid/ask" in str(w[0].message)
        
        # Check calculations
        assert result.iloc[0]["price"] == 12.5  # (12.0 + 13.0) / 2
        assert result.iloc[1]["price"] == 8.2   # Fallback to last_price (bid missing)
        assert result.iloc[2]["price"] == 5.1   # Fallback to last_price (ask missing)
        assert result.iloc[3]["price"] == 3.1   # Fallback to last_price (bid missing)
        assert result.iloc[4]["price"] == 1.5   # (1.0 + 2.0) / 2

    def test_mid_price_with_no_bid_ask_data(self):
        """Test mid-price when no bid/ask data is available."""
        df = pd.DataFrame({
            "strike": [90.0, 95.0, 100.0],
            "last_price": [12.5, 8.2, 5.1],
            "bid": [np.nan, np.nan, np.nan],
            "ask": [np.nan, np.nan, np.nan],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _calculate_price(df, "mid")
            
            # Should warn about fallback to last method
            assert len(w) == 1
            assert "Requested price_method='mid' but bid/ask data not available" in str(w[0].message)
            assert "Falling back to price_method='last'" in str(w[0].message)
        
        # Should use last_price for all rows
        pd.testing.assert_series_equal(result["price"], df["last_price"], check_names=False)

    def test_rnd_from_dataframe_missing_bid_ask(self):
        """Test RND calculation with missing bid/ask columns."""
        df = pd.DataFrame({
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            # No bid/ask columns
        })
        
        market = MarketInputs(
            valuation_date=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
            risk_free_rate_mode="continuous",
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = RND.from_dataframe(
                df, market, model=ModelParams(pricing_engine="bs")
            )
            
            # Should warn about missing optional columns
            assert any("Optional columns not present" in str(warning.message) for warning in w)
        
        # Should complete successfully
        assert result.prices is not None
        assert result.pdf is not None

    def test_rnd_with_mid_price_and_missing_columns(self):
        """Test RND with mid-price method but missing bid/ask columns."""
        df = pd.DataFrame({
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "last_price": [12.5, 8.2, 5.1, 3.1, 1.6],
            # No bid/ask columns
        })
        
        market = MarketInputs(
            valuation_date=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            underlying_price=100.0,
            risk_free_rate=0.05,
            risk_free_rate_mode="continuous",
        )
        
        model = ModelParams(price_method="mid", pricing_engine="bs")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(CalculationError):
                RND.from_dataframe(df, market, model=model)

            warning_messages = [str(warning.message) for warning in w]
            assert any("Optional columns not present" in msg for msg in warning_messages)
