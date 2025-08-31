import numpy as np
import pandas as pd
import pytest
from datetime import date

from oipd.estimator import RND, ModelParams
from oipd.market_inputs import MarketInputs
from oipd.core.pdf import _calculate_price


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
            "last_price": [12.0, 8.0, 5.0, 3.0, 1.5],
            "bid": [11.0, 7.0, 4.0, 2.0, 1.0],
            "ask": [13.0, 9.0, 6.0, 4.0, 2.0],
        })

    @pytest.fixture
    def market_inputs(self):
        """Create market inputs for testing."""
        return MarketInputs(
            valuation_date=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            spot_price=100.0,
            risk_free_rate=0.05,
        )

    def test_rnd_from_dataframe_last_price(self, sample_options_data, market_inputs):
        """Test RND calculation using last price method."""
        model = ModelParams(price_method="last")
        
        result = RND.from_dataframe(sample_options_data, market_inputs, model=model)
        
        # Should complete without error
        assert result.prices is not None
        assert result.pdf is not None
        assert result.cdf is not None

    def test_rnd_from_dataframe_mid_price(self, sample_options_data, market_inputs):
        """Test RND calculation using mid price method."""
        model = ModelParams(price_method="mid")
        
        result = RND.from_dataframe(sample_options_data, market_inputs, model=model)
        
        # Should complete without error
        assert result.prices is not None
        assert result.pdf is not None
        assert result.cdf is not None

    def test_price_methods_give_different_results(self, sample_options_data, market_inputs):
        """Test that last and mid price methods produce different results."""
        model_last = ModelParams(price_method="last")
        model_mid = ModelParams(price_method="mid")
        
        result_last = RND.from_dataframe(sample_options_data, market_inputs, model=model_last)
        result_mid = RND.from_dataframe(sample_options_data, market_inputs, model=model_mid)
        
        # Results should be different (unless bid-ask spread is zero, which it isn't in our test data)
        assert not np.array_equal(result_last.pdf, result_mid.pdf)

    def test_backward_compatibility_default_last(self, sample_options_data, market_inputs):
        """Test that default behavior (no explicit price_method) uses last price."""
        # No explicit model - should default to last price
        result_default = RND.from_dataframe(sample_options_data, market_inputs)
        
        # Explicit last price
        model_last = ModelParams(price_method="last")
        result_explicit_last = RND.from_dataframe(sample_options_data, market_inputs, model=model_last)
        
        # Results should be identical
        assert np.array_equal(result_default.pdf, result_explicit_last.pdf)
        assert np.array_equal(result_default.cdf, result_explicit_last.cdf)