"""Tests for put-call parity preprocessing functionality."""

import pytest
import pandas as pd
import numpy as np
import warnings

from oipd.core.parity import (
    infer_forward_from_atm,
    apply_put_call_parity,
    detect_parity_opportunity,
    preprocess_with_parity,
)


class TestDetectParityOpportunity:
    """Test detection of when parity preprocessing would be beneficial."""
    
    def test_detect_option_type_format(self):
        """Test detection with option_type format."""
        df = pd.DataFrame({
            'strike': [95, 95, 100, 100, 105],
            'last_price': [2.5, 1.2, 1.8, 2.1, 0.5],
            'option_type': ['C', 'P', 'C', 'P', 'C']
        })
        
        assert detect_parity_opportunity(df) is True
        
    def test_detect_separate_price_format(self):
        """Test detection with separate call_price/put_price format."""
        df = pd.DataFrame({
            'strike': [95, 100, 105],
            'call_price': [2.5, 1.8, 0.5],
            'put_price': [1.2, 2.1, 3.5]
        })
        
        assert detect_parity_opportunity(df) is True
        
    def test_detect_no_opportunity_calls_only(self):
        """Test that calls-only data returns False."""
        df = pd.DataFrame({
            'strike': [95, 100, 105],
            'last_price': [2.5, 1.8, 0.5],
            'option_type': ['C', 'C', 'C']
        })
        
        assert detect_parity_opportunity(df) is False
        
    def test_detect_requires_pair(self):
        """Parity detection requires at least one same-strike pair."""
        df = pd.DataFrame({
            'strike': [95, 100],
            'last_price': [2.5, 1.2],
            'option_type': ['C', 'P']
        })

        assert detect_parity_opportunity(df) is False

    def test_detect_single_pair(self):
        """Detection should succeed with a single call-put pair."""
        df = pd.DataFrame({
            'strike': [100, 100],
            'last_price': [2.5, 1.2],
            'option_type': ['C', 'P']
        })

        assert detect_parity_opportunity(df) is True


class TestInferForwardFromATM:
    """Test forward price inference from ATM call-put pairs."""
    
    @pytest.fixture
    def sample_options_data(self):
        """Create sample options data with known forward price."""
        # Create data where forward should be ~100
        strikes = [95, 100, 105]
        spot = 100
        forward = 100.5  # Slightly above spot due to interest rates
        discount_factor = 0.99
        
        data = []
        for strike in strikes:
            # Calculate theoretical call and put prices
            call_intrinsic = max(0, discount_factor * (forward - strike))
            put_intrinsic = max(0, discount_factor * (strike - forward))
            
            # Add some time value
            call_price = call_intrinsic + 1.0
            put_price = put_intrinsic + 1.0
            
            data.append({'strike': strike, 'last_price': call_price, 'option_type': 'C'})
            data.append({'strike': strike, 'last_price': put_price, 'option_type': 'P'})
            
        return pd.DataFrame(data)
    
    def test_infer_forward_option_type_format(self, sample_options_data):
        """Test forward inference with option_type format."""
        spot_price = 100.0
        discount_factor = 0.99
        
        forward = infer_forward_from_atm(sample_options_data, spot_price, discount_factor)
        
        # Should be close to our expected forward price of 100.5
        assert 99.5 <= forward <= 101.5  # Allow some tolerance
        
    def test_infer_forward_separate_price_format(self):
        """Test forward inference with separate price columns."""
        df = pd.DataFrame({
            'strike': [95, 100, 105],
            'call_price': [6.0, 2.0, 0.5],
            'put_price': [1.0, 2.5, 5.0]
        })
        
        spot_price = 100.0
        discount_factor = 0.99
        
        forward = infer_forward_from_atm(df, spot_price, discount_factor)
        
        # From 100 strike: F = 100 + (2.0 - 2.5) / 0.99 = 100 - 0.505 = 99.495
        expected = 100 + (2.0 - 2.5) / 0.99
        assert abs(forward - expected) < 0.1
        
    def test_infer_forward_no_pairs(self):
        """Test that error is raised when no call-put pairs exist."""
        df = pd.DataFrame({
            'strike': [95, 100, 105],
            'last_price': [2.5, 1.8, 0.5],
            'option_type': ['C', 'C', 'C']  # No puts
        })
        
        with pytest.raises(ValueError, match="No strikes found with both call and put"):
            infer_forward_from_atm(df, 100.0, 0.99)


class TestApplyPutCallParity:
    """Test the core put-call parity application logic."""
    
    @pytest.fixture
    def balanced_options_data(self):
        """Create balanced options data for testing."""
        return pd.DataFrame({
            'strike': [95, 95, 100, 100, 105, 105],
            'last_price': [6.0, 1.0, 2.5, 2.0, 0.5, 4.5],
            'option_type': ['C', 'P', 'C', 'P', 'C', 'P']
        })
    
    def test_parity_basic_application(self, balanced_options_data):
        """Test basic put-call parity application."""
        forward_price = 100.0
        discount_factor = 0.99
        
        result = apply_put_call_parity(balanced_options_data, forward_price, discount_factor)
        
        # Should have one row per strike
        assert len(result) == 3
        assert 'last_price' in result.columns
        assert 'source' in result.columns
        
        # Above forward (105): should use original call
        call_105 = result[result['strike'] == 105]
        assert len(call_105) == 1
        assert call_105['source'].iloc[0] == 'call'
        assert call_105['last_price'].iloc[0] == 0.5  # Original call price
        
        # At/below forward (95, 100): should use synthetic calls from puts
        put_95 = result[result['strike'] == 95]
        assert put_95['source'].iloc[0] == 'put_converted'
        
    def test_parity_boundary_condition(self, balanced_options_data):
        """Test boundary condition where K = F exactly."""
        forward_price = 100.0  # Exactly at 100 strike
        discount_factor = 0.99
        
        result = apply_put_call_parity(balanced_options_data, forward_price, discount_factor)
        
        # Strike 100 should be treated as K <= F (use put)
        strike_100 = result[result['strike'] == 100]
        assert strike_100['source'].iloc[0] == 'put_converted'
        
class TestPreprocessWithParity:
    """Test the main preprocessing entry point."""
    
    def test_preprocess_with_parity_beneficial(self):
        """Test preprocessing when parity is beneficial."""
        df = pd.DataFrame({
            'strike': [95, 95, 100, 100, 105, 105],
            'last_price': [6.0, 1.0, 2.5, 2.0, 0.5, 4.5],
            'option_type': ['C', 'P', 'C', 'P', 'C', 'P']
        })
        
        spot_price = 100.0
        discount_factor = 0.99
        
        result = preprocess_with_parity(df, spot_price, discount_factor)
        
        # Should return cleaned data with 'last_price' column
        assert 'last_price' in result.columns
        assert len(result) == 3  # One row per strike
        
    def test_preprocess_no_benefit_calls_only(self):
        """Test preprocessing with calls-only data (no benefit)."""
        df = pd.DataFrame({
            'strike': [95, 100, 105],
            'last_price': [2.5, 1.8, 0.5]
        })
        
        spot_price = 100.0
        discount_factor = 0.99
        
        result = preprocess_with_parity(df, spot_price, discount_factor)
        
        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, df)
        
    def test_preprocess_fallback_on_error(self, monkeypatch):
        """Test that preprocessing falls back gracefully on errors."""
        df = pd.DataFrame({
            'strike': [100, 100],
            'last_price': [2.0, 1.5],
            'option_type': ['C', 'P']
        })

        spot_price = 100.0
        discount_factor = 0.99

        def boom(*args, **kwargs):
            raise ValueError("boom")

        monkeypatch.setattr("oipd.core.parity.infer_forward_from_atm", boom)

        with warnings.catch_warnings(record=True) as w:
            result = preprocess_with_parity(df, spot_price, discount_factor)

            assert any("Put-call parity preprocessing failed" in str(msg.message) for msg in w)

        pd.testing.assert_frame_equal(result, df)


class TestIntegrationScenarios:
    """Integration tests with realistic market scenarios."""
    
    def test_realistic_market_scenario(self):
        """Test with realistic market data scenario."""
        # Create realistic options data
        # Spot = 100, forward ~= 100.25 (5% rate, 90 days)
        strikes = [85, 90, 95, 100, 105, 110, 115]
        
        data = []
        for strike in strikes:
            # Realistic call prices (decreasing with strike)
            call_price = max(0.1, 15 - 0.15 * strike + np.random.uniform(-0.2, 0.2))
            # Realistic put prices (increasing with strike)  
            put_price = max(0.1, 0.05 * strike - 2 + np.random.uniform(-0.2, 0.2))
            
            data.append({'strike': strike, 'last_price': call_price, 'option_type': 'C'})
            data.append({'strike': strike, 'last_price': put_price, 'option_type': 'P'})
        
        df = pd.DataFrame(data)
        
        spot_price = 100.0
        discount_factor = np.exp(-0.05 * 90/365)  # 5% rate, 90 days
        
        result = preprocess_with_parity(df, spot_price, discount_factor)
        
        # Basic checks
        assert 'last_price' in result.columns
        assert len(result) == len(strikes)
        assert all(result['last_price'] > 0)  # All prices should be positive
        assert 'source' in result.columns
        
        # Should have mix of call and put_converted
        sources = result['source'].value_counts()
        assert 'call' in sources
        assert 'put_converted' in sources
        
    def test_missing_data_handling(self):
        """Test handling of missing data in realistic scenarios."""
        # Create data with some missing puts or calls
        df = pd.DataFrame({
            'strike': [95, 95, 100, 105, 105],  # Missing put at 100, call at 105
            'last_price': [6.0, 1.0, 2.5, 0.5, 4.5],
            'option_type': ['C', 'P', 'C', 'C', 'P']  # Note: 105 has both C and P
        })
        
        spot_price = 100.0
        discount_factor = 0.99
        
        result = preprocess_with_parity(df, spot_price, discount_factor)

        # Should return rows only where usable prices exist
        assert len(result) == 2
        assert set(result['strike']) == {95, 105}
        assert result.loc[result['strike'] == 95, 'source'].iloc[0] == 'put_converted'
        assert result.loc[result['strike'] == 105, 'source'].iloc[0] == 'call'


if __name__ == "__main__":
    pytest.main([__file__])
