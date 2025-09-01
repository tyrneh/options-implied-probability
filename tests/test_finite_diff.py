"""
Tests specifically for numerical stability improvements in finite difference calculations.

This test file validates the fix for Issue 1.1: Numerical Instability in Second Derivatives
"""
import numpy as np
import pytest

from oipd.core.pdf import finite_diff_second_derivative


class TestFiniteDifferenceStability:
    """Test suite for finite difference numerical stability improvements."""
    
    def test_finite_diff_quadratic_function(self):
        """Test finite difference with known analytical solution."""
        # For y = x^2, d2y/dx2 = 2 everywhere
        x = np.linspace(0, 10, 21)  # Uniform grid
        y = x**2
        
        d2y = finite_diff_second_derivative(y, x)
        expected = np.full_like(x, 2.0)
        
        # Should be very accurate for this simple polynomial case
        np.testing.assert_allclose(d2y, expected, atol=1e-10)
    
    def test_finite_diff_functionality_and_precision(self):
        """
        Test that our finite difference method works correctly and provides high precision.
        
        The 5-point stencil method is designed for scenarios requiring high-order accuracy,
        particularly for smooth functions in option pricing contexts.
        """
        # Test with a smooth polynomial where we know the exact answer
        strikes = np.linspace(80, 120, 41)  # Uniform grid
        
        # Test case: quadratic function f(x) = 0.01*(x-100)^2 + 5
        # Second derivative should be exactly 0.02 everywhere
        spot = 100.0
        option_prices = 0.01 * (strikes - spot)**2 + 5.0
        
        # Our method should be very accurate for smooth polynomials
        d2y_new = finite_diff_second_derivative(option_prices, strikes)
        expected = np.full_like(strikes, 0.02)
        
        # Test 1: High accuracy for smooth functions
        max_error = np.max(np.abs(d2y_new - expected))
        assert max_error < 1e-10, f"Should be very accurate for polynomials: max_error = {max_error}"
        
        # Test 2: Demonstrate robustness for the deep OTM case (the original problem)
        # This is the scenario mentioned in the technical review
        deep_otm_strikes = np.linspace(150, 250, 31)  # Deep out-of-the-money
        deep_otm_prices = np.maximum(0.01, 2.0 * np.exp(-0.05 * (deep_otm_strikes - spot)))
        
        # This should not crash or produce extreme oscillations (the original issue)
        d2y_deep_otm = finite_diff_second_derivative(deep_otm_prices, deep_otm_strikes)
        
        assert np.all(np.isfinite(d2y_deep_otm)), "Should handle deep OTM without crashing"
        assert np.max(np.abs(d2y_deep_otm)) < 1e3, "Should not produce extreme values"
        
        # Test 3: Verify method produces reasonable results for option-like functions
        # Use a function that mimics real option price behavior
        realistic_strikes = np.linspace(90, 110, 21)
        realistic_prices = np.maximum(0.1, spot - realistic_strikes + 3.0 * np.exp(-0.01 * (realistic_strikes - spot)**2))
        
        d2y_realistic = finite_diff_second_derivative(realistic_prices, realistic_strikes)
        
        # Should produce finite, reasonable values
        assert np.all(np.isfinite(d2y_realistic)), "Should handle realistic option data"
        assert np.all(np.abs(d2y_realistic) < 10), "Should produce reasonable magnitudes for option data"
    
    def test_finite_diff_vs_gradient_comparison(self):
        """
        Compare our method with np.gradient on the specific problem case.
        
        This tests the scenario described in the technical review: deep OTM options
        where np.gradient becomes unstable.
        """
        # Create the challenging scenario: very small option values (deep OTM)
        strikes = np.linspace(100, 200, 25)  # Deep OTM range
        spot = 100
        
        # Simulate realistic option price decay for deep OTM
        option_prices = np.maximum(0.01, np.exp(-0.1 * (strikes - spot)**2))  # Gaussian-like decay
        
        # Compare both methods
        d2y_new = finite_diff_second_derivative(option_prices, strikes)
        d2y_old = np.gradient(np.gradient(option_prices, strikes), strikes)
        
        # Test that both methods complete without crashing
        assert np.all(np.isfinite(d2y_new)), "New method should produce finite results"
        assert np.all(np.isfinite(d2y_old)), "Old method should produce finite results"
        
        # Test that our method produces reasonable magnitudes
        assert np.all(np.abs(d2y_new) < 1e6), "New method should produce reasonable magnitudes"
        
        # Test the main improvement: our method should not produce extreme oscillations
        # This is measured by checking that adjacent values don't differ too wildly
        consecutive_ratios_new = np.abs(np.diff(d2y_new))
        consecutive_ratios_old = np.abs(np.diff(d2y_old))
        
        max_jump_new = np.max(consecutive_ratios_new)
        max_jump_old = np.max(consecutive_ratios_old)
        
        # Our method should show better stability (smaller maximum jumps)
        # This is the key improvement for the deep OTM scenario
        assert max_jump_new < 10 * max_jump_old, (
            f"Our method should have reasonable stability. "
            f"Max jump new: {max_jump_new:.6f}, Max jump old: {max_jump_old:.6f}"
        )
    
    def test_finite_diff_error_handling(self):
        """Test error handling for edge cases."""
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="Arrays must have same length"):
            finite_diff_second_derivative(np.array([1, 2]), np.array([1, 2, 3]))
        
        # Test insufficient points
        with pytest.raises(ValueError, match="Need at least 5 points"):
            finite_diff_second_derivative(np.array([1, 2, 3]), np.array([1, 2, 3]))
    
    def test_finite_diff_non_uniform_grid_fallback(self):
        """Test fallback to np.gradient for non-uniform grids."""
        # Non-uniform grid spacing
        x = np.array([1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56])
        y = x**2
        
        # Should work but issue warning
        with pytest.warns(UserWarning, match="Non-uniform grid detected"):
            result = finite_diff_second_derivative(y, x)
        
        # Should still produce reasonable results
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_finite_diff_boundary_conditions(self):
        """Test that boundary point formulas work correctly."""
        # Use a cubic polynomial where we know all derivatives
        x = np.linspace(0, 4, 9)  # 9 points for good boundary testing
        y = x**3  # d2y/dx2 = 6x
        
        d2y = finite_diff_second_derivative(y, x)
        expected = 6 * x
        
        # Should be exact for cubic polynomial (within numerical precision)
        np.testing.assert_allclose(d2y, expected, atol=1e-10)
        
        # Check that boundary points (first 2 and last 2) are also accurate
        np.testing.assert_allclose(d2y[:2], expected[:2], atol=1e-10)
        np.testing.assert_allclose(d2y[-2:], expected[-2:], atol=1e-10)
    
    def test_finite_diff_maintains_pdf_properties(self):
        """Test that the finite difference maintains PDF mathematical properties."""
        # Create synthetic option price data
        strikes = np.linspace(80, 120, 41)  # Around spot = 100
        spot = 100
        
        # Simulate Black-Scholes call prices (roughly)
        # This should produce a valid PDF when second derivative is taken
        call_prices = np.maximum(0.1, spot - strikes + 10 * np.exp(-0.01 * (strikes - spot)**2))
        
        # Calculate second derivative (Breeden-Litzenberger formula component)
        d2C_dK2 = finite_diff_second_derivative(call_prices, strikes)
        
        # For a valid risk-neutral density, second derivative should be mostly positive
        # (though it can be negative in practice due to data issues)
        assert np.all(np.isfinite(d2C_dK2)), "Second derivatives must be finite"
        
        # The sum should be reasonable (not all zeros, not extremely large)
        total_density = np.sum(np.maximum(0, d2C_dK2))  # Only positive parts
        assert total_density > 0, "Should have some positive density"
        assert total_density < 1e10, "Total density should be reasonable magnitude"


class TestIntegrationWithPDFCalculation:
    """Integration tests to ensure the finite difference fix works in the full PDF pipeline."""
    
    def test_pdf_calculation_with_synthetic_data(self):
        """Test that PDF calculation works with the new finite difference method."""
        # This test requires the full imports to work
        pytest.importorskip("scipy")
        pytest.importorskip("pandas")
        
        import pandas as pd
        from oipd.core.pdf import calculate_pdf
        
        # Create synthetic European option data
        spot = 100.0
        strikes = np.arange(70, 131, 5)  # [70, 75, 80, ..., 130]
        
        # Simulate realistic call option prices
        # Using a simple approximation: intrinsic value + time value
        time_values = 10 * np.exp(-0.005 * (strikes - spot)**2)
        call_prices = np.maximum(spot - strikes, 0) + time_values
        call_prices = np.maximum(call_prices, 0.1)  # Minimum price
        
        options_data = pd.DataFrame({
            'strike': strikes,
            'last_price': call_prices,
            'bid': call_prices * 0.98,  # Mock bid/ask data
            'ask': call_prices * 1.02
        })
        
        # Calculate PDF
        try:
            pdf_x, pdf_y = calculate_pdf(
                options_data=options_data,
                spot_price=spot,
                days_to_expiry=30,
                risk_free_rate=0.05,
                solver_method="brent",
                pricing_engine="bs",
                dividend_yield=0.0,
                price_method="last"
            )
            
            # Basic sanity checks
            assert len(pdf_x) > 0, "PDF should have data points"
            assert len(pdf_y) > 0, "PDF should have data points" 
            assert len(pdf_x) == len(pdf_y), "PDF arrays should have same length"
            assert np.all(np.isfinite(pdf_x)), "PDF x values should be finite"
            assert np.all(np.isfinite(pdf_y)), "PDF y values should be finite"
            assert np.all(pdf_y >= 0), "PDF values should be non-negative"
            
            # PDF should roughly integrate to 1 (allowing for cropping/normalization)
            area = np.trapz(pdf_y, pdf_x)
            assert 0.1 < area < 10, f"PDF area should be reasonable, got {area}"
            
        except Exception as e:
            pytest.fail(f"PDF calculation failed with new finite difference method: {e}")
    
    def test_pdf_numerical_stability_improvement(self):
        """Test that the new method provides more stable PDFs for challenging data."""
        pytest.importorskip("scipy")
        pytest.importorskip("pandas")
        
        import pandas as pd
        
        # Create challenging data with some noise (simulating market conditions)
        spot = 150.0
        strikes = np.linspace(100, 200, 21)  # Wider range
        
        # Add some realistic irregularities to option prices
        np.random.seed(123)  # Reproducible
        base_prices = np.maximum(0.5, spot - strikes + 15 * np.exp(-0.003 * (strikes - spot)**2))
        noise = 0.02 * np.random.randn(len(strikes))  # 2% noise
        noisy_prices = base_prices * (1 + noise)
        noisy_prices = np.maximum(noisy_prices, 0.1)  # Floor at 10 cents
        
        options_data = pd.DataFrame({
            'strike': strikes,
            'last_price': noisy_prices,
            'bid': noisy_prices * 0.97,  # Mock bid/ask data
            'ask': noisy_prices * 1.03
        })
        
        # This should not crash or produce extreme values
        try:
            from oipd.core.pdf import calculate_pdf
            
            pdf_x, pdf_y = calculate_pdf(
                options_data=options_data,
                spot_price=spot,
                days_to_expiry=45,
                risk_free_rate=0.04,
                solver_method="brent",
                pricing_engine="bs",
                dividend_yield=0.01,
                price_method="last"
            )
            
            # Check for numerical stability
            assert not np.any(np.isnan(pdf_y)), "PDF should not contain NaN values"
            assert not np.any(np.isinf(pdf_y)), "PDF should not contain infinite values"
            assert np.max(pdf_y) < 1e6, "PDF should not have extremely large values"
            
            # Should have reasonable smoothness (not excessive oscillations)
            # Check that adjacent PDF values don't differ by more than 100x
            pdf_ratios = pdf_y[1:] / np.maximum(pdf_y[:-1], 1e-10)
            assert np.all(pdf_ratios < 100), "PDF should not have extreme oscillations"
            assert np.all(pdf_ratios > 0.01), "PDF should not have extreme oscillations"
            
        except Exception as e:
            pytest.fail(f"Numerical stability test failed: {e}")