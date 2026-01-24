"""Unit tests for plotting argument refactor (x_axis/y_axis)."""

import pytest
import warnings
import pandas as pd
import numpy as np
from oipd.presentation.iv_plotting import plot_iv_smile


def test_plot_args_new_style():
    """Verify x_axis and y_axis work correctly."""
    df = pd.DataFrame({
        "strike": [100, 110, 120],
        "fitted_iv": [0.2, 0.2, 0.2],
    })
    
    # Test 1: x_axis="strike"
    try:
        plot_iv_smile(df, x_axis="strike", y_axis="iv", show_forward=False)
    except Exception as e:
        pytest.fail(f"New arguments failed: {e}")

    # Test 2: y_axis="total_variance"
    try:
        plot_iv_smile(df, x_axis="strike", y_axis="total_variance", t_to_expiry=1.0, show_forward=False)
    except Exception as e:
         pytest.fail(f"New arguments failed: {e}")





def test_invalid_axis_choice():
    """Verify invalid x_axis raises ValueError."""
    df = pd.DataFrame({"strike": [100], "fitted_iv": [0.2]})
    # Test explicit log_moneyness usage
    try:
        # Mock forward price for log_moneyness
        from oipd.presentation.iv_plotting import ForwardPriceAnnotation
        fwd = ForwardPriceAnnotation(value=100.0, label="Ref")
        plot_iv_smile(df, x_axis="log_moneyness", forward_price=fwd, show_forward=False)
    except Exception as e:
        pytest.fail(f"log_moneyness failed: {e}")

    with pytest.raises(ValueError, match="x_axis must be"):
        plot_iv_smile(df, x_axis="log_strike_over_forward")

    with pytest.raises(ValueError, match="x_axis must be"):
        plot_iv_smile(df, x_axis="invalid")

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_plot_args_new_style()
        print("test_plot_args_new_style passed")
        # For warning tests, we need pytest context or manual catch
        print("Run with pytest to verify warnings.")
    except Exception as e:
        print(f"Failed: {e}")
