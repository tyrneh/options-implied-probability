"""
Test Skeleton for Regression / Golden Master Tests
====================================================
Tests that verify outputs match known-good reference values.

These tests protect against unintentional changes to mathematical outputs.
"""

import pytest
import json
import os
import numpy as np
import pandas as pd
from datetime import date


# =============================================================================
# Golden Master Fixture
# =============================================================================

@pytest.fixture
def golden_master():
    """Load golden master reference data."""
    gm_path = os.path.join(
        os.path.dirname(__file__), "../data/golden_master.json"
    )
    with open(gm_path, "r") as f:
        return json.load(f)


@pytest.fixture
def aapl_chain():
    """Load AAPL test data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "../../data/AAPL_data.csv"
    )
    return pd.read_csv(data_path)


# =============================================================================
# SVI Parameters Regression
# =============================================================================

class TestSviParametersRegression:
    """Tests that SVI calibration produces identical parameters to golden master."""

    def test_svi_params_match_golden_master(self, golden_master, aapl_chain):
        """Fitted SVI parameters match golden master exactly."""
        from oipd import VolCurve, MarketInputs
        
        # Setup market inputs from golden master
        gm = golden_master
        val_date = pd.Timestamp(gm["metadata"]["valuation_date"]).tz_localize(None)
        exp_date = pd.Timestamp(gm["metadata"]["expiry_date"]).tz_localize(None)
        
        market = MarketInputs(
            valuation_date=val_date,
            risk_free_rate=gm["metadata"]["r"],
            underlying_price=gm["metadata"]["underlying"],
        )
        
        # Filter to matching expiry
        df_slice = aapl_chain[aapl_chain["expiration"] == "2026-01-16"]
        
        # Fit with same seed
        column_mapping = {
            "strike": "strike",
            "last_price": "last_price",
            "type": "option_type",
            "bid": "bid",
            "ask": "ask",
            "expiration": "expiry",
        }
        
        vc = VolCurve(method="svi")
        vc.method_options = {"random_seed": 42}
        vc.fit(df_slice, market, column_mapping=column_mapping)
        
        # Verify parameters match
        for key, expected in gm["svi_params"].items():
            actual = vc.params[key]
            assert np.isclose(actual, expected, atol=1e-8), \
                f"Parameter {key}: expected {expected}, got {actual}"


# =============================================================================
# Implied Volatility Regression
# =============================================================================

class TestImpliedVolRegression:
    """Tests that IV evaluation matches golden master."""

    def test_implied_vols_match_golden_master(self, golden_master, aapl_chain):
        """Implied vols at test strikes match golden master."""
        from oipd import VolCurve, MarketInputs
        
        gm = golden_master
        val_date = pd.Timestamp(gm["metadata"]["valuation_date"]).tz_localize(None)
        exp_date = pd.Timestamp(gm["metadata"]["expiry_date"]).tz_localize(None)
        
        market = MarketInputs(
            valuation_date=val_date,
            risk_free_rate=gm["metadata"]["r"],
            underlying_price=gm["metadata"]["underlying"],
        )
        
        df_slice = aapl_chain[aapl_chain["expiration"] == "2026-01-16"]
        
        column_mapping = {
            "strike": "strike",
            "last_price": "last_price",
            "type": "option_type",
            "bid": "bid",
            "ask": "ask",
            "expiration": "expiry",
        }
        
        vc = VolCurve(method="svi")
        vc.method_options = {"random_seed": 42}
        vc.fit(df_slice, market, column_mapping=column_mapping)
        
        test_strikes = np.array(gm["test_points"]["strikes"])
        expected_ivs = np.array(gm["test_points"]["implied_vols"])
        actual_ivs = vc(test_strikes)
        
        assert np.allclose(actual_ivs, expected_ivs, atol=1e-8)


# =============================================================================
# PDF/CDF Regression
# =============================================================================

class TestDistributionRegression:
    """Tests that probability distribution matches golden master."""

    def test_pdf_matches_golden_master(self, golden_master, aapl_chain):
        """PDF values match golden master."""
        from oipd import VolCurve, MarketInputs
        
        gm = golden_master
        val_date = pd.Timestamp(gm["metadata"]["valuation_date"]).tz_localize(None)
        exp_date = pd.Timestamp(gm["metadata"]["expiry_date"]).tz_localize(None)
        
        market = MarketInputs(
            valuation_date=val_date,
            risk_free_rate=gm["metadata"]["r"],
            underlying_price=gm["metadata"]["underlying"],
        )
        
        df_slice = aapl_chain[aapl_chain["expiration"] == "2026-01-16"]
        
        column_mapping = {
            "strike": "strike",
            "last_price": "last_price",
            "type": "option_type",
            "bid": "bid",
            "ask": "ask",
            "expiration": "expiry",
        }
        
        vc = VolCurve(method="svi")
        vc.method_options = {"random_seed": 42}
        vc.fit(df_slice, market, column_mapping=column_mapping)
        prob = vc.implied_distribution()
        
        expected_prices = np.array(gm["distribution"]["prices"])
        expected_pdf = np.array(gm["distribution"]["pdf"])
        expected_cdf = np.array(gm["distribution"]["cdf"])
        
        assert np.allclose(prob.prices, expected_prices, atol=1e-8)
        assert np.allclose(prob.pdf, expected_pdf, atol=1e-8)
        assert np.allclose(prob.cdf, expected_cdf, atol=1e-8)
