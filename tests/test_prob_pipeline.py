import json
import os
import pytest
import pandas as pd
import numpy as np
from datetime import date

from oipd import MarketInputs
from oipd.pipelines.prob_estimation import derive_distribution_internal
from oipd.pipelines.market_inputs import resolve_market


def test_prob_pipeline_vs_golden_master():
    # 1. Load Golden Master
    gm_path = os.path.join(os.path.dirname(__file__), "data/golden_master.json")
    with open(gm_path, "r") as f:
        gm = json.load(f)

    # 2. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "../data/AAPL_data.csv")
    df_appl = pd.read_csv(data_path)
    df_slice = df_appl[df_appl["expiration"] == "2026-01-16"]

    # 3. Setup Inputs (Same as GM generation)
    # Ensure valuation_date is compatible with pandas operations
    val_date = pd.Timestamp(gm["metadata"]["valuation_date"]).tz_localize(None)
    expiry_date = pd.Timestamp(gm["metadata"]["expiry_date"]).tz_localize(None)

    market = MarketInputs(
        valuation_date=val_date,
        expiry_date=expiry_date,
        risk_free_rate=gm["metadata"]["r"],
        underlying_price=gm["metadata"]["underlying"],
    )

    resolved_market = resolve_market(market, vendor=None, mode="strict")

    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
    }
    df_slice = df_slice.rename(columns=column_mapping)

    # 4. Run Pipeline
    # We must use the same seed as the GM generation (42)
    method_options = {"random_seed": 42}

    prices, pdf, cdf, meta = derive_distribution_internal(
        df_slice,
        resolved_market,
        pricing_engine="black76",
        price_method="mid",
        method="svi",
        method_options=method_options,
    )

    # 5. Verify PDF/CDF
    gm_prices = np.array(gm["distribution"]["prices"])
    gm_pdf = np.array(gm["distribution"]["pdf"])
    gm_cdf = np.array(gm["distribution"]["cdf"])

    # Verify Shapes
    assert prices.shape == gm_prices.shape, "Price grid shape mismatch"
    assert pdf.shape == gm_pdf.shape, "PDF shape mismatch"
    assert cdf.shape == gm_cdf.shape, "CDF shape mismatch"

    # Verify Values
    # We use a slightly looser tolerance for the distribution arrays as they are derived
    # but since we are wrapping the exact same code, it should be very close.
    print("\nComparison of Distribution:")
    print(f"Max Price Diff: {np.max(np.abs(prices - gm_prices))}")
    print(f"Max PDF Diff: {np.max(np.abs(pdf - gm_pdf))}")
    print(f"Max CDF Diff: {np.max(np.abs(cdf - gm_cdf))}")

    assert np.allclose(prices, gm_prices, atol=1e-8), "Mismatch in Price Grid"
    assert np.allclose(pdf, gm_pdf, atol=1e-8), "Mismatch in PDF"
    assert np.allclose(cdf, gm_cdf, atol=1e-8), "Mismatch in CDF"


if __name__ == "__main__":
    test_prob_pipeline_vs_golden_master()
    print("\nTest Passed!")
