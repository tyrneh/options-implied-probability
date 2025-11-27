import json
import os
import pytest
import pandas as pd
import numpy as np
from datetime import date

from oipd import MarketInputs
from oipd.pipelines.vol_estimation import fit_vol_curve_internal
from oipd.pipelines.market_inputs import resolve_market


def test_vol_pipeline_vs_golden_master():
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
    # The pipeline accepts method_options for SVI
    method_options = {"random_seed": 42}

    vol_curve, metadata = fit_vol_curve_internal(
        df_slice,
        resolved_market,
        pricing_engine="black76",  # Default in RND
        price_method="mid",  # Default in RND
        method="svi",
        method_options=method_options,
    )

    # 5. Verify SVI Params
    # The vol_curve.params is a dict
    new_params = vol_curve.params
    gm_params = gm["svi_params"]

    print("\nComparison of SVI Parameters:")
    for key in gm_params:
        val_gm = gm_params[key]
        val_new = new_params[key]
        print(f"{key}: GM={val_gm:.8f}, New={val_new:.8f}")
        assert np.isclose(val_new, val_gm, atol=1e-8), f"Mismatch in {key}"

    # 6. Verify Implied Vols
    test_strikes = np.array(gm["test_points"]["strikes"])
    gm_ivs = np.array(gm["test_points"]["implied_vols"])
    new_ivs = vol_curve(test_strikes)

    print("\nComparison of Implied Vols:")
    for i, strike in enumerate(test_strikes):
        print(f"Strike {strike}: GM={gm_ivs[i]:.8f}, New={new_ivs[i]:.8f}")
        assert np.isclose(
            new_ivs[i], gm_ivs[i], atol=1e-8
        ), f"Mismatch in IV at strike {strike}"


if __name__ == "__main__":
    test_vol_pipeline_vs_golden_master()
    print("\nTest Passed!")
