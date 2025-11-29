import json
import os

import numpy as np
import pandas as pd

from oipd import MarketInputs
from oipd.market_inputs import resolve_market
from oipd.interface.volatility import VolCurve


def test_vol_curve_class_vs_golden_master():
    """VolCurve wrapper should match golden-master SVI params and IVs."""

    gm_path = os.path.join(os.path.dirname(__file__), "../data/golden_master.json")
    with open(gm_path, "r") as f:
        gm = json.load(f)

    data_path = os.path.join(os.path.dirname(__file__), "../../data/AAPL_data.csv")
    df_appl = pd.read_csv(data_path)
    df_slice = df_appl[df_appl["expiration"] == "2026-01-16"]

    val_date = pd.Timestamp(gm["metadata"]["valuation_date"]).tz_localize(None)
    expiry_date = pd.Timestamp(gm["metadata"]["expiry_date"]).tz_localize(None)

    market = MarketInputs(
        valuation_date=val_date,
        expiry_date=expiry_date,
        risk_free_rate=gm["metadata"]["r"],
        underlying_price=gm["metadata"]["underlying"],
    )

    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
    }

    vol_curve = VolCurve(method="svi", method_options={"random_seed": 42})
    vol_curve.fit(
        df_slice,
        market,
        vendor=None,
        fill_mode="strict",
        column_mapping=column_mapping,
    )

    gm_params = gm["svi_params"]
    for key, gm_value in gm_params.items():
        assert np.isclose(vol_curve.params[key], gm_value, atol=1e-8)

    test_strikes = np.array(gm["test_points"]["strikes"])
    gm_ivs = np.array(gm["test_points"]["implied_vols"])
    new_ivs = vol_curve(test_strikes)
    assert np.allclose(new_ivs, gm_ivs, atol=1e-8)


if __name__ == "__main__":
    test_vol_curve_class_vs_golden_master()
    print("VolCurve matched golden master.")
