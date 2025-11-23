import json
import os

import numpy as np
import pandas as pd

from oipd import MarketInputs
from oipd.interface.volatility import VolSurface


def test_vol_surface_slice_matches_golden_master():
    """VolSurface slice should match golden-master SVI params and IVs."""

    gm_path = os.path.join(os.path.dirname(__file__), "../data/golden_master.json")
    with open(gm_path, "r") as f:
        gm = json.load(f)

    data_path = os.path.join(os.path.dirname(__file__), "../../data/AAPL_data.csv")
    df_appl = pd.read_csv(data_path)

    val_date = pd.Timestamp(gm["metadata"]["valuation_date"]).tz_localize(None)
    expiry_date = pd.Timestamp(gm["metadata"]["expiry_date"]).tz_localize(None)

    market = MarketInputs(
        valuation_date=val_date,
        expiry_date=None,
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

    surface = VolSurface(method="svi", method_options={"random_seed": 42})
    surface.fit(
        df_appl,
        market,
        vendor=None,
        fill_mode="strict",
        column_mapping=column_mapping,
    )

    slice_curve = surface.slice(expiry_date)

    gm_params = gm["svi_params"]
    for key, gm_value in gm_params.items():
        assert np.isclose(slice_curve.params[key], gm_value, atol=1e-8)

    test_strikes = np.array(gm["test_points"]["strikes"])
    gm_ivs = np.array(gm["test_points"]["implied_vols"])
    new_ivs = slice_curve(test_strikes)
    assert np.allclose(new_ivs, gm_ivs, atol=1e-8)


if __name__ == "__main__":
    test_vol_surface_slice_matches_golden_master()
    print("VolSurface slice matched golden master.")
