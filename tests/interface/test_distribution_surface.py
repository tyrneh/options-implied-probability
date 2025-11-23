import json
import os

import numpy as np
import pandas as pd

from oipd import MarketInputs
from oipd.interface.volatility import VolSurface


def test_distribution_surface_slice_matches_golden_master():
    """DistributionSurface slice should match golden-master PDF/CDF exactly."""

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

    dist_surface = surface.implied_distribution()
    dist_slice = dist_surface.slice(expiry_date)

    gm_prices = np.array(gm["distribution"]["prices"])
    gm_pdf = np.array(gm["distribution"]["pdf"])
    gm_cdf = np.array(gm["distribution"]["cdf"])

    assert np.allclose(dist_slice.prices, gm_prices, atol=1e-8)
    assert np.allclose(dist_slice.pdf, gm_pdf, atol=1e-8)
    assert np.allclose(dist_slice.cdf, gm_cdf, atol=1e-8)


if __name__ == "__main__":
    test_distribution_surface_slice_matches_golden_master()
    print("DistributionSurface slice matched golden master.")
