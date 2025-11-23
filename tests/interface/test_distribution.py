import json
import os

import numpy as np
import pandas as pd

from oipd import MarketInputs
from oipd.interface.probability import Distribution


def test_distribution_class_vs_golden_master():
    """Distribution wrapper should match golden-master PDF/CDF exactly."""

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

    dist = Distribution(method="svi", method_options={"random_seed": 42})
    dist.fit(
        df_slice,
        market,
        vendor=None,
        fill_mode="strict",
        column_mapping=column_mapping,
    )

    gm_prices = np.array(gm["distribution"]["prices"])
    gm_pdf = np.array(gm["distribution"]["pdf"])
    gm_cdf = np.array(gm["distribution"]["cdf"])

    assert np.allclose(dist.prices, gm_prices, atol=1e-8)
    assert np.allclose(dist.pdf, gm_pdf, atol=1e-8)
    assert np.allclose(dist.cdf, gm_cdf, atol=1e-8)

    # Spot checks on probability helpers
    mid = float(np.median(gm_prices))
    assert np.isclose(dist.prob_below(mid) + dist.prob_above(mid), 1.0, atol=1e-8)


if __name__ == "__main__":
    test_distribution_class_vs_golden_master()
    print("Distribution matched golden master.")
