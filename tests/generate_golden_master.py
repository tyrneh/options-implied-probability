import json
import numpy as np
import pandas as pd
from datetime import date
import os

from oipd import RND, MarketInputs, ModelParams


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, date):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


def generate_golden_master():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "../data/AAPL_data.csv")
    print(f"Loading data from {data_path}")
    df_appl = pd.read_csv(data_path)

    # Filter for 2026-01-16 expiration
    df_slice = df_appl[df_appl["expiration"] == "2026-01-16"]

    # 2. Setup Market
    market = MarketInputs(
        valuation_date=date(2025, 10, 6),
        expiry_date=date(2026, 1, 16),
        risk_free_rate=0.04,
        underlying_price=256.69,
    )

    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
    }

    # 3. Run Estimation
    print("Running RND estimation...")
    # Enforce deterministic seed for SVI calibration
    model_params = ModelParams(surface_random_seed=42)
    est = RND.from_dataframe(
        df_slice, market, column_mapping=column_mapping, model=model_params
    )

    # 4. Extract Data
    svi_params = est.svi_params()

    # Get implied vols at specific strikes for verification
    # We'll use the strikes from the input data to be safe
    test_strikes = [200.0, 250.0, 300.0]
    iv_smile = est.iv_smile(strikes=test_strikes)
    implied_vols = iv_smile["fitted_iv"].tolist()

    # Get PDF/CDF
    # We'll sample the PDF at specific points
    pdf_prices = est.prices
    pdf_values = est.pdf
    cdf_values = est.cdf

    # 5. Structure Output
    output = {
        "metadata": {
            "valuation_date": market.valuation_date.isoformat(),
            "expiry_date": market.expiry_date.isoformat(),
            "underlying": market.underlying_price,
            "r": market.risk_free_rate,
        },
        "svi_params": svi_params,
        "test_points": {"strikes": test_strikes, "implied_vols": implied_vols},
        "distribution": {"prices": pdf_prices, "pdf": pdf_values, "cdf": cdf_values},
    }

    # 6. Save
    output_path = os.path.join(os.path.dirname(__file__), "data/golden_master.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving Golden Master to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, cls=NumpyEncoder, indent=2)

    print("Done.")


if __name__ == "__main__":
    generate_golden_master()
