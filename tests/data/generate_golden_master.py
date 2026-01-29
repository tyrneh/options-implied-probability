
"""
Generate Golden Master Data
===========================
Regenerates tests/data/golden_master.json using the current
VolCurve and ProbCurve API.

Usage:
    python tests/data/generate_golden_master.py
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import date
from oipd import VolCurve, MarketInputs

def generate_golden_master():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "../../data/AAPL_data.csv")
    df_appl = pd.read_csv(data_path)
    df_slice = df_appl[df_appl["expiration"] == "2026-01-16"]

    # 2. Setup Inputs
    val_date = date(2025, 1, 15)
    risk_free_rate = 0.045
    underlying_price = 220.0
    
    # Expiry from data (for metadata)
    expiry_date_str = "2026-01-16"
    expiry_date = date.fromisoformat(expiry_date_str)

    market = MarketInputs(
        valuation_date=val_date,
        risk_free_rate=risk_free_rate,
        underlying_price=underlying_price,
    )

    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
        "expiration": "expiry",
    }

    # 3. Fit VolCurve
    print("Fitting VolCurve...")
    vc = VolCurve(method="svi")
    vc.method_options = {"random_seed": 42}
    vc.fit(df_slice, market, column_mapping=column_mapping)

    # 4. Generate ProbCurve
    print("Generating ProbCurve...")
    prob = vc.implied_distribution()

    # 5. Extract Test Points
    test_strikes = [150.0, 200.0, 220.0, 250.0, 300.0]
    test_ivs = vc(test_strikes).tolist()

    # 6. Construct JSON Structure
    gm_data = {
        "metadata": {
            "valuation_date": val_date.isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "r": risk_free_rate,
            "underlying": underlying_price,
            "generated_by": "VolCurve",
        },
        "svi_params": {k: float(v) for k, v in vc.params.items() if isinstance(v, (int, float, np.number))},
        "test_points": {
            "strikes": test_strikes,
            "implied_vols": test_ivs,
        },
        "distribution": {
            "prices": prob.prices.tolist(),
            "pdf": prob.pdf.tolist(),
            "cdf": prob.cdf.tolist(),
        }
    }

    # 7. Save
    out_path = os.path.join(os.path.dirname(__file__), "golden_master.json")
    with open(out_path, "w") as f:
        json.dump(gm_data, f, indent=4)
    
    print(f"Saved golden master to {out_path}")

if __name__ == "__main__":
    generate_golden_master()
