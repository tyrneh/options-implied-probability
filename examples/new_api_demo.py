"""
New API Demonstration Script
============================

This script demonstrates the new, modular API for the OIPD library.
It follows the pipeline:
1. Data Loading (oipd.sources)
2. Volatility Fitting (oipd.VolCurve)
3. Probability Derivation (oipd.Distribution)
"""

import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

from oipd import sources, MarketInputs
from oipd.interface.volatility import VolCurve, VolSurface
from oipd.interface.probability import Distribution

def main():
    print("Running New API Demo...\n")

    # ---------------------------------------------------------
    # 1. Data Loading (The Receptionist)
    # ---------------------------------------------------------
    print("1. Loading Data...")
    # In a real scenario, you might use:
    # df = sources.from_csv("data/my_options.csv")
    # df = sources.from_ticker("AAPL")
    
    # For this demo, we'll use the sample data we have
    data_path = "tests/../data/AAPL_data.csv"
    
    # We need to map the columns to the standard names if they differ
    # The sample data has 'expiration', 'type', 'last_price', etc.
    column_mapping = {
        "strike": "strike",
        "last_price": "last_price",
        "type": "option_type",
        "bid": "bid",
        "ask": "ask",
        "expiration": "expiration"
    }
    
    df_all = sources.from_csv(data_path, column_mapping=column_mapping)
    print(f"   Loaded {len(df_all)} rows.")
    
    # Let's pick a specific expiry for the single curve demo
    target_expiry = "2026-01-16"
    df_slice = df_all[df_all["expiration"] == target_expiry].copy()
    print(f"   Selected slice for {target_expiry}: {len(df_slice)} rows.")

    # ---------------------------------------------------------
    # 2. Market Setup
    # ---------------------------------------------------------
    print("\n2. Setting up Market Inputs...")
    market = MarketInputs(
        valuation_date=date(2025, 10, 6),
        expiry_date=date(2026, 1, 16),
        risk_free_rate=0.04,
        underlying_price=256.69,
    )
    print(f"   Valuation: {market.valuation_date}")
    print(f"   Expiry: {market.expiry_date}")
    print(f"   Spot: {market.underlying_price}")

    # ---------------------------------------------------------
    # 3. Volatility Fitting (The VolCurve)
    # ---------------------------------------------------------
    print("\n3. Fitting Volatility Curve...")
    # Initialize the model (stateless config)
    vc = VolCurve(method="svi", method_options={"random_seed": 42})
    
    # Fit the model (stateful operation)
    vc.fit(df_slice, market)
    
    print("   Fit complete.")
    print(f"   ATM Vol: {vc.at_money_vol:.2%}")
    print(f"   SVI Params: {vc.params}")

    # ---------------------------------------------------------
    # 4. Probability Derivation (The Distribution)
    # ---------------------------------------------------------
    print("\n4. Deriving Probability Distribution...")
    # We can derive it directly from the fitted VolCurve
    dist = vc.implied_distribution()
    
    print("   Derivation complete.")
    ev = dist.expected_value()
    print(f"   Expected Value: {ev:.2f}")
    
    # Probability Queries
    prob_below_200 = dist.prob_below(200)
    prob_above_300 = dist.prob_at_or_above(300)
    print(f"   Prob < 200: {prob_below_200:.2%}")
    print(f"   Prob >= 300: {prob_above_300:.2%}")

    # ---------------------------------------------------------
    # 5. VolSurface (Multi-Expiry)
    # ---------------------------------------------------------
    print("\n5. Fitting Full Volatility Surface...")
    # For surface, we don't pass a single expiry in MarketInputs
    market_surface = MarketInputs(
        valuation_date=date(2025, 10, 6),
        risk_free_rate=0.04,
        underlying_price=256.69,
    )
    
    vs = VolSurface(method="svi", method_options={"random_seed": 42})
    vs.fit(df_all, market_surface)
    
    print(f"   Surface fit complete. Found {len(vs.expiries)} expiries.")
    print(f"   Expiries: {[d.strftime('%Y-%m-%d') for d in vs.expiries[:3]]} ...")

    print("\nDemo Completed Successfully!")

if __name__ == "__main__":
    main()
