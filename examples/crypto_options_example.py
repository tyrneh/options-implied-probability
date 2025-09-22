"""
Example: Analyzing Bitcoin Options Implied Probability Distribution using Bybit

This example demonstrates how to use OIPD with cryptocurrency options data
from Bybit to extract risk-neutral probability distributions.
"""

from datetime import date, timedelta
import matplotlib.pyplot as plt

from oipd import RND, MarketInputs

def main():
    """Main example function."""
    
    print("🚀 OIPD Crypto Options Example - Bitcoin via Bybit")
    print("=" * 60)
    
    # Step 1: Define market parameters for Bitcoin options
    # Note: For crypto, risk-free rate should reflect funding costs in crypto markets
    market = MarketInputs(
        valuation_date=date.today(),
        expiry_date=date.today() + timedelta(days=30),  # 30 days from now
        risk_free_rate=0.05,  # 5% annual rate (adjust based on crypto funding rates)
    )
    
    print(f"📅 Analysis Date: {market.valuation_date}")
    print(f"📅 Expiry Date: {market.expiry_date}")
    print(f"💰 Risk-free Rate: {market.risk_free_rate:.1%}")
    print()
    
    try:
        # Step 2: List available expiry dates for Bitcoin options
        print("📋 Available Bitcoin option expiry dates:")
        expiry_dates = RND.list_expiry_dates("BTCUSDT", vendor="bybit")
        for i, expiry in enumerate(expiry_dates[:10]):  # Show first 10
            print(f"   {i+1:2d}. {expiry}")
        if len(expiry_dates) > 10:
            print(f"   ... and {len(expiry_dates) - 10} more")
        print()
        
        # Step 3: Use the first available expiry date
        if expiry_dates:
            target_expiry = expiry_dates[0]
            market = MarketInputs(
                valuation_date=date.today(),
                expiry_date=date.fromisoformat(target_expiry),
                risk_free_rate=0.05,
            )
            
            print(f"🎯 Analyzing Bitcoin options expiring on {target_expiry}")
            print("⏳ Fetching data from Bybit...")
            
            # Step 4: Fetch Bitcoin options data and estimate RND
            est = RND.from_ticker("BTC", market, vendor="bybit")
            
            print("✅ Data fetched successfully!")
            print()
            
            # Step 5: Display summary
            print("📊 Market Summary:")
            print(est.summary())
            print()
            
            # Step 6: Calculate key probabilities
            current_price = est.market.underlying_price
            
            # Calculate probability ranges
            prob_above_current = est.prob_at_or_above(current_price * 1.1)  # +10%
            prob_below_current = est.prob_below(current_price * 0.9)        # -10%
            prob_stable = 1 - prob_above_current - prob_below_current       # ±10%
            
            print("🎲 Probability Analysis:")
            print(f"   P(BTC ≥ ${current_price * 1.1:,.0f}) = {prob_above_current:.1%}")
            print(f"   P(BTC ≤ ${current_price * 0.9:,.0f}) = {prob_below_current:.1%}")
            print(f"   P(${current_price * 0.9:,.0f} < BTC < ${current_price * 1.1:,.0f}) = {prob_stable:.1%}")
            print()
            
            # Step 7: Generate probability distribution plot
            print("📈 Generating probability distribution plot...")
            
            fig = est.plot(
                kind="both",
                figsize=(12, 6),
                title=f"Bitcoin Options Implied Probability Distribution\nExpiry: {target_expiry}",
                source="Data: Bybit API via OIPD",
                show_current_price=True
            )
            
            # Save the plot
            plt.tight_layout()
            plt.savefig("bitcoin_probability_distribution.png", dpi=300, bbox_inches='tight')
            print("💾 Plot saved as 'bitcoin_probability_distribution.png'")
            
            # Step 8: Export results to CSV
            est.to_csv("bitcoin_rnd_results.csv")
            print("💾 Results exported to 'bitcoin_rnd_results.csv'")
            print()
            
            # Step 9: Advanced analysis - quartiles
            df_results = est.to_frame()
            
            # Find quartile prices
            q25_price = est.prices[est.cdf >= 0.25][0] if any(est.cdf >= 0.25) else est.prices[-1]
            q50_price = est.prices[est.cdf >= 0.50][0] if any(est.cdf >= 0.50) else est.prices[-1]
            q75_price = est.prices[est.cdf >= 0.75][0] if any(est.cdf >= 0.75) else est.prices[-1]
            
            print("📊 Price Quartiles (Market Expectations):")
            print(f"   25th percentile: ${q25_price:,.0f}")
            print(f"   50th percentile: ${q50_price:,.0f} (median)")
            print(f"   75th percentile: ${q75_price:,.0f}")
            print()
            
            print("✨ Analysis complete! Check the generated files for detailed results.")
            
        else:
            print("❌ No Bitcoin options expiry dates found on Bybit")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Ensure you have pybit installed: pip install pybit")
        print("   2. Check your internet connection")
        print("   3. Verify Bybit API is accessible")
        print("   4. Try a different expiry date")


if __name__ == "__main__":
    main()
