from oipd import OIPDEstimator, MarketParams, ModelParams
import matplotlib.pyplot as plt
from datetime import date

# 1️⃣  what the library expects internally
#     strike , last_price , bid , ask
column_mapping={
    "Strike": "strike",
    "Last Price": "last_price",
    "Bid": "bid",
    "Ask": "ask"
    }

# 2️⃣  market parameters
market = MarketParams(current_price=117.90,
                      current_date=date(2025,3,3),
                      expiry_date=date(2025,5,16),
                      risk_free_rate=0.04)

# 3️⃣  optional model knobs (could omit)
model  = ModelParams(fit_kde=True)

# 4️⃣  run
est = OIPDEstimator.from_csv(
        "data/nvidia_date20250303_strikedate20250516_price11790.csv",
        market,
        model=model,
        column_mapping=column_mapping,      # ← here
      )

df = est.to_frame()
df.head()

# --- 2. pull out the arrays -------------------------------------------
prices = est.pdf_.prices if hasattr(est.pdf_, "prices") else est.result.prices
pdf    = est.pdf_
cdf    = est.cdf_

# --- 3. plot -----------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(8, 4))

# PDF on the left y-axis
ax1.plot(prices, pdf, color="tab:blue", label="PDF")
ax1.set_xlabel("Price at expiry")
ax1.set_ylabel("Density", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# CDF on the right y-axis
ax2 = ax1.twinx()
ax2.plot(prices, cdf, color="tab:orange", linestyle="--", label="CDF")
ax2.set_ylabel("Cumulative probability", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(0, 1)  # CDF range

# A legend that merges both lines
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")

plt.title("Risk-neutral PDF & CDF at T+30 days")
plt.tight_layout()
plt.show()
