from oipd import RND, MarketInputs, ModelParams, RNDSurface, VolModel
from oipd.core.vol_surface_fitting.shared.svi import check_butterfly
from oipd.core.vol_surface_fitting.shared.ssvi import (
    check_ssvi_constraints,
    check_ssvi_calendar,
)

import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Surface example (NVDA term structure)
# ---------------------------------------------------------------------------
market = MarketInputs(
    valuation_date=date.today(),  # date pulled
    risk_free_rate=0.04,
)
model = ModelParams(price_method="mid", max_staleness_days=1)


surface_nvda = RNDSurface.from_ticker(
    "NVDA",
    market,
    model=model,
    horizon="1Y",
    vol=VolModel(method="ssvi"),
)

fig = surface_nvda.plot_iv(layout="grid")
plt.show()

fig = surface_nvda.plot_iv_3d()
fig.show()

surface_nvda.plot()
plt.show()

# # Interactive 3D surface (requires Plotly)
# fig_3d = surface_nvda.plot_iv_3d()
# # fig_3d.show()  # Uncomment in interactive environments

# ssvi_diag = surface_nvda.check_no_arbitrage()
# print("SSVI diagnostics:")
# print(f"  objective           : {ssvi_diag['objective']:.4e}")
# print(f"  min calendar margin : {ssvi_diag['min_calendar_margin']:.4e}")
# print(f"  min θφ margin       : {ssvi_diag['min_theta_phi_margin']:.4e}")

# # Optional: validate the fitted parameters on a custom grid
# constraints = check_ssvi_constraints(
#     surface_nvda._ssvi_fit.params.theta,  # type: ignore[attr-defined]
#     surface_nvda._ssvi_fit.params.rho,  # type: ignore[attr-defined]
#     surface_nvda._ssvi_fit.params.eta,  # type: ignore[attr-defined]
#     surface_nvda._ssvi_fit.params.gamma,  # type: ignore[attr-defined]
# )
# calendar_chk = check_ssvi_calendar(
#     surface_nvda._ssvi_fit.params.theta,  # type: ignore[attr-defined]
#     surface_nvda._ssvi_fit.params.rho,  # type: ignore[attr-defined]
#     surface_nvda._ssvi_fit.params.eta,  # type: ignore[attr-defined]
#     surface_nvda._ssvi_fit.params.gamma,  # type: ignore[attr-defined]
#     k_grid=np.linspace(-2.0, 2.0, 51),
# )
# print(f"  validator min θφ margin : {constraints['min_theta_phi_margin']:.4e}")
# print(f"  validator min calendar  : {calendar_chk['min_margin']:.4e}")

# # Build a raw-SVI surface on the same horizon to showcase the α-tilt diagnostics
# surface_nvda_raw = RNDSurface.from_ticker(
#     "NVDA",
#     market,
#     model=model,
#     horizon="1M",
# )
# raw_diag = surface_nvda_raw.check_no_arbitrage()
# print("Raw SVI diagnostics:")
# print(f"  objective           : {raw_diag['objective']:.4e}")
# print(f"  min calendar margin : {raw_diag['min_calendar_margin']:.4e}")
# print(f"  alpha tilt applied  : {raw_diag['alpha']:.4e}")

# # Inspect butterfly margins on the first raw slice
# # NOTE: accessing protected fit objects for diagnostics; this mirrors what power users
# # typically do when building bespoke reporting dashboards.
# first_slice = surface_nvda_raw._ensure_raw_fit().slices[0]  # type: ignore[attr-defined]
# butterfly_diag = check_butterfly(first_slice.params, np.linspace(-0.8, 0.8, 61))
# print(f"  raw slice min g(k)  : {butterfly_diag['min_margin']:.4e}")

# # --- smile example --- #

# # --- Example 1 - GME --- #
# # --- using yfinance connection --- #

# # 1. Get a list of available expiry dates
# expiry_dates = RND.list_expiry_dates("GME")
# print(expiry_dates[:])  # ['2025-09-05', '2025-09-12', '2025-09-19',...]

# # 2. Use ticker data with market parameters (current price fetched automatically)
# market = MarketInputs(
#     valuation_date=date.today(),
#     expiry_date=date(2025, 10, 17),
#     risk_free_rate=0.04199,  # US 3-month nominal Treasury yield
# )

# model_gme = ModelParams(price_method="mid", max_staleness_days=None)

# # 3. Fetch and estimate - auto-fetched data is available in the result
# est_gamestop = RND.from_ticker("NVDA", market, model=model_gme)

# # 4. Check the final market parameters used in estimation
# est_gamestop.market

# # 5. Plot using the result object
# est_gamestop.plot_iv(x_axis="strike")
# plt.show()
