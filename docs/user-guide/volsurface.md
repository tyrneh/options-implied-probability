---
title: The VolSurface Interface
parent: User Guide
nav_order: 2
---

## 3.2. The `VolSurface` Interface

The `VolSurface` class allows you to work with multiple expiries at once. It takes the same configuration parameters as `VolCurve`.

```python
# 1. Fetch data for multiple expiries (this is a simplified example)
# In a real scenario, you would concatenate chains for different expiries.
multi_expiry_df = pd.concat([
    ticker.option_chain(expiry.strftime("%Y-%m-%d")).calls,
    ticker.option_chain(next_expiry.strftime("%Y-%m-%d")).calls
])


# 2. Fit the surface
vol_surface = oipd.VolSurface(expiry_column='expiry').fit(multi_expiry_df, market)

# 3. Get the distribution surface
dist_surface = vol_surface.implied_distribution()

# 4. Slice the surface for a specific expiry
expiry_to_get = pd.to_datetime(expiry)
single_dist = dist_surface.slice(expiry_to_get)
```