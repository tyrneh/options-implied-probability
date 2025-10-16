

Below is a focused code + theory review of your SVI/SSVI stack. I read the modules you shared (`surface.py`, `ssvi_surface.py`, `raw_svi_surface.py`, `svi.py`, `svi_types.py`, `vol_model.py`, `estimator.py`, `ssvi.py`). I’ll call out the precise places in the code where behavior diverges from Gatheral–Jacquier and from practical calibration practice, explain why that hurts the fit against market bid/ask, and show concrete code-level fixes.

> **References used**
> 
> -   Gatheral & Jacquier (2012/2013), *Arbitrage‑free SVI volatility surfaces* (incl. raw SVI (3.1), g‑function (2.1); SSVI and its no‑arbitrage conditions).
>     
> -   Lee (2004), *Moment formula for implied vol at extreme strikes* (wing slope ≤ 2).
>     
> -   Your own package docs (data preprocessing, Black‑76 in forward space, parity, OTM‑only cleaning, current roadmap item on forward inference). TECHNICAL\_README README
>     

---

## Why your SSVI and raw‑SVI fits miss the market

### 1) **ρ is hard‑capped too tightly in SSVI (|ρ| ≤ 0.95)**

-   **Where in code:** `ssvi.py` defines `RHO_BOUND: float = 0.95`. This is then used in the unconstrained→constrained parameter map inside `ssvi_surface.py` to cap ρ.
    
-   **Why this degrades the fit:** In the SSVI surface
    
    $$
    w(k,\theta)=\tfrac{\theta}{2}\Big(1+\rho\,\phi(\theta)\,k + \sqrt{(\phi(\theta)k+\rho)^2 + 1-\rho^2}\Big),
    $$
    
    with $\phi(\theta)=\eta \theta^{-\gamma}(1+\theta)^{\gamma-1}$ (Eq. 4.5), the ATM skew and both wing slopes scale with ρ; equity smiles commonly need $|\rho|$ very close to 1. Over‑restricting to 0.95 prevents the model from achieving market skews/wings within bid/ask even when no‑arbitrage conditions are otherwise met (conditions summarized in Thm. 4.1–4.2 / Cor. 4.1). The theory only requires $|\rho|<1$ together with bounds on $\theta\phi(\theta)$ and its derivative, not $|\rho|\le 0.95$.
    
-   **Fix:** Raise the cap to ~0.999 and continue to enforce the SSVI inequalities through your “margin” checks:
    
    ```python
    # ssvi.py
    RHO_BOUND: float = 0.999  # allow realistic skews/wings; constraints still enforced
    ```
    

---

### 2) **Bid/ask envelope is *not* penalized by default in slice fits**

-   **Where in code:** `svi_types.py` defaults `envelope_weight = 0.0`. In `svi.py` a hinge penalty `_bid_ask_penalty()` exists, but with weight 0.0 it’s effectively disabled in the objective.
    
-   **Why this degrades the fit:** With no penalty, the optimizer freely leaves the market envelope when doing a Huber‑loss fit in total variance/IV. That is precisely the symptom you described—model IV sitting outside bid/ask even when the raw data carry a tight spread. (Your docs also say you default to mid pricing; without an envelope penalty the optimizer has no incentive to respect the microstructure bounds.) TECHNICAL\_README
    
-   **Fix (minimal):** turn on the envelope penalty and scale it relative to the typical IV spread for the slice.
    
    ```python
    # svi_types.py  (set a live default)
    @dataclass(frozen=True)
    class SVICalibrationOptions:
        ...
        envelope_weight: float = 25.0  # > 0 to activate hinge penalty
    
    # Example: when calling calibrate, let power users override per slice:
    opts = SVICalibrationOptions(envelope_weight=50.0)
    params, diag = calibrate_svi_parameters(k, w_obs, T, opts, bid_iv=bid, ask_iv=ask)
    ```
    
    *Rationale:* The hinge form you already coded penalizes violations only when the model leaves the envelope; keeping `envelope_weight` in the 10–100 range is enough to pull the fit inside the quoted bounds while still letting the optimizer use the rest of the loss to follow the smile.
    

---

### 3) **Forward $F_t$ inference looks to rely on a single near‑ATM pair**

-   **Where in code:** In `surface.py::_prepare_single_slice` you call `apply_put_call_parity(...)` and then pick a single `parity_forward` to compute log‑moneyness $k=\ln(K/F_t)$. Your own technical README lists as a roadmap item “infer forward using a band of near‑ATM option-pairs, rather than the one nearest pair” — indicating the current implementation uses “the one nearest pair.” This is a known source of jitter across maturities. TECHNICAL\_README
    
-   **Why this degrades the fit:** SVI/SSVI are *forward‑space* models. If $F_t$ is noisy by a few ticks, every strike’s $k=\ln(K/F_t)$ is mismeasured, shifting the entire smile and distorting the ATM variance $\theta_t$ that SSVI is trying to fit. This is a frequent root cause of “fits that are off from bid/ask” despite otherwise correct code. Gatheral’s calendar condition is $\partial_t w(k,t)\ge 0$ (monotone total variance); jittery $F_t$ breaks monotonicity and forces the optimizer to trade off calendar penalties vs. fitting the quotes.
    
-   **Fix:** replace the single‑pair parity inference with a *band regression* over many near‑ATM call/put pairs (weight by liquidity or 1/spread). Sketch:
    
    ```python
    # surface.py (inside _prepare_single_slice or a helper)
    # 1) select near-ATM band, e.g. |log(K/spot)| < 5% or top-N strikes by vega
    band = parity_adjusted.loc[parity_adjusted['moneyness_abs'] < 0.05].copy()
    
    # 2) compute theoretical parity residual for a candidate F:
    #    C - P - DF*(F - K)  ≈ 0  -> minimize in least-squares across the band
    def objective(F):
        df = np.exp(-resolved.risk_free_rate * maturity_years)
        resid = band['call_price'] - band['put_price'] - df*(F - band['strike'])
        w = 1.0 / np.maximum(band['ask'] - band['bid'], 1e-4)  # tighter quotes weigh more
        return np.sum(w * resid**2)
    
    F_hat = scipy.optimize.brent(objective, brack=(0.5*spot*e_rt, 2.0*spot*e_rt))
    # Use F_hat instead of a single-pair parity_forward
    ```
    
    This stabilizes $k$, improves the raw‑SVI slice fits, and makes SSVI calibration face a smoother $\theta_t$ term structure.
    

---

### 4) **Constraint checks use a narrow, fixed k‑grid; wings are under‑controlled**

-   **Where in code:** In both SSVI and raw‑SVI you evaluate diagnostics on `k_grid = np.linspace(-2.5, 2.5, 61)` (see `ssvi.py` and used in `ssvi_surface.py`); the raw‑SVI slice `g(k)` butterfly check is also computed on a fixed grid.
    
-   **Why this degrades the fit:** Gatheral’s butterfly condition is $g(k)\ge 0$ with $g$ given in (2.1), and Lee’s bound caps the asymptotic slope at 2. If you only check $|k|\le 2.5$ regardless of the observed range, the optimizer can “cheat” in the wings (where market quotes may still exist) without being penalized; that often manifests as model IV leaving the bid/ask on deep OTM calls/puts.
    
-   **Fix:** build the diagnostic grid from the **observed** log‑moneyness and pad it by a safety margin, and optionally extend to where Lee’s slope would bite:
    
    ```python
    # ssvi_surface.py / raw_svi_surface.py
    k_min, k_max = float(np.min(obs.log_moneyness)), float(np.max(obs.log_moneyness))
    pad = 0.5  # or proportional to sqrt(theta)
    k_grid = np.linspace(k_min - pad, k_max + pad, 401)
    
    # Optional: cap outer evaluation such that model's wing slope stays < 2 (Lee)
    # i.e., ensure theta*phi(theta)*(1+|rho|) <= 4 is checked at the calibrated theta
    ```
    
    This prevents “good looking” fits centrally that violate the envelope in the traded wings.
    

---

### 5) **Raw‑SVI surface is ‘stitched’ post‑hoc; calendar arbitrage not pushed *during* the optimization**

-   **Where in code:** `raw_svi_surface.py` calibrates each slice, then at the end computes pairwise calendar margins and throws if `vol_model.strict_no_arbitrage and calendar_margin < -1e-3`. There is no evidence the *calendar* constraints (monotone total variance in t at fixed k) are part of the objective while fitting each slice.
    
-   **Why this degrades the fit:** Raw‑SVI is notorious: slice‑by‑slice local fits can land in mutually incompatible minima. Checking calendar after the fact means the optimizer never “feels” the cross‑maturity monotonicity requirement $\partial_t w(k,t)\ge 0$; it will often push the slice to a position that matches local data but forces adjacent slices (and SSVI later) to miss the envelope. Gatheral’s necessary/sufficient calendar condition is exactly monotonicity of total variance (Lemma 2.1 / Thm. 4.1), so it should appear **inside** the optimization.
    
-   **Fix:** build a *global* raw‑SVI objective over all expiries or, at minimum, add a **pairwise calendar hinge penalty** while optimizing each slice so it aligns with neighbors:
    
    ```python
    # raw_svi_surface.py (sketch)
    def calendar_margin(slice_i_params, slice_j_params, k_grid):
        wi = svi_total_variance(k_grid, slice_i_params)
        wj = svi_total_variance(k_grid, slice_j_params)
        return np.min(wj - wi)  # should be >= 0
    
    # During per-slice optimization include:
    penalty = 0.0
    for nb in neighbors:  # previous & next maturities
        m = calendar_margin(params_candidate, params_nb, k_grid)
        penalty += lambda_cal * np.maximum(0.0, -m)**2  # hinge^2
    objective = base_slice_loss + penalty
    ```
    
    The robust, production‑grade alternative is to **fit SSVI first** (ρ, η, γ, θ(t)) and then project each maturity to the closest raw‑SVI slice in total‑variance norm. That projection preserves static no‑arbitrage by construction (SSVI) and gives you stable raw‑SVI parameters for downstream analytics.
    

---

## Additional “near‑misses” worth fixing

-   **Huber δ selection & weight model.** Your `svi.py` uses a Huber loss, but δ defaults to `None` (then inferred) and weights are **vega‑based only**. With tight quotes, make δ proportional to the *median* bid‑ask IV width and **scale weights by 1/(bid‑ask width)** when available—this aligns the objective with the actual measurement error. (You already pass `bid_iv`/`ask_iv`; use that for weights as well as for the envelope penalty.)
    
-   **θ grid for SSVI derivative condition.** Your `compute_ssvi_margins` checks $\partial_\theta(\theta\phi(\theta))$ but only at discrete θ values from the data. Build a monotone interpolator (you already do: PCHIP in `SSVISurfaceParams.interpolator()`) and evaluate the derivative bound on a denser θ grid; Theorem 4.1 puts an *upper* bound of $\frac{1}{\rho^2}(1+\sqrt{1-\rho^2})\phi(\theta)$ and requires non‑negativity.
    

---

## Patches / concrete code edits

### A) Loosen the ρ bound (SSVI)

```python
# ssvi.py
RHO_BOUND: float = 0.999  # (was 0.95); theory requires |rho|<1, not 0.95.
```

No‑arbitrage remains enforced through your `theta*phi` margins (Thm. 4.2 / Cor. 4.1).

---

### B) Activate the bid/ask envelope penalty & measurement‑error weighting

```python
# svi_types.py
@dataclass(frozen=True)
class SVICalibrationOptions:
    ...
    envelope_weight: float = 25.0             # >0 activates your existing hinge penalty
    weighting_mode: str = "vega+spread"       # new option

# svi.py
def _vega_based_weights(..., mode: str, ...):
    ...
    if mode.lower() in {"vega+spread", "spread"} and (bid_iv is not None and ask_iv is not None):
        spread = np.maximum(np.asarray(ask_iv) - np.asarray(bid_iv), 1e-4)
        w_me = 1.0 / spread                    # measurement-error weights
        if mode.lower() == "vega+spread":
            weights = weights * w_me           # combine with vega
        else:
            weights = w_me
    ...
    return weights / np.clip(weights.max(), 1.0, np.inf), True
```

Usage:

```python
opts = SVICalibrationOptions(envelope_weight=50.0, weighting_mode="vega+spread")
params, diag = calibrate_svi_parameters(k, w_obs, T, opts, bid_iv=bid, ask_iv=ask)
```

---

### C) Robust forward inference (stabilize k)

```python
# surface.py  (inside _prepare_single_slice)
def infer_forward_from_band(parity_df, r, T):
    # choose band near ATM by |log(K/spot)| or top-N vega
    df = parity_df.copy()
    df = df[np.isfinite(df['call_price']) & np.isfinite(df['put_price'])]
    df['width'] = np.maximum(df['ask'] - df['bid'], 1e-4)
    band = df.nsmallest(15, 'width')  # tightest 15 quotes
    disc = np.exp(-r*T)
    def obj(F):
        resid = band['call_price'] - band['put_price'] - disc*(F - band['strike'])
        return float(np.sum((resid**2) / band['width']))
    # bracket around spot-forward
    F0 = float(self._forward)  # or S*exp((r-q)T) if you have q
    result = scipy.optimize.minimize_scalar(obj, bounds=(0.5*F0, 2.0*F0), method="bounded")
    return float(result.x)

parity_forward = infer_forward_from_band(parity_adjusted, resolved.risk_free_rate, maturity_years)
```

This addresses the roadmap item you already identified. TECHNICAL\_README

---

### D) Expand diagnostic grids based on observed k

```python
# ssvi_surface.py and raw_svi_surface.py
k_obs = obs.log_moneyness
k_min, k_max = float(np.min(k_obs)), float(np.max(k_obs))
pad = 0.5
k_grid = np.linspace(k_min - pad, k_max + pad, 401)

# feed k_grid into compute_ssvi_margins / g(k) checks
```

The stronger check in the traded wings helps keep fits inside bid/ask where it matters. Conditions use g(k)≥0 for butterfly and Lee’s slope implicitly via Corollary 4.1 (θφ(θ)(1+|ρ|) ≤ 4).

---

### E) Put calendar into the raw‑SVI objective

```python
# raw_svi_surface.py  (conceptual)
lambda_cal = 1e3  # tune

def slice_objective(params_vec, neighbors):
    # base Huber + weights residual (existing)
    base, grad = base_slice_loss(params_vec, ...)
    # calendar penalty vs. neighbors (same k_grid for all)
    cal_pen = 0.0
    for nb in neighbors:
        m = np.min(svi_total_variance(k_grid, params_vec) - svi_total_variance(k_grid, nb))
        cal_pen += np.maximum(0.0, -m)**2
    return base + lambda_cal*cal_pen
```

By pushing the monotonicity condition ($\partial_t w \ge 0$) *during* optimization you avoid the post‑hoc “min margin < 0” failures and reduce cross‑maturity tug‑of‑war.

---

## Why these fixes line up with the paper

-   **SSVI constraints:** Your code already uses the $\phi(\theta)=\eta\theta^{-\gamma}(1+\theta)^{\gamma-1}$ form (Eq. 4.5) and checks $\theta\phi(\theta)(1+|\rho|)\le 4$ & derivative bounds. The above changes (ρ bound, denser θ checks) simply let the optimizer explore the **full allowed** $|\rho|<1$ range while continuing to enforce Theorem 4.1–4.2 / Corollary 4.1.
    
-   **Raw‑SVI:** You’re using the raw parameterization  
    $w(k)=a+b\{\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\}$ (Eq. 3.1). The no‑arbitrage characterisation is expressed via the g‑function and calendar monotonicity; moving those constraints *into* the objective is exactly what Gatheral & Jacquier recommend to achieve a usable surface (Sec. 5).
    
-   **Wings:** Lee’s bound explains why wing slopes must be controlled; your SSVI margins already tie to it via $\theta\phi(\theta)(1+|\rho|)\le 4$. Extending the grids ensures you enforce the bound where the market is actually quoted.
    

---

## Quick checklist to re‑run after edits

1.  **Per slice (raw‑SVI):** check diagnostics: `min_g ≥ 0`, `envelope_violations_pct ≈ 0%`, residuals scale like bid/ask width (use your new weighting).
    
2.  **Across slices:** `min_calendar_margin ≥ 0` on your union‑of‑k grid; SVI‑JW wing slopes monotone with t.
    
3.  **SSVI:** `min_theta_phi_margin ≥ 0`, `min_theta_phi_sq_margin ≥ 0`, `min_derivative_margin ≥ 0`; fitted ρ now near market values (often −0.7 … −0.99 in equity).
    
4.  **Visuals:** model IV curves lie inside the bid/ask shadings at each maturity; wings don’t flare out.
    

---

If you implement the five changes above—(i) loosen ρ, (ii) activate envelope penalties and measurement‑error weights, (iii) robustify forward inference, (iv) evaluate constraints on adaptive k‑grids, and (v) push calendar into the raw‑SVI objective—you’ll find the SSVI surface fits within bid/ask and the raw‑SVI slices stop “fighting” each other across maturities. These are fully consistent with the mathematical conditions in the paper and with the data‑handling in your own docs (forward‑space pricing, OTM parity, etc.).

If you want, I can turn the sketches above into PR‑ready diffs for `ssvi.py`, `svi_types.py`, `svi.py`, `surface.py`, and `raw_svi_surface.py` that preserve your current public API.