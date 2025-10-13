
## Executive highlights

-   **What’s strong**
    
    -   A modern, **slice-level SVI** engine with: Huberised least‑squares in **total variance**, **vega×volume weighting**, **bid/ask envelope**, **butterfly** + **call‑spread** penalties, and a **global (DE) → local (L‑BFGS‑B / Nelder) pipeline**.
        
    -   First‑class **JW mapping**: `raw_to_jw()` and `jw_to_raw()` are implemented, and JW parameters are surfaced on the returned `VolCurve`.
        
    -   Data preprocessing is aligned with the PRD: forward inference via parity, OTM‑only filtering, calibration in forward log‑moneyness. svi\_jw\_calibration\_prd
        
-   **What’s missing**
    
    -   No **across‑maturity** construction (no SSVI or price‑space stitching), so there’s **no calendar‑arbitrage guarantee** yet.
        
    -   Some knobs are static rather than **data/tenor‑adaptive** (call‑spread step, Huber δ).
        
    -   Options/diagnostics are dictionaries rather than typed containers; logging goes through `warnings` instead of structured logging.
        
    -   Docs and API focus on “RND from a smile” but do not yet expose a term‑surface consistent with the PRD’s goals. README TECHNICAL\_README
        

---

## 1) Theoretical correctness

### ✅ Aligned with PRD (per‑maturity)

-   **SVI model & JW mapping.**
    
    -   Raw SVI is implemented with standard constraints.
        
    -   **JW ⇄ raw** mappings are present; `jw_to_raw` guards boundary pathologies by clipping $\rho$ and $k=\rho-\psi/b$ to $(1-10^{-6})$ and checks positivity of the auxiliary $s=\sqrt{m^2+\sigma^2}$.
        
    -   **Recommendation:** add a **round‑trip unit test** to assert `raw_to_jw(jw_to_raw(.))≈id` and vice‑versa over random, arbitrage‑free draws.
        
-   **Objective & penalties.**
    
    -   **Huberised LS** on **total variance** residuals (robust to wing outliers).
        
    -   **Weights**: Black‑76 **vega** with optional **volume multiply**, normalised and clipped—this is exactly the PRD’s guidance to emphasise liquid strikes.
        
    -   **Bid/ask envelope penalty**: hinge penalty in IV space keeps the fit inside quoted spreads.
        
    -   **Arbitrage checks**: the **butterfly diagnostic** $g(k)$ is implemented in its canonical (Gatheral–Jacquier) form; **call‑spread monotonicity** enforced in price space using B76 with $F=1$.
        
-   **Optimisation.**
    
    -   Global **Differential Evolution** (seeded) → **multi‑start** → **polish** with L‑BFGS‑B / Nelder. Good convergence hygiene and deterministic seeds.
        
-   **Bounds/initialisation.**
    
    -   **Data‑adaptive** bounds for $m$, tenor‑aware `b_upper`, and reasonable `sigma` limits.
        
    -   Initial guess uses argmin of $w$ for $m$, local slopes for $(b,\rho)$, and stdev of $k$ for $\sigma$ — sensible.
        

### ⚠️ Gaps / improvements

-   **Huber scale is absolute.** `huber_delta=1e‑3` in *total variance* units may be too tight/loose depending on tenor and underlying.  
    **Fix:** set `delta = max(delta_floor, beta * median(w))`, with `beta≈1%`; add to options.
    
-   **Call‑spread step is fixed.** `0.05` in log‑moneyness can be too coarse (long tenors) or too fine (short).  
    **Fix:** make step **adaptive**—`0.5 * median(diff(k))` bounded away from zero, with tenor scaling.
    
-   **QE split seed (outer $(m,\sigma)$, inner LS for $(a,b,\rho)$)** is not implemented.  
    **Benefit:** faster/better seeds, often a material runtime + stability win on skewed/short‑dated slices. svi\_jw\_calibration\_prd
    
-   **Across maturities is out of scope today.**  
    The PRD calls for **SSVI** or **call‑price interpolation** across $T$ to eliminate calendar arbitrage.  
    **Plan:** new `RNDTermSurface`: fit slices, then **monotone call‑price interpolation in $T$** (or SSVI with Gatheral–Jacquier conditions) to enforce no calendar arbitrage. svi\_jw\_calibration\_prd
    

---

## 2) Software engineering & architecture

### ✅ What’s clean

-   Modular decomposition matches the user docs’ flow: `prep.py` (parity), `iv.py` (solvers), `surface_fitting.py` (dispatch), `svi.py` (math & optimisation), `estimator.py` (facade). Diagnostics from calibration are rich and useful (RMSE weighted/unweighted, min‑g, envelope breach %, optimiser lineage). TECHNICAL\_README
    

### 🛠 Refactor proposals (no behavioural change unless noted)

1.  **Typed configuration**  
    Replace `DEFAULT_SVI_OPTIONS`/`merge_svi_options` with `@dataclass SVICalibrationOptions` (or `TypedDict`): static typing, IDE support, and early validation (reject unknown keys).
    
2.  **Typed outputs**  
    Return `SVIParams(raw: SVIParameters, jw: JWParams)` and `SVIFitDiagnostics` dataclasses instead of ad‑hoc dicts on `VolCurve`.
    
3.  **Logging**  
    Use `logging.getLogger("oipd.svi")`; keep `INFO` for convergence, `WARNING` when any constraint breach occurs (e.g., `min_g < 0`), and surface RNG seed/SciPy version.
    
4.  **Determinism**  
    Thread `random_seed` from high‑level `RND` into SVI and persist it in diagnostics for reproducibility audits.
    
5.  **Adaptive defaults**
    
    -   Make Huber `delta` **scale‑aware** (see above).
        
    -   Make call‑spread `step` **adaptive** and tenor‑aware.
        
    -   Consider scaling envelope penalty by `mean_iv` to keep the objective well‑conditioned across names/tenors.
        
6.  **QE split seed**  
    Optional, but recommended for short‑dated, skewed, or sparse smiles to cut tail failures and polish iterations. (Adds no change to the optimality conditions.)
    
7.  **Docs & examples**  
    Add an “SVI‑JW mode” example that prints diagnostics (min‑g, envelope %, RMSEs) and explains interpretation. Update the README to mention JW/SSVI roadmap. README
    


---