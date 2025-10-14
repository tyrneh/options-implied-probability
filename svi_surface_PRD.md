

Below I do three things:

1.  **Audit** your current SVI implementation against Gatheral–Jacquier (2012).
    
2.  **Explain the theory simply** and then give a **step‑by‑step implementation guide** to extend from a single SVI slice to a full, arbitrage‑free SVI/SSVI surface across maturities.
    
3.  Propose a **clean API** for term‑structure functionality that fits naturally into your package.
    

---

## (1) Audit — does your code line up with Gatheral–Jacquier (2012)?

**What I reviewed (core files):** `svi.py`, `svi_types.py`, `surface_fitting.py`, `prep.py`, `parity.py`, `estimator.py`, `market_inputs.py`.

### A. Single‑slice SVI model (theory vs code) — ✔️ good alignment

**Theory.** A raw SVI slice parameterizes *total* variance  
$w(k)=a+b\{\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\}$ with $b\ge0$, $|\rho|<1$, $\sigma>0$ (their Eq. (3.1)). A slice is free of butterfly arbitrage iff the “$g(k)$” function is non‑negative everywhere (their Eq. (2.1) and Lemma 2.2). [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)

**Your code.**

-   **Model & constraints.** `SVIParameters` implements the raw form and enforces $b\ge 0$, $|\rho|<1$, $\sigma>0$. ✅ (matches Eq. (3.1)). [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
-   **Derivatives.** You compute $w, w', w''$ analytically for the raw form (needed for diagnostics, gradients, and penalties). ✅
    
-   **Butterfly condition.** Your `g_function(k, params)` implements  
    $g(k)=\Big(1-\frac{k w'}{2w}\Big)^2-\frac{(w')^2}{4}\big(\frac1w+\frac14\big)+\frac12 w''$ and you penalize $g(k)<0$ on a diagnostic grid during calibration. ✅ (matches Eq. (2.1) & Lemma 2.2). [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    
-   **Parameter systems.** You support conversions between raw SVI and JW “jump‑wings” parameters (atm variance $v$, atm skew $\psi$, wing slopes $p,c$, $v_{\min}$), which are the paper’s trader‑friendly parameters (Section 3.3 with Eq. (3.5) and Lemma 3.2). ✅ [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
-   **Objective.** The calibrator uses a robust Huber loss on total variance residuals, optional **vega‑based weighting**, and three meaningful regularizers/penalties:
    
    -   **Butterfly penalty** via $g(k)$ (density non‑negativity) ✅
        
    -   **Call‑spread monotonicity penalty** (ensures call prices decrease with strike; related to convexity/monotonicity of call prices) ✅ — consistent with the paper’s focus that absence of butterfly arbitrage corresponds to convex call prices. [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
        
    -   **Bid/ask envelope penalty** to keep model IV within quotes ✅ (pragmatic and consistent with the paper’s calibration section §5).
        
-   **Seeds.** You implement a quasi‑explicit split seeding heuristic and random restarts; then L‑BFGS‑B (and a fallback). ✅ (the paper also advocates good initial guesses and a practical penalty‑based fit in §5).
    

> **Verdict for a single expiry:** Your implementation closely follows the theory and the **spirit** of Gatheral–Jacquier for **slice calibration with no butterfly arbitrage**. I don’t see conceptual divergences here.

### B. Pre‑processing and pricing conventions — ✔️ aligned with your package docs

-   **Put–call parity / forward inference / OTM‑only filtering.** Your `parity.py` + `prep.py` infer the forward from near‑ATM call–put pairs, keep OTM options, and synthesize calls when needed — exactly the robust preprocessing advocated in the literature you cite in your own docs. ✅ TECHNICAL\_README
    
-   **Forward‑based pricing.** The package works naturally in forward space (Black‑76), and converts prices↔IVs consistently. ✅ TECHNICAL\_README
    

### C. What’s **not** there yet (to match the paper’s *surface* results)

The 2012 paper is about **static‑arbitrage‑free *surfaces***, not only slices. Two ingredients are missing:

1.  **Calendar‑spread arbitrage control** — globally require $\partial_t w(k,t)\ge 0$ for all $k$ (Lemma 2.1 / Definition 2.2). Your calibrator currently has no cross‑maturity term that prevents two slices from crossing each other in total variance, nor a “crossedness” penalty from §5.2 of the paper. ✖️ [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
2.  **SSVI (Surface‑SVI) parameterization** —  
    $w(k,\theta_t)=\frac{\theta_t}{2}\left\{1+\rho\,\phi(\theta_t)\,k+\sqrt{(\phi(\theta_t)\,k+\rho)^2+(1-\rho^2)}\right\}$  
    with explicit **no‑arbitrage** conditions across **all** maturities (Theorem 4.1 & 4.2; Corollary 4.1). Your code fits slices, but it doesn’t yet model a global $\theta_t,\rho,\phi(\cdot)$ nor enforce the SSVI inequalities. ✖️ [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

Your own roadmap notes “full term‑structure surface (`RNDTermSurface`)” as a future step — this audit confirms that’s the right next milestone. TECHNICAL\_README

---

## (2) Extend to a full surface — high‑level theory first, then a step‑by‑step plan

### High‑level picture (in plain terms)

-   A **slice** fit tells you one maturity’s smile with no butterfly arbitrage.
    
-   A **surface** stitches all slices together so that **as time increases**, the **total variance never goes down** for any strike (no calendar arbitrage), **and** each slice keeps non‑negative density (no butterfly arbitrage).
    
-   **SSVI** is a clever way to write the whole surface with just a few **functions of time** (ATM total variance $\theta_t$, a correlation‑like $\rho$, and a skew‑shape $\phi(\theta)$). If you pick $\theta_t,\rho,\phi$ so they satisfy a handful of inequalities, you **guarantee** no static arbitrage everywhere on the surface. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

The key, practical sufficient conditions (Corollary 4.1) are:

-   $\partial_t \theta_t \ge 0$ (ATM total variance must be non‑decreasing in $t$);
    
-   $0\ \le\ \partial_\theta(\theta\,\phi(\theta))\ \le\ \frac{1}{\rho^2}\!\big(1+\sqrt{1-\rho^2}\big)\,\phi(\theta)$ (controls calendar arbitrage for the whole smile);
    
-   $\theta\,\phi(\theta)\,(1+|\rho|) < 4$ and $\theta\,\phi(\theta)^2\,(1+|\rho|)\le 4$ (wing‑slope bounds that ensure no butterfly arbitrage). [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

A widely used choice that satisfies these with simple parameter bounds is

$$
\phi(\theta)\;=\; \eta\,\theta^{-\gamma}\,(1+\theta)^{\gamma-1}, \quad \eta>0,\; \gamma\in[0,1],
$$

often with **$\rho$** taken constant in time. Under $\eta(1+|\rho|)\le 2$, this form yields a surface free of static arbitrage. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)

### Step‑by‑step implementation guide

**Stage 0 — Data prep (you already do this).**

1.  **Infer forwards** per maturity from near‑ATM call–put pairs;  
    **OTM‑only** filter; synthesize quotes via parity when needed. TECHNICAL\_README
    
2.  Convert quotes to **log‑moneyness** $k=\log(K/F_t)$ and **total variance** $w=\sigma^2T$.
    

**Stage 1 — Calibrate each maturity slice (you already do this well).**

1.  Fit **raw SVI** parameters $(a,b,\rho,m,\sigma)$ to $w(k)$: robust loss + vega weights.
    
2.  Enforce **no butterfly** via your $g(k)\ge0$ penalty, and **monotone call prices** via your call‑spread penalty.
    
3.  From the fitted slice, compute **JW parameters** $(v,\psi,p,c,v_{\min})$ — keep these; we’ll use them to initialize the surface. (Paper §3.3.) [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    

**Stage 2 — Build an initial *surface* guess.** Two proven routes:

-   **(A) Penalty‑based slice stitching (paper §5.2).**  
    Keep each slice in raw SVI, but add a **calendar penalty** that discourages **crossings of total‑variance curves** between adjacent maturities (“crossedness” in Definition 5.1) and iterate to re‑optimize slices jointly. This gives you a surface that is *empirically* free of calendar arbitrage. [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
-   **(B) Move to SSVI (recommended).**  
    Parameterize the full surface as
    
    $$
    w(k,\theta_t)=\frac{\theta_t}{2}\!\left\{1+\rho\,\phi(\theta_t)\,k+\sqrt{(\phi(\theta_t)\,k+\rho)^2+1-\rho^2}\right\},
    $$
    
    with $\theta_t$ the per‑maturity **ATM total variance**, $\rho\in(-1,1)$, and $\phi(\cdot)>0$ as above. Calibrate $\{\theta_{t_i}\}$ on your expiries and global $(\rho,\eta,\gamma)$ by minimizing squared **price** (or IV) errors subject to the **SSVI inequalities** below. This route gives a **proof‑level** arbitrage‑free surface once the constraints are respected. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

**Stage 3 — Enforce *static no‑arbitrage* explicitly (SSVI).**  
Use **soft constraints** (penalties) or a constrained optimizer:

-   **Calendar:** $\partial_t\theta_t \ge 0$ (monotone sequence of $\theta_{t_i}$).
    
-   **SSVI slope bounds:**  
    $\theta_t\,\phi(\theta_t)\,(1+|\rho|) < 4$ and $\theta_t\,\phi(\theta_t)^2\,(1+|\rho|)\le4$ for all maturities $t_i$.
    
-   **Calendar sufficiency:**  
    $0\le \partial_\theta(\theta\phi(\theta)) \le \frac{1}{\rho^2}\big(1+\sqrt{1-\rho^2}\big)\phi(\theta)$.  
    All are straight from Theorem 4.1, Theorem 4.2 and Corollary 4.1. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

**Stage 4 — Interpolation & extrapolation (surface).**

-   **Between fitted maturities:** interpolate **$\theta_t$** *monotonically* (e.g., monotone cubic Hermite in $t$), keep $\rho$ constant (or smoothly varying), and evaluate $w$ via SSVI. Lemma 5.1 shows a way to interpolate prices while preserving no‑arbitrage. [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
-   **Beyond last maturity:** the paper suggests **adding** a non‑decreasing $\alpha(t)$ to total variance (Theorem 4.3) so $w_\alpha(k,t)=w(k,t)+\alpha t$ stays static‑arbitrage‑free. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

**Stage 5 — Diagnostics and tests.**

-   Verify on a dense grid: **(i)** $g(k,t)\ge0$ for all $k$ on each $t$; **(ii)** $w(k,t+\Delta t)\ge w(k,t)$ for all grid $k$. (Lemmas 2.1 & 2.2). [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    
-   Check **Lee’s wing slope** bound emerges automatically from the SSVI constraints (remark in §4). [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

---

## (3) API design for term structure

This section specifies the user‑facing API consistent with the theory above while adopting a clean, unified interface for both single‑expiry smiles and full term structures.

### A. User‑Facing Objects

- `RND`: single‑expiry estimator (unchanged). Accepts the same `MarketInputs` and optional `ModelParams` as today.
- `RNDSurface`: term‑structure estimator. Constructs an arbitrage‑aware surface across maturities.
- `VolModel`: unified volatility‑model selector used by both `RND` and `RNDSurface`.

```python
@dataclass(frozen=True)
class VolModel:
    method: Optional[Literal["svi", "svi-jw", "bspline", "ssvi", "raw_svi"]] = None
    strict_no_arbitrage: bool = True
```

Semantics
- When `method is None`: `RND` defaults to `"svi"`; `RNDSurface` defaults to `"ssvi"`.
- RND accepts: `"svi"` (raw SVI), `"svi-jw"` (JW parametrisation), `"bspline"` (legacy smoother).
- RNDSurface accepts: `"ssvi"` (theorem‑backed surface) or `"raw_svi"` (penalty‑stitched raw slices).
- `strict_no_arbitrage=True` enables butterfly diagnostics/penalties for slices and SSVI theorem constraints + calendar checks for surfaces.

Notes
- `"svi"` means raw SVI in `(a,b,ρ,m,σ)`; `"svi-jw"` calibrates in JW then maps to raw for evaluation/diagnostics.
- Internal objectives, weights, restarts, and penalty scales are not exposed; robust defaults follow the theory in this PRD.

### B. Constructors

Single‑expiry (unchanged entry point)
```python
from oipd import MarketInputs, ModelParams, VolModel, RND

est = RND.from_ticker(
    "AAPL",
    market,
    model=ModelParams(...),
    vol=VolModel(method=None),  # defaults to "svi"
)
```

Surface from vendor data (auto expiries)
```python
from oipd import MarketInputs, ModelParams, VolModel, RNDSurface

surface = RNDSurface.from_ticker(
    "AAPL",
    market,
    horizon="12M",                 # accepts "3M", "1Y", 180 (days), 0.5 (years), or timedelta
    model=ModelParams(...),
    vol=VolModel(method=None),     # defaults to "ssvi"
)
```

Surface from CSV/DataFrame (multi‑maturity)
```python
surface = RNDSurface.from_dataframe(
    df,
    market,
    expiry_col="expiry",          # date/datetime column required per quote row
    strike_col="strike",
    price_cols=("bid","ask")      # or ("last",)
    # even if IV columns exist, the pipeline computes IVs from prices; IVs are used for envelopes/diagnostics only
    , vol=VolModel(method="ssvi")
)
```

Behavior
- `RNDSurface.from_ticker` auto‑fetches all vendor expiries up to `valuation_date + horizon`; no `expiries/tenors/frequency` arguments are required.
- `RNDSurface.from_dataframe` requires `expiry_col` as a true date/datetime. Rows are grouped by expiry and processed through the same robust pipeline (parity → forward → OTM filtering → price→IV inversion).
- For both `RND` and `RNDSurface`, provided bid/ask IVs (if any) are used for envelope checks and diagnostics; calibration IVs are computed from prices.

### C. Evaluation & Utilities

```python
surface.total_variance(k, t)
surface.iv(K, t)
surface.price(K, F_t, t, call=True)
surface.slice(t)           # returns an RND view at maturity t
surface.check_no_arbitrage()  # {min_g_by_slice, min_calendar_diff, ssvi_margins}
```

### D. No‑Arbitrage, Interpolation, Extrapolation

- Slices (SVI): enforce butterfly via `g(k) ≥ 0` diagnostics and call‑spread guards as in §2 and §5.
- Surface (SSVI): calibrate global `ρ` and `θ(t_i)` with φ(θ)=η θ^{-γ} (1+θ)^{γ−1}. Enforce the four inequalities from Theorems 4.1/4.2 and Corollary 4.1, plus calendar monotonicity `∂_t w ≥ 0`.
- ρ across maturities: keep constant by default (per SSVI spec).
- Interpolate θ(t) monotonically between fitted maturities (monotone cubic Hermite/PCHIP). Evaluate w via SSVI.
- Extrapolation beyond last maturity: include an additive variance hook α(t) so `w_α(k,t)=w(k,t)+α t` (Theorem 4.3). Default α=0.

### E. Developer Notes

- Raw stitched alternative: `method="raw_svi"` performs joint refinement over per‑maturity raw SVI slices with a calendar “crossedness” penalty (paper §5.2) and per‑slice butterfly checks.
- Reuse the existing single‑slice SVI fitter for initial guesses; surface routines layer calendar/theorem constraints on top.
- Objectives default to total‑variance Huber with vega‑based weights; no user toggle.

### F. Examples

```python
# Evaluate
iv = surface.iv(K=[350, 450], t=0.5)
price = surface.price(K=[400], F_t=395.0, t=1.0)

# Diagnostics
diag = surface.check_no_arbitrage()

# Slice view
rnd_6m = surface.slice(t=0.5)
rnd_6m.plot_iv()
```

---

## Concrete, quick wins to close the gap

1.  **Add a calendar penalty** to your multi‑maturity fit (raw‑stitch mode): compute total‑variance intersections between adjacent fitted slices and penalize “crossedness” (paper §5.2). [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
2.  **Introduce SSVI** as an alternative **surface** model with hard/soft constraints implementing Theorems 4.1 & 4.2 (and Corollary 4.1). [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    
3.  **Ship validators**:
    
    -   `check_butterfly(params, k_grid)` → min $g(k)$ per slice;
        
    -   `check_calendar(surface, k_grid, t_grid)` → min $\Delta_t w(k,t)$;
        
    -   `check_ssvi_constraints(theta, rho, eta, gamma)` → report any inequality violations.
        

---

## Why these choices fit your package

-   They respect your **forward‑based**, parity‑aware, **robust** calibration pipeline (already documented in your Technical README). TECHNICAL\_README
    
-   They let you keep the current **single‑expiry** `RND` API intact, while exposing a new `RNDSurface`/`SSVISurface` for users who need the full term structure.
    
-   They are **theorem‑level** safe: once SSVI constraints hold, **static arbitrage is ruled out by construction** (not just “unlikely”). [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

---

### Appendix — References mapped to the code

-   **Raw SVI** slice $w(k)$, parameter constraints — Eq. (3.1). Your `SVIParameters` and evaluators match. [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
-   **Butterfly‑free test** $g(k)\ge0$ — Eq. (2.1) & Lemma 2.2. Your `g_function`/penalty match. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    
-   **SVI‑JW** mapping & interpretations — §3.3. Your raw↔JW helpers match. [ar5iv](https://ar5iv.labs.arxiv.org/html/1204.0646)
    
-   **Calendar‑free** requirement — $\partial_t w(k,t)\ge 0$. Needs surface‑level enforcement. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    
-   **SSVI surface** and **no‑arbitrage conditions** — Eq. (4.1), Theorems 4.1 & 4.2, Corollary 4.1. Recommend adding. [Imperial College London](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/stochastic-analysis-group/preprints-2012/12-11.pdf)
    

---

If you’d like, I can sketch pseudo‑code for `SSVISurface.fit` and a first pass at the constraint functions; but at a high level, the checklist above gives you all the pieces to move from robust **slice** fits (which you already have) to a provably **arbitrage‑free surface** that conforms to Gatheral–Jacquier’s theory.
