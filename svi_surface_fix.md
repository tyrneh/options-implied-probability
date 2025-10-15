

Below I’ll do three things:

1.  explain Gatheral–Jacquier (2012) at a first‑principles level and the practical recipe it prescribes for fitting an arbitrage‑free IV surface,
    
2.  review your repo module‑by‑module against the intent of the paper (what is aligned vs. misaligned), and
    
3.  diagnose why *both* `VolModel(method="ssvi")` and `VolModel(method="raw_svi")` fail in your run, and give concrete fixes.
    

---

## (1) Gatheral’s paper, simply and from first principles

**What are we fitting?**  
For maturity $t$, define log‑moneyness relative to the forward $F_t$:

$$
k=\log\!\left(\frac{K}{F_t}\right).
$$

Work in **total variance**

$$
w(k,t)=\sigma^2_{\text{BS}}(k,t)\,t,
$$

not in volatility. Static no‑arbitrage on a surface means **both**:

-   **No calendar arbitrage:** for each $k$, total variance increases with maturity:  
    $\partial_t w(k,t)\ge 0$.
    
-   **No butterfly arbitrage (per slice):** the call price curve is convex in strike, equivalently the risk‑neutral density is non‑negative. (In practice one checks a function $g(k)\ge 0$ derived from $w$.)
    

**SVI (per‑slice) parameterization.** For a fixed $t$ (drop $t$ in notation),

$$
w(k)=a+b\Big(\rho\,(k-m)+\sqrt{(k-m)^2+\sigma^2}\Big),
$$

with $b\ge0,\ \sigma>0,\ |\rho|<1$. This gives the right linear “wings” and shape, but **raw SVI by itself is not guaranteed no‑arb**; naive fits may have negative density.

**SSVI (“Surface SVI”).** The paper’s key idea is to parameterize the whole surface in terms of *ATM total variance* $\theta_t=\sigma^2_{\text{BS}}(0,t)\,t$ and a positive “shape” function $\phi(\theta)$:

 
$$
\boxed{\,w(k,\theta)=\frac{\theta}{2}\Big(1+\rho\,\phi(\theta)\,k+\sqrt{(\phi(\theta)k+\rho)^2+(1-\rho^2)}\Big)\,} \tag{SSVI}
$$

(think of $\theta=\theta_t$). This is the natural SVI form with $\Delta=\mu=0$, $\omega=\theta$, $\zeta=\phi(\theta)$.

**The two crucial no‑arbitrage theorems for SSVI:**

-   **Theorem 4.1 (No calendar arbitrage).** SSVI is calendar‑arb‑free *iff*  
    (i) $\partial_t\theta_t\ge0$, and  
    (ii) $0\le\partial_\theta(\theta\phi(\theta))\le \frac{1+\sqrt{1-\rho^2}}{\rho^2}\,\phi(\theta)$ for all $\theta>0$ (upper bound is $+\infty$ if $\rho=0$). Intuition: the combination $\theta\phi(\theta)$ (“skew in **variance** terms”) must not grow too fast with $\theta$.
    
-   **Theorem 4.2 (Sufficient conditions for no butterfly arbitrage per slice).** For all $\theta>0$,
    
     
    $$
    \boxed{\ \theta\,\phi(\theta)\,(1+|\rho|) < 4\ ,\qquad \theta\,\phi(\theta)^2 \le 4\,(1-\rho^2)\ } 
    $$
    
    (the first strict to ensure the density integrates to 1). These cap the *wings* and curvature.
    

**Mappings you repeatedly need.** It’s useful to switch between SSVI $(\theta,\rho,\phi)$ and raw‑SVI $(a,b,\rho,m,\sigma)$ via $\psi=\theta\phi$:

$$
a=\tfrac{\theta}{2}(1-\rho^2),\quad b=\tfrac{\psi}{2},\quad m=-\frac{\rho}{\phi}=-\frac{\theta\rho}{\psi},\quad \sigma=\frac{\sqrt{1-\rho^2}}{\phi}=\frac{\theta\sqrt{1-\rho^2}}{\psi}.
$$

These identities are used by modern “eSSVI” calibrations and are convenient when enforcing constraints slice‑by‑slice.

**A practical recipe (what practitioners actually do).**

1.  **Data prep.** Infer forwards from put–call parity; keep OTM options (calls above $F_t$, puts below $F_t$); work in total variance vs. log‑moneyness $k$. (This is exactly what your library’s parity + prep do. TECHNICAL\_README)
    
2.  **Per‑slice fit or global fit.**  
    *Either* (A) fit each maturity with raw SVI (vega‑weighted) and **repair** to eliminate any butterfly arbitrage (paper gives a construction), then interpolate across $t$ in a calendar‑arb‑free way;  
    *or* (B) fit SSVI (or eSSVI) **directly**, choosing a $\phi(\theta)$ with few parameters (e.g., “power‑law”: $\phi(\theta)=\eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}$) and enforcing Theorems 4.1–4.2 as hard constraints.
    
3.  **Check and tune.** Verify $g(k)\ge0$ on each slice and $w(k,t)$ increasing in $t$ on a $k$‑grid; adjust using the paper’s “$+\alpha t$” trick (adding a non‑negative increasing function of $t$ to all slices preserves no‑arb).
    

---

## (2) Your repo vs. the paper — where it’s aligned and where it drifts

Below I’m reading through the modules you attached and what I can infer from the stack traces and file structure you showed.

### Strong alignments

-   **Parity & OTM‑only preprocessing.** `parity.py` infers forwards from near‑ATM call–put pairs and converts to synthetic OTM calls, which is the recommended Aït‑Sahalia & Lo style preprocessing and exactly what you describe in your technical README. ✔️ TECHNICAL\_README
    
-   **Work in forward log‑moneyness and total variance.** `surface_fitting.py` refers to `log_moneyness(...)` and “total variance”; `svi.py` implements SVI math in total‐variance space, which is the right domain for calendar checks. ✔️
    
-   **SVI parameter families.** You have raw SVI, SVI‑JW mapping, and a spline fallback. That mirrors the paper’s Section 3 (raw, natural, JW) and the “regularization/smoothing” option many desks keep for diagnostics. ✔️
    
-   **SSVI infrastructure.** `ssvi.py` implements $\phi(\theta)=\eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}$ (“power‑law” family mentioned by Gatheral) and checks margins that look like exactly the three constraints: $ \theta\phi$, $\theta\phi^2$ and the derivative bound $\partial_\theta(\theta\phi)$. ✔️
    
-   **Surface object that calibrates term structures.** `surface.py` constructs multi‑maturity slices and then calls `calibrate_ssvi_surface` or `calibrate_raw_svi_surface`. That’s architecturally on point. ✔️
    

### Misalignments / fragile choices

-   **Global SSVI fit with L‑BFGS‑B + penalties.** Your SSVI surface calibration (per the trace) runs a single `optimize.minimize(..., method="L-BFGS-B")` on all slices with transformed variables. L‑BFGS‑B handles **bounds** but not **inequality constraints**; so the no‑arb conditions must be encoded via heavy penalties. That frequently causes line‑search **“ABNORMAL\_TERMINATION”** when the objective produces NaNs/inf or wildly non‑smooth penalties. The paper steers you toward parameterizations that *stay inside* the no‑arb domain by construction (SSVI/eSSVI), not toward unconstrained global minimization with soft penalties. ✖️
    
-   **Constant $\rho$ across maturities (pure SSVI) on stressed single‑name equity.** GME’s skew moves a lot with maturity; fixing a single $\rho$ across the surface often forces $\phi(\theta)$ to extreme values to match short‑dated skews, violating the derivative bound in Theorem 4.1 or the wing caps in Theorem 4.2. That’s why desks typically move to **eSSVI** (allowing $\rho(t)$) with Hendriks–Martini calendar constraints between slices. ✖️
    
-   **Raw SVI surface assembled “slice‑by‑slice” without calendar coupling.** Your `calibrate_raw_svi_surface` fits each maturity then checks the surface and throws when the minimum calendar margin across a k‑grid is negative. That is *after‑the‑fact* detection. The paper prescribes either (i) interpolate/extrapolate in **total variance** so that $\partial_t w\ge0$ is preserved, or (ii) use SSVI constraints to make calendar monotonicity hold *during* calibration. ✖️
    
-   **Initialization and weighting.** From names I see `DEFAULT_SVI_OPTIONS`, but I don’t see explicit vega‑weighting or ATM anchoring in the objective. The paper and Gatheral’s lecture recommend **vega weighting** (or 1/vol‑spread weighting) and **anchoring the ATM point** to stabilize skew/level. Otherwise the optimizer will chase far‑OTM noise and push parameters against no‑arb walls. ✖️
    
-   **Forward inference per expiry.** Your parity module infers a forward from near‑ATM pairs (good), but when you assemble multiple expiries in `RNDSurface.from_dataframe`, forward estimation must be **per maturity**. If a single forward is reused or if staleness filtering is inconsistent across puts/calls, log‑moneyness can be biased per expiry and calendar lines will cross in total‑variance space even with perfect SVI slices. (I can’t see the full parity code, but this is a common pitfall.) ✖️
    

---

## (3) Why your `ssvi` and `raw_svi` surfaces fail — causes & fixes

### A. `VolModel(method="ssvi")` → `CalculationError: SSVI calibration failed: ABNORMAL ...`

**What’s happening.**  
SciPy’s L‑BFGS‑B line‑search aborts when the objective (your global SSVI fit) hits non‑smooth regions or yields NaN/Inf. That’s typical when:

1.  **The no‑arb constraints are encoded as penalties.** For some trial $(\rho,\eta,\gamma,\{\theta_i\})$, one of  
    $\theta\phi(\theta)(1+|\rho|) < 4$,  
    $\theta\phi(\theta)^2 \le 4(1-\rho^2)$,  
    $0\le \partial_\theta(\theta\phi(\theta)) \le \frac{1+\sqrt{1-\rho^2}}{\rho^2}\phi(\theta)$  
    is violated badly; the penalty explodes or the derivative of the penalty is undefined; the line‑search cannot proceed.
    
2.  **Constant $\rho$ is too restrictive** for GME short‑dated skews. The optimizer tries to push $\phi(\theta)$ high to match near‑term skew, breaching the wing caps (Theorem 4.2) for some maturities.
    
3.  **Poor initialization / anchoring.** If $\theta_i$ are free and not initialized from ATM quotes, or if the objective isn’t vega‑weighted, the first steps can be far out‑of‑domain.
    
4.  **Numerical k‑grid + noisy quotes.** With very sparse/illiquid strikes (common in GME), total‑variance and skew estimates at short maturities are extremely noisy; this creates a jagged objective that’s hard for a quasi‑Newton solver.
    

**Fixes that work in practice (and that align with the paper).**

-   **Switch to (extended) SSVI calibration that enforces no‑arb *by parameterization*, not by penalties.**  
    Calibrate per‑slice parameters $(\theta_i,\rho_i,\psi_i)$ with $\psi_i=\theta_i\phi(\theta_i)$ and enforce the slice inequalities  
    $\psi_i\le \frac{4}{1+|\rho_i|}$ and $\psi_i \le 2\sqrt{(1-\rho_i^2)\,\theta_i}$ *as bounds*, not penalties. Then ensure calendar no‑arb between adjacent slices via the Hendriks–Martini inequalities (sequential or global eSSVI). This is the robust “eSSVI” route used by many desks.
    
-   **If you keep the pure SSVI $\phi(\theta)=\eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}$ with a *single* $\rho$:**
    
    -   Constrain $\gamma\in(0,\frac12]$ (the paper’s “power‑law” example) and $\eta$ so that the two slice bounds hold for the largest $\theta$ in your data. Use a tanh reparam for $\rho$.
        
    -   Initialize $\theta_i$ from ATM quotes; initialize $\eta,\gamma$ from the **ATM skew** relation (eq. (4.2) in the paper):  
        $\partial_k\sigma\big|_{k=0}=\dfrac{\rho\sqrt{\theta}}{2\sqrt{t}}\phi(\theta)$.
        
    -   Use **vega‑weighted** squared‑vol errors (or price errors weighted by 1/bid‑ask) and pin the ATM point (hard constraint).
        
-   **Smooth & monotone $\theta(t)$.** Fit $\theta(t)$ by a monotone spline in maturity (or in forward‑variance time), then hold $\theta_i$ fixed in the SSVI fit. This satisfies $\partial_t\theta_t\ge0$ a priori, meeting Theorem 4.1 (i).
    

### B. `VolModel(method="raw_svi")` → `Calendar arbitrage detected (min margin -7.642e-02)`

**What’s happening.**  
Your code fits each slice (raw SVI), then checks calendar monotonicity on a common $k$‑grid. Because slices are calibrated **independently** (and possibly with slightly inconsistent forwards or noisy quotes), some $k$ values end up with $w(k,t_2)<w(k,t_1)$ for $t_2>t_1$. Your `strict_no_arbitrage=True` then (rightly) raises.

**Why it happens easily:**

-   Slice‑only fits have no cross‑maturity coupling; with equities like GME the ATM point can move and near‑ATM strikes are sparse. The paper warns that independently good slices do **not** guarantee an arbitrage‑free surface without careful interpolation/extrapolation in total‑variance.
    
-   If forwards were not perfectly inferred **per maturity**, the definition of $k$ differs across slices, making calendar lines cross spuriously.
    

**Two battle‑tested fixes:**

1.  **Stop assembling a surface from independent raw‑SVI slices.** Use SSVI/eSSVI to fit all maturities within the no‑arb domain (above).
    
2.  **If you insist on raw SVI slices:**
    
    -   Constrain interpolation/extrapolation in **total‑variance**: interpolate linearly in $\theta$ and SVI‑JW parameters in a way that preserves $\partial_t w\ge0$. This is essentially the “Lemma 5.1” approach (there exists a no‑arb interpolation between two arbitrage‑free slices with $w(\cdot,t_2)\ge w(\cdot,t_1)$).
        
    -   Add the paper’s “$+\alpha t$” adjustment: if you start from an SSVI surface $w$ that’s no‑arb, then $w_\alpha(k,t)=w(k,t)+\alpha(t)\,t$ with $\alpha(t)\ge0$ increasing is also no‑arb; this can *repair* small negative calendar margins due to numerical noise.
        

---

## Concrete, file‑level suggestions

-   **`ssvi.py`**
    
    -   Keep your `phi_eta_gamma` (“power‑law”) but enforce at construction time:
        
        -   $\gamma\in(0,\tfrac12]$ (use a logistic map to that interval),
            
        -   $|\rho|\le 0.95$ (tighter than 0.999 for stability on idiosyncratic names),
            
        -   compute and expose the three margins you already compute; *clip parameters* so margins stay positive instead of relying on penalties to push them back.
            
    -   Add a fast evaluator that returns $(a,b,\rho,m,\sigma)$ from $(\theta,\rho,\phi)$ using the mapping above; calibrate in that space if you already have raw‑SVI machinery.
        
-   **`surface_fitting.py` / `svi.py`**
    
    -   Use **vega‑weighted** least squares (or 1/vol‑spread weights) as default. Gatheral emphasizes this in his lecture because ATM errors dominate price P&L.
        
    -   Initialize raw‑SVI from SVI‑JW computed off ATM variance, ATM skew, and wing slopes estimated from far OTM quotes (paper gives formulas), then refine—this avoids the optimizer shooting into arbitrage regions early.
        
-   **`parity.py` / `prep.py`**
    
    -   Ensure forwards are inferred **per expiry** (it looks like you do this) and that you drop stale quotes consistently for both puts and calls before parity; this greatly reduces artificial calendar crossings. Your Technical README stresses staleness filtering—make sure it’s applied slice‑wise. TECHNICAL\_README
        
-   **`surface.py`**
    
    -   For `method="ssvi"`, either:
        
        -   replace the global L‑BFGS‑B with a **sequential eSSVI** fit: for each maturity, choose $(\theta_i,\rho_i,\psi_i)$ satisfying the slice bounds; then step to the next maturity enforcing the Hendriks–Martini calendar constraints between slices; or
            
        -   keep a global optimizer but **reparameterize to a rectangular domain**: optimize in $(u_i,v_i,r_i)$ with  
            $\theta_i = \text{monotone\_spline}(u)$,  
            $\rho_i=\tanh(r_i)\cdot\rho_{\max}$,  
            $\psi_i = \min\!\big(\frac{4}{1+|\rho_i|},\ 2\sqrt{(1-\rho_i^2)\theta_i}\big)\cdot \sigma(v_i)$  
            where $\sigma(\cdot)\in(0,1)$ (logistic). That way no‑arb is enforced by construction and L‑BFGS‑B sees only box constraints.
            
    -   For `method="raw_svi"`, if you keep the checker that raised
        
        ```arduino
        Calendar arbitrage detected ... min margin -7.642e-02
        ```
        
        add an *automatic repair loop*: nudge slices by a small **increasing** additive $+\alpha_i t_i$ per Theorem 4.3 until the minimum margin is non‑negative (or switch to eSSVI as above).
        

---

## Why this matters for your specific run (`gme`, 1M horizon)

-   **Sparse, noisy, short‑dated GME quotes** make ATM skew and wing estimates volatile. A constant‑$\rho$ SSVI will try to accommodate this via large $\phi(\theta)$, violating the slice bounds—hence “ABNORMAL” termination.
    
-   **Independently fitted raw‑SVI slices** on such data easily cross in total variance at some $k$ (especially if forwards differ slightly or quotes are stale), hence the negative calendar margin your error reports.
    
-   Your defaults (`strict_no_arbitrage=True`) are doing the right thing by refusing to return a surface that fails the checks; the calibration path just needs to move from “penalty‑based” to “domain‑by‑construction”.
    

---

## A short implementation checklist (that will make your solvers converge)

1.  **Estimate $\theta_i$ from ATM vols** and smooth $\theta(t)$ monotonically. (Hard‑enforce $\partial_t\theta\ge0$.)
    
2.  **Choose eSSVI**: calibrate $(\rho_i,\psi_i)$ per slice with bounds  
    $\psi_i \le \frac{4}{1+|\rho_i|}$ and $\psi_i \le 2\sqrt{(1-\rho_i^2)\theta_i}$;  
    enforce Hendriks–Martini between slices to prevent calendar arbitrage.
    
3.  **Use vega‑weighted objectives and pin ATM**; discard obviously stale/far‑OTM quotes.
    
4.  **If you keep pure SSVI with $\phi(\theta)=\eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}$:** constrain $\gamma\in(0,\tfrac12]$, $|\rho|\le 0.95$, and bound $\eta$ so the two slice inequalities hold at your largest $\theta$. Initialize $(\eta,\gamma,\rho)$ from ATM skews via eq. (4.2).
    
5.  **Always validate** $g(k)\ge0$ on a wide $k$‑grid and $w(k,t)$ is increasing in $t$. If small negatives arise from numerics, repair with the  $\,+\alpha t\,$ adjustment.
    

---

### Final note on your documentation vs. the SSVI workstream

Your public docs (README/TECHNICAL\_README) emphasize RND extraction via IV smoothing and parity preprocessing (good), and list “implement no‑arbitrage checks” and “full term‑structure surface” as roadmap items. Adding the SSVI/eSSVI calibration path I’ve outlined above is exactly the missing piece that will turn your current prototype into a reliable, arbitrage‑aware surface builder that matches the 2012 paper’s spirit.

If you want, I can sketch a minimal eSSVI calibration loop in pseudocode next, or point out exactly where to drop each check into your current `calibration/ssvi_surface.py` and `raw_svi_surface.py`.