# PRD: SVI‑JW Calibration (practice + code)

**TL;DR:** Build a fast, **SVI‑JW** (Stochastic Volatility Inspired – *Jump‑Wings*) calibration service that fits each maturity slice robustly, enforces **no static arbitrage** within slices and across maturities, and exposes a clean API with well‑chosen bounds, penalties, and reproducible defaults.

---

## 1) McKinsey‑style top‑down plan

- **What we’re solving.** We convert noisy option quotes into an **arbitrage‑free volatility surface** using the **SVI‑JW** parameterization for interpretability (ATM level/skew and wing slopes), then publish parameters and pricing functions. **Why it matters:** fewer hedging pathologies and stable extrapolation.
- **How we’ll do it.**
  1. Clean data → forward moneyness → implied vols/total variance.
  2. **Per maturity**, calibrate **JW** parameters under tight bounds; penalize butterfly arbitrage and out‑of‑spread fits; polish with a deterministic method.
  3. **Across maturities**, enforce term‑structure consistency using **SSVI** conditions or call‑price interpolation to remove calendar arbitrage.
- **What we’ll deliver.** A Python library returning JW and raw‑SVI parameters, diagnostics, and pricers, with SLAs on speed and stability.
- **Design choice.** **JW** is more interpretable; **raw SVI** remains common for slice fitting; **SSVI** anchors the surface for no‑arbitrage across maturities.

---

## 2) Detailed spec and steps

### A. Scope & users
- **Users.** Volatility traders, risk managers, and model engineers.
- **In‑scope.** Equity/Index vanillas; multiple expiries; outlier‑robust calibration; static no‑arbitrage checks.
- **Out‑of‑scope (v1).** Smile dynamics, local‑vol extraction, and exotics pricing.

### B. Inputs and preprocessing
- **Inputs.** Time‑stamped quotes (bid/ask/mid, size), strikes, expiries, spot, dividends, and interest rates.
- **Preprocessing.** Compute forwards per expiry; log‑moneyness \(x=\ln(K/F)\); implied vol \(\sigma\); **total variance \(w=\sigma^2\,\tau\)**. Filter stale/locked markets; down‑weight unreliable tails.

### C. Per‑maturity model and bounds
- **Model.** Use **SVI‑JW** parameters \((v_\tau,\psi_\tau,p_\tau,c_\tau,\hat v_\tau)\) with the standard mapping to raw SVI to evaluate \(w(x)\). JW parameters correspond to **ATM variance**, **ATM skew**, **left/right wing slopes**, and **minimum variance**.
- **Guardrails (raw‑SVI domain).** Enforce: \(\sigma>0\), \(b\ge 0\), \(\rho\in[-1,1]\), reasonable bounds on \(m\) from the strike range, and small positive lower bounds \(a,\sigma\) to avoid degeneracy. Cap \(b\) to prevent pathological wings.

### D. Objective function (per slice)
- **Loss.** Weighted squared error in **implied vol** or **total variance** versus market mid.
- **Penalties.**
  - **Butterfly arbitrage penalty:** penalize when Durrleman’s convexity density test fails (i.e., negative density).
  - **Bid‑ask envelope penalty:** penalize if fitted vols leave quoted spreads.
  - **Regularization:** soft priors linking neighboring expiries for stability.

### E. Optimization
- **Strategy.** Run a global search (e.g., Differential Evolution) followed by a **polish** step (e.g., L‑BFGS or Nelder‑Mead). Enable multi‑start; warm‑start from neighbor expiries.
- **Speed‑up (QE split).** Use a **Quasi‑Explicit split** that treats \((\sigma,m)\) as outer variables and solves an inner least‑squares for the remaining parameters each iteration; map back to raw/JW. This materially reduces runtime.

### F. Across‑maturity surface (no calendar arbitrage)
- **Preferred:** Fit slices first, then **interpolate call prices** monotonically in maturity to guarantee no calendar arbitrage.
- **Alternative:** Calibrate a global **SSVI** (Surface SVI) with conditions on \(\theta(\tau)\) and \(\phi(\theta)\) that are necessary/sufficient to avoid butterfly **and** calendar arbitrage; then read off induced JW‑like quantities.

### G. Data weighting & robustness
- Weight points by **vega** and **volume**; down‑weight illiquid tails; cap extreme residuals. Use expiry‑specific outlier filters to stabilize short‑dated wings.

### H. API & outputs
- **Endpoints.**
  - `POST /calibrate`: input = quotes by expiry; output = per‑expiry **JW** and **raw SVI** params, fit error, and arbitrage flags.
  - `GET /surface`: return smooth samplers \(w(\tau,x)\) and \(\sigma(\tau,x)\).
  - `POST /price`: price via Black–Scholes using fitted \(\sigma\).
- **Artifacts.** JSON parameters, PNG plots, and CSV dumps for auditability.

### I. Acceptance criteria
- **Within‑slice:** Non‑negative density on a grid; ≥99% of liquid points inside bid–ask; RMSE below a preset threshold.
- **Across maturities:** Monotone ATM total variance; no calendar arbitrage under price‑space checks or SSVI conditions.
- **Ops:** 95th‑percentile latency < 200 ms per slice on ~200 points; reproducible to 1e‑6 with the same seed.

### J. Risks & mitigations
- **Noisy wings / short‑dated jumps.** Use wing penalties and vega/volume weighting.
- **Parameter degeneracy.** Guard with bounds and the QE split; monitor condition numbers.
- **Surface stitching.** Prefer price‑space interpolation or SSVI constraints to avoid calendar arbitrage.

---

## 3) Notes on SVI‑JW vs other variations (practice reality)

- **Does SVI‑JW have only upsides?** Not entirely. **Upsides:** parameters map to trader‑intuitive quantities (ATM level/skew and wing slopes) and are helpful for constraints. **Trade‑offs:** you still must enforce no‑arbitrage; JW adds a mapping step to/from raw SVI; numerical edge cases appear near parameter boundaries.
- **What is most used in practice?** **Raw SVI** is still the workhorse for **per‑expiry** fitting; **JW** is used for interpretation and stabilizing constraints; **SSVI** or price‑space interpolation is used to ensure a **calendar‑free surface** across maturities.

---

## 4) Implementation checklist (engineer’s to‑do)

1. Data pipeline: forwards, moneyness, implied vols, total variance.
2. Bounds + QE split; mappings between JW and raw SVI.
3. Objective with butterfly and bid–ask penalties; global‑then‑local solver.
4. Slice diagnostics; export JW/raw; pricing helper.
5. Surface stitching (price‑space or SSVI); calendar‑arbitrage checks.
6. Tests: regression, stress tests on wings/short tenors; performance budget.

---

## Glossary

- **SVI:** *Stochastic Volatility Inspired*; a five‑parameter total‑variance smile model.
- **JW:** *Jump‑Wings*; an SVI re‑parameterization using ATM level/skew and wing slopes.
- **ATM / OTM:** *At‑ / Out‑of‑the‑Money*; strike near / far from the forward price.
- **SSVI:** *Surface SVI*; an SVI family with term‑structure conditions that prevent static arbitrage.
- **Butterfly arbitrage:** A within‑slice convexity violation implying negative risk‑neutral density.

---

## References (practical guides)

- Aurell, *SVI – Calibration and Arbitrage Considerations* (MSc thesis). https://www.diva-portal.org/smash/get/diva2:744907/FULLTEXT02.pdf
- Quant Chronicles, *Model Calibration Using Optimization Engines – An Example Using Implied Volatility*. https://medium.com/quant-chronicles/model-calibration-using-optimization-engines-an-example-using-implied-volatility-cf6c3233a8ba



##  2. SVI Calibration Code Review and Refactor Plan

2.1 Executive Summary
The current package correctly: cleans quotes, infers a forward via put–call parity, computes implied volatilities (IV), and calibrates a raw SVI slice with sensible bounds and two no‑arbitrage penalties (butterfly via g(k)g(k)g(k) and call‑spread monotonicity). This is implemented mainly in estimator.py (data flow & orchestration), surface_fitting.py (dispatch to “svi” vs “bspline”), and svi.py (parameters, objective, L‑BFGS‑B optimizer). However, it does not yet implement SVI‑JW, uses only a local optimizer (no global search), lacks bid–ask envelope penalties and vega/volume weighting, and exposes limited diagnostics. The refactor below adds JW as a first‑class parametrization, cleanly separates preprocessing/fitting/diagnostics, and introduces a robust global + polish calibration path.

2.2 Technical Plan
🧩 Part 1: Code vs PRD Alignment
Per‑slice calibration flow
	• ✅ Forward & tenor: You infer a forward via apply_put_call_parity(..) and pass it into SVI fitting (surface_fitting._fit_svi(...)), then price on a strike grid and derive the PDF/CDF (Breeden–Litzenberger).
	• ⚠️ Single‑expiry only: No cross‑maturity stitching or SSVI‑style (Surface SVI) constraints, which is fine for RND v1 but out of scope for the PRD’s “calendar‑free surface” stretch goal.
Objective, constraints, penalties (in svi.py)
	• ✅ Objective in total variance: Least‑squares on w(k)=σ2τw(k)=\sigma^2\tauw(k)=σ2τ is used.
	• ✅ Arbitrage checks: You include a butterfly penalty via g_function(k) and a call‑spread monotonicity penalty.
	• ⚠️ No bid–ask envelope penalty: The PRD requires a hinge penalty if the fitted IV leaves [bid_iv,ask_iv][{\rm bid\_iv}, {\rm ask\_iv}][bid_iv,ask_iv]. Currently the bid/ask IVs are computed (for plotting) but not enforced in the loss.
	• ⚠️ No data weighting: Residuals are unweighted. The PRD calls for vega/volume weighting and down‑weighting wings.
	• ⚠️ Static bounds are coarse: Bounds (_build_bounds) are mostly fixed caps; the PRD recommends data‑adaptive bounds (e.g., wider mmm when strikes span is narrow, lower σmin⁡\sigma_{\min}σmin​ safeguards) and guardrails linked to tenor.
Optimization strategy
	• ⚠️ Local only: You use L‑BFGS‑B from a heuristic _initial_guess(...). The PRD specifies global search (e.g., Differential Evolution or CMA‑ES) then polish (L‑BFGS‑B/Nelder–Mead), with optional warm‑starts from neighboring expiries.
Parametrizations
	• ⚠️ Raw SVI only: SVIParameters(a,b,ρ,m,σ) are supported; SVI‑JW is not implemented. The PRD requires JW for interpretability (ATM level/skew, wing slopes) and explicit JW↔raw mappings.
Diagnostics & outputs
	• ✅ You attach evaluation grids to the fitted curve and expose plotting.
	• ⚠️ Missing structured diagnostics: Min g(k)g(k)g(k), fraction of points inside bid–ask, RMSE metrics, and a summary of constraint activity are not exposed as a standard “fit report.”


🛠 Part 2: Refactor & Implementation Advice
A. Make SVI‑JW a first‑class citizen
	1. Param classes

@dataclass(frozen=True)
class SVIParameters:
    a: float; b: float; rho: float; m: float; sigma: float

@dataclass(frozen=True)
class SVIJW:
    v: float      # ATM total variance w(0)
    psi: float    # ATM skew w'(0)
    p: float      # left wing slope magnitude
    c: float      # right wing slope magnitude
    vmin: float   # minimum total variance
	2. Deterministic mappings (raw→JW always analytic).

def raw_to_jw(raw: SVIParameters) -> SVIJW:
    a,b,rho,m,sigma = raw.a,raw.b,raw.rho,raw.m,raw.sigma
    s = (m*m + sigma*sigma) ** 0.5
    v    = a + b*(s - rho*m)                  # ATM level
    psi  = b*(rho - m/s)                      # ATM skew
    p    = b*(1 - rho)                        # left wing slope (>0)
    c    = b*(1 + rho)                        # right wing slope (>0)
    vmin = a + b*sigma*(1 - rho*rho) ** 0.5   # minimum variance
    return SVIJW(v, psi, p, c, vmin)
	3. Numerical mapping (JW→raw).
Compute b=p+c2b=\tfrac{p+c}{2}b=2p+c​, ρ=c−pp+c\rho=\tfrac{c-p}{p+c}ρ=p+cc−p​, then solve for s=m2+σ2s=\sqrt{m^2+\sigma^2}s=m2+σ2​ from the two ATM identities w(0)=vw(0)=vw(0)=v and w′(0)=ψw'(0)=\psiw′(0)=ψ plus vmin⁡v_{\min}vmin​ (use a 1‑D root for sss; recover m=s(ρ−ψ/b)m=s(\rho - \psi/b)m=s(ρ−ψ/b), σ=s1−(ρ−ψ/b)2\sigma=s\sqrt{1-(\rho-\psi/b)^2}σ=s1−(ρ−ψ/b)2​, a=vmin⁡−b σ1−ρ2a=v_{\min}-b\,\sigma\sqrt{1-\rho^2}a=vmin​−bσ1−ρ2​). Verify numerically that raw_to_jw(jw_to_raw(.)) is identity to 1e‑8.
	4. Add 'svi-jw' as an additional option to surface_method
B. Split optimization paths cleanly
	• Internal representation: Run the optimizer in raw space only (evaluation is easiest), but accept either raw or JW inputs/outputs. When the user requests JW, map the initial guess JW→raw, optimize raw, then return both raw and JW in the result.
	• Global + polish:
		○ Global: scipy.optimize.differential_evolution over _build_bounds(k, τ) to get a robust seed.
		○ Polish: L-BFGS-B (with analytic gradients optional) or Nelder–Mead for noisy data.
		○ Determinism: Allow random_seed and max_iter in surface_options.
C. Strengthen the objective
	• Weighted residuals: Minimize ∑iwi[wmodel(ki)−wi]2\sum_i w_i [w_{\text{model}}(k_i)-w_i]^2∑i​wi​[wmodel​(ki​)−wi​]2 with wi∝w_i \proptowi​∝ vega × volume (fallback to vega if volume absent). Cap weights in deep wings; optionally Huberize residuals.
	• Bid–ask envelope penalty: Hinge penalty when fitted IV leaves [bid_iv,ask_iv][{\rm bid\_iv},{\rm ask\_iv}][bid_iv,ask_iv].

def bid_ask_penalty(model_iv, bid_iv, ask_iv, lam=1e3):
    below = np.maximum(0.0, bid_iv - model_iv)
    above = np.maximum(0.0, model_iv - ask_iv)
    return lam * float(np.sum(below*below + above*above))
	• Keep existing: g(k) butterfly penalty and call‑spread penalty; expose their weights in options.
D. Improve bounds & initial guesses
	• Data‑adaptive bounds:
		○ m∈[kmin⁡−Δ, kmax⁡+Δ]m \in [k_{\min}-\Delta,\, k_{\max}+\Delta]m∈[kmin​−Δ,kmax​+Δ] with Δ=max⁡(1,kmax⁡−kmin⁡)\Delta=\max(1, k_{\max}-k_{\min})Δ=max(1,kmax​−kmin​).
		○ σ≥σmin⁡\sigma \ge \sigma_{\min}σ≥σmin​ (e.g., 1e-41\text{e-}41e-4), ∣ρ∣≤0.999|\rho| \le 0.999∣ρ∣≤0.999.
		○ Set bbb upper bound wider for short maturities; optionally relate to tenor.
	• Initial guess: from observed w(k)w(k)w(k): m0=arg⁡min⁡wm_0=\arg\min wm0​=argminw, a0=w(m0)a_0=w(m_0)a0​=w(m0​), small b0b_0b0​, ρ0=0\rho_0=0ρ0​=0, σ0≈\sigma_0\approxσ0​≈ stdev of kkk. When JW is requested, seed via JW→raw mapping from a heuristic JW guess.
E. Modularity & readability
	• Preprocessing module (prep.py): Keep parity, staleness filtering, price selection. Add a “robust strike filter” (remove obvious outliers, optional).
	• Fitting module (svi.py):
		○ svi_options(...) → replace dicts with a small @dataclass (type‑safe).
		○ Expose calibrate_slice(...) -> FitResult returning: params (raw & JW), RMSE (weighted), min g(k)g(k)g(k), envelope breaches %, and solver diagnostics.
	• Surface dispatch (surface_fitting.py):
		○ fit_surface(method="svi", options=...) now reads svi_parametrization, global_solver, polish_solver.
		○ Returned callable attaches .params_raw, .params_jw, .diagnostics, and .grid.
	• Diagnostics: Add result.iv_smile() to include bid/ask IVs, weights, residuals, g(k)g(k)g(k) on a grid, and flags for any arbitrage violation.
F. Tests & examples
	• Round‑trip tests for JW↔raw mappings.
	• Noise stress tests (short‑dated wings), regression tests on a few expiries.
	• Timing tests to keep p95 < 200 ms/slice at ~200 points (with global+polish, cache the global seed per expiry).


## 3) SVI Calibration Failure Analysis (why it fails and how to harden it)
3.1 What’s different vs the literature (and why this bites)
	• Butterfly test g(k) is wrong. The code computes
g = (1 − k·w'/(2w)) * (1 − w'/2) − (w'^2)*(1/(4w)+1/16) + 0.5·w'',
but the canonical condition is
g = (1 − k·w'/(2w))^2 − (w'^2)/4·(1/w + 1/4) + 0.5·w''.
The missing square causes spurious negatives of g(k) and “failure” even when the slice is fine. Imperial College London
	• Local optimizer only. Calibration uses L‑BFGS‑B with a single heuristic seed and wide, mostly static bounds; no global search or multi‑start is attempted. This is brittle for skewed, short‑dated, or sparse smiles. 
	• Loss is unweighted and lacks market guards. Residuals are plain least squares in total variance; there is no vega/volume weighting and no bid–ask envelope penalty, so deep‑wing noise can dominate and push the fit into arbitrage. 
	• Auxiliary penalties/knobs. The call‑spread penalty uses a fixed step 0.05 in log‑moneyness, which is too coarse for short tenors and too fine for long ones; penalty weights are large and static, making the objective ill‑conditioned on some names. 
	• Fallback hides root causes. On failure you silently switch to B‑spline and proceed, so the observed “many tickers fail SVI” accumulates without targeted fixes. 
Why it matters: users see success, but the model is not the one intended in the PRD; diagnostics do not reveal why SVI failed. The public docs also emphasize a B‑spline smile by default, reinforcing the mismatch. 

3.2 Concrete fixes (code) and the tests to prove them
A) Fix the butterfly diagnostic now
Replace your g_function with the literature‑correct form (Gatheral–Jacquier).

def g_function(k, params):
    k  = np.asarray(k, float)
    w  = svi_total_variance(k, params)
    wp = svi_first_derivative(k, params)
    wpp = svi_second_derivative(k, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - (k*wp)/(2.0*w))**2 - 0.25*(wp**2)*(1.0/w + 0.25) + 0.5*wpp
    return np.where(np.isfinite(g), g, -np.inf)
Test: For known arbitrage‑free SVI/SSVI parameter sets from the literature, assert min g(k) ≥ 0 on a dense grid; for obviously bad sets, assert min g(k) < 0. Imperial College London
B) Add a robust optimizer pipeline
	• Global + polish: Wrap the current objective in scipy.optimize.differential_evolution (or CMA‑ES) to get a good seed, then polish with L‑BFGS‑B. Keep random seeds and iteration budgets in options.
	• Multi‑start (cheap mode): If global is disabled, run 5–10 randomized seeds (jitter m, σ, ρ) and keep the best.
Tests: On a daily SPY slice and 3–5 idiosyncratic tickers, record success rate and RMSE before/after; require ≥95% success and ≥30% RMSE reduction vs current. (Use the repo’s existing I/O & plotting for visual regression.) 
C) Weight the data and respect the tape
	• Vega/volume weights: Weight squared residuals by vega × volume (clip extremes; fallback to vega).
	• Bid–ask envelope: Add a hinge penalty if model IV exits [bidIV, askIV]; expose weight in options.
Tests:
	1. On liquid underlyings, require ≥99% of points inside the envelope.
	2. On low‑volume names, confirm wings do not dominate: compare weighted vs unweighted RMSE on the central 50% of strikes (weighted should be lower). 
D) Make penalties adaptive
	• Call‑spread step: Use step = 0.5 * median(diff(k)) rather than a constant 0.05; scale weight down for very short maturities.
	• Bounds: Keep σ ≥ 1e−4, |ρ| ≤ 0.999 but set m ∈ [k_min − span, k_max + span] with span = max(1, k_max − k_min), and cap b with a tenor‑aware upper bound (shorter expiries need tighter b_max).
Tests: Parameter‑stress suite that sweeps step, weights, and bounds over grids; assert stability (no explosions, consistent minima).
E) Improve the initial guess
Initialize m at the empirical min of total variance; set a = w_min, ρ=0, b≈0.1, σ≈std(k) (already close), then run a QE split (outer (m,σ)(m,σ)(m,σ), inner linear LS for the remaining three) to get a sharper seed before the global stage. Test: With the global stage off, QE‑seeded local runs should beat current RMSE ≥20% on average.
F) Observability
Expose a standard fit report: RMSE (weighted/unweighted), min g(k), % inside bid–ask, active constraints, and optimizer diagnostics; log when the fallback to B‑spline is used. This turns “SVI failed a lot” into measurable categories. 

3.3 Minimal patch set you can ship first
	1. Fix g_function (one‑line square);
	2. Add bid–ask envelope penalty and adaptive call‑spread step;
	3. Enable multi‑start (5 seeds) before L‑BFGS‑B;
	4. Weight by vega (volume optional).
These four changes typically take SVI success from “spotty” to “robust” even before you add a full global optimizer.
