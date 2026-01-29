from __future__ import annotations

import numpy as np
import pytest

from oipd.core.vol_surface_fitting.shared.svi import (
    SVICalibrationOptions,
    SVIParameters,
    _CallSpreadContext,
    _adaptive_call_spread_step,
    _build_bounds,
    _bid_ask_penalty,
    _butterfly_penalty,
    _call_spread_penalty,
    _compute_huber_delta,
    _initial_guess,
    _penalty_terms,
    _svi_total_variance_with_derivatives,
    _vega_based_weights,
    JWParams,
    RawSVI,
    from_total_variance,
    g_function,
    jw_to_raw,
    log_moneyness,
    min_g_on_grid,
    merge_svi_options,
    raw_to_jw,
    svi_first_derivative,
    svi_minimum_location,
    svi_minimum_variance,
    svi_second_derivative,
    svi_total_variance,
    to_total_variance,
)


def make_sample_params() -> SVIParameters:
    # Representative arbitrage-free parameters
    return SVIParameters(a=0.04, b=0.2, rho=-0.4, m=-0.1, sigma=0.3)


def test_merge_svi_options_returns_dataclass():
    options = merge_svi_options({"max_iter": 123})
    assert isinstance(options, SVICalibrationOptions)
    assert options.max_iter == 123

    with pytest.raises(TypeError):
        merge_svi_options({"invalid_option": 1})


def test_log_moneyness_basic():
    forward = 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    k = log_moneyness(strikes, forward)
    np.testing.assert_allclose(np.exp(k) * forward, strikes)


def test_log_moneyness_invalid():
    with pytest.raises(ValueError):
        log_moneyness([100.0], -1.0)
    with pytest.raises(ValueError):
        log_moneyness([0.0], 100.0)


def test_total_variance_round_trip():
    iv = np.array([0.2, 0.3])
    T = 0.25
    total_var = to_total_variance(iv, T)
    recovered_iv = from_total_variance(total_var, T)
    np.testing.assert_allclose(recovered_iv, iv)


def test_total_variance_invalid_inputs():
    with pytest.raises(ValueError):
        to_total_variance([0.2], 0.0)
    with pytest.raises(ValueError):
        from_total_variance([-0.1], 0.5)


def test_svi_minimum_matches_formula():
    params = make_sample_params()
    k_star = svi_minimum_location(params)
    w_star = svi_minimum_variance(params)

    k_grid = np.linspace(k_star - 1.0, k_star + 1.0, 5)
    w_vals = svi_total_variance(k_grid, params)
    assert w_vals.min() == pytest.approx(w_star, rel=1e-6)


def test_g_function_arbitrage_free_positive():
    params = make_sample_params()
    k_grid = np.linspace(-2.0, 2.0, 101)
    g_vals = g_function(k_grid, params)
    assert np.all(g_vals >= -1e-10)  # allow tiny numerical slack
    assert min_g_on_grid(params, k_grid) >= -1e-10


def test_g_function_detects_arbitrage():
    params = SVIParameters(a=0.01, b=5.0, rho=0.99 - 1e-6, m=0.0, sigma=0.01)
    k_grid = np.linspace(-0.5, 0.5, 51)
    g_vals = g_function(k_grid, params)
    assert np.any(g_vals < 0)
    assert min_g_on_grid(params, k_grid) < 0


def _canonical_g(k: np.ndarray, params: SVIParameters) -> np.ndarray:
    """Compute the canonical Gatheral–Jacquier g(k) formula."""

    w = svi_total_variance(k, params)
    wp = svi_first_derivative(k, params)
    wpp = svi_second_derivative(k, params)
    term1 = 1 - (k * wp) / (2 * w)
    term1_sq = np.square(term1)
    term3 = (wp**2) * (1 / (4 * w) + 1 / 16)
    return term1_sq - term3 + 0.5 * wpp


def test_g_function_matches_canonical_formula():
    params = make_sample_params()
    k_grid = np.linspace(-1.5, 1.5, 101)
    numerical = g_function(k_grid, params)
    canonical = _canonical_g(k_grid, params)
    np.testing.assert_allclose(numerical, canonical, rtol=1e-12, atol=1e-12)


def test_g_function_differs_from_erroneous_formula():
    params = SVIParameters(a=0.03, b=0.4, rho=-0.6, m=0.05, sigma=0.2)
    k_grid = np.linspace(-2.0, 2.0, 301)

    correct = g_function(k_grid, params)

    w = svi_total_variance(k_grid, params)
    wp = svi_first_derivative(k_grid, params)
    wpp = svi_second_derivative(k_grid, params)
    term1 = 1 - (k_grid * wp) / (2 * w)
    erroneous = term1 * (1 - wp / 2) - (wp**2) * (1 / (4 * w) + 1 / 16) + 0.5 * wpp

    assert not np.allclose(correct, erroneous)
    np.testing.assert_allclose(
        correct, _canonical_g(k_grid, params), rtol=1e-12, atol=1e-12
    )


def test_build_bounds_expand_with_span():
    config = merge_svi_options({})
    k_narrow = np.linspace(-0.1, 0.1, 11)
    bounds_narrow = _build_bounds(k_narrow, 0.5, config)

    k_wide = np.linspace(-2.0, 2.0, 11)
    bounds_wide = _build_bounds(k_wide, 0.5, config)

    narrow_m_span = bounds_narrow[3][1] - bounds_narrow[3][0]
    wide_m_span = bounds_wide[3][1] - bounds_wide[3][0]
    assert wide_m_span > narrow_m_span


def test_build_bounds_respect_sigma_min():
    config = merge_svi_options({"sigma_min": 0.25})
    k = np.linspace(-0.5, 0.5, 9)
    bounds = _build_bounds(k, 0.25, config)
    assert bounds[4][0] == pytest.approx(0.25)


def test_initial_guess_responds_to_skew():
    params = SVIParameters(a=0.03, b=0.4, rho=-0.7, m=-0.05, sigma=0.25)
    k = np.linspace(-0.4, 0.4, 9)
    total_var = svi_total_variance(k, params)
    guess = _initial_guess(k, total_var)

    assert guess.shape == (5,)
    assert guess[1] > 0  # b0 positive
    # rho initial guess should respond to skew (not default zero)
    assert abs(guess[2]) > 0


def test_bid_ask_penalty_behavior():
    params = make_sample_params()
    maturity = 0.5
    k = np.linspace(-0.3, 0.3, 7)
    total_var = svi_total_variance(k, params)
    model_iv = from_total_variance(total_var, maturity)

    bid_iv = model_iv - 0.01
    ask_iv = model_iv + 0.01
    assert _bid_ask_penalty(model_iv, bid_iv, ask_iv, 1e3) == pytest.approx(0.0)

    tight_ask = model_iv - 0.02
    penalty = _bid_ask_penalty(model_iv, bid_iv, tight_ask, 1.0)
    assert penalty > 0


def test_raw_to_jw_round_trip():
    raw = SVIParameters(a=0.04, b=0.25, rho=-0.3, m=0.1, sigma=0.2)
    jw = raw_to_jw(raw)
    recovered = jw_to_raw(jw)
    np.testing.assert_allclose(
        [raw.a, raw.b, raw.rho, raw.m, raw.sigma],
        [recovered.a, recovered.b, recovered.rho, recovered.m, recovered.sigma],
        rtol=1e-9,
        atol=1e-9,
    )


def test_raw_tuple_to_jw_round_trip():
    raw_tuple: RawSVI = (0.03, 0.18, -0.2, 0.05, 0.25)
    jw: JWParams = raw_to_jw(raw_tuple)
    recovered = jw_to_raw(jw)
    np.testing.assert_allclose(
        raw_tuple,
        (recovered.a, recovered.b, recovered.rho, recovered.m, recovered.sigma),
        rtol=1e-9,
        atol=1e-9,
    )


def test_objective_gradient_matches_numeric():
    params = make_sample_params()
    maturity = 0.5
    k = np.linspace(-0.4, 0.4, 21)
    total_var = svi_total_variance(k, params)
    options = merge_svi_options(
        {
            "global_solver": "none",
            "n_starts": 0,
            "envelope_weight": 0.0,
        }
    )

    weights, _, _ = _vega_based_weights(
        k,
        total_var,
        maturity,
        options.weighting_mode,
        options.weight_cap,
        None,
    )
    callspread_step = _adaptive_call_spread_step(k, maturity, options)
    huber_delta = _compute_huber_delta(total_var, options, maturity_years=maturity)
    diagnostic_grid = np.linspace(
        k.min() - options.diagnostic_grid_pad,
        k.max() + options.diagnostic_grid_pad,
        options.diagnostic_grid_points,
    )
    context = _CallSpreadContext.build(diagnostic_grid, callspread_step, maturity)

    def objective_with_grad(vec: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            trial = SVIParameters(*vec)
        except ValueError:
            return 1e9, np.zeros(5, dtype=float)

        (_, w, dw, _, _, _, _) = _svi_total_variance_with_derivatives(k, trial)

        residual = w - total_var
        abs_residual = np.abs(residual)
        mask = abs_residual <= huber_delta
        quadratic = 0.5 * residual**2
        linear = huber_delta * (abs_residual - 0.5 * huber_delta)
        huber_loss = np.where(mask, quadratic, linear)
        base = float(np.sum(weights * huber_loss))

        huber_grad = weights * np.where(mask, residual, huber_delta * np.sign(residual))
        grad = np.sum(huber_grad[:, None] * dw, axis=0)

        penalty_val, penalty_grad = _penalty_terms(
            vec, options.regularisation, return_grad=True
        )
        value = base + penalty_val
        grad = grad + penalty_grad

        butterfly_val, butterfly_grad = _butterfly_penalty(
            trial, diagnostic_grid, options.butterfly_weight, return_grad=True
        )
        call_val, call_grad = _call_spread_penalty(
            trial,
            context,
            options.callspread_weight,
            return_grad=True,
        )

        value += butterfly_val + call_val
        grad += butterfly_grad + call_grad
        return value, grad

    initial = _initial_guess(k, total_var)
    test_vec = initial + np.array([0.02, -0.03, 0.04, -0.01, 0.05])
    _, analytic_grad = objective_with_grad(test_vec)

    eps = np.sqrt(np.finfo(float).eps)
    numeric_grad = np.zeros_like(test_vec)
    for idx in range(test_vec.size):
        step = eps * max(1.0, abs(test_vec[idx]))
        forward = test_vec.copy()
        forward[idx] += step
        backward = test_vec.copy()
        backward[idx] -= step
        f_plus = objective_with_grad(forward)[0]
        f_minus = objective_with_grad(backward)[0]
        numeric_grad[idx] = (f_plus - f_minus) / (2.0 * step)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-6, atol=1e-8)
