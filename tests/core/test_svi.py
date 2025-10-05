from __future__ import annotations

import numpy as np
import pytest

from oipd.core.svi import (
    SVIParameters,
    from_total_variance,
    g_function,
    log_moneyness,
    min_g_on_grid,
    svi_minimum_location,
    svi_minimum_variance,
    svi_total_variance,
    to_total_variance,
)


def make_sample_params() -> SVIParameters:
    # Representative arbitrage-free parameters
    return SVIParameters(a=0.04, b=0.2, rho=-0.4, m=-0.1, sigma=0.3)


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

