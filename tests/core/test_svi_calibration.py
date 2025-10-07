from __future__ import annotations

import numpy as np
import pytest

from oipd.core.surface_fitting import fit_surface
from oipd.core.svi import (
    SVIParameters,
    from_total_variance,
    log_moneyness,
    svi_total_variance,
    to_total_variance,
)


def generate_synthetic_smile(params: SVIParameters, maturity_years: float, forward: float):
    k = np.linspace(-0.4, 0.4, 11)
    strikes = forward * np.exp(k)
    total_var = svi_total_variance(k, params)
    iv = from_total_variance(total_var, maturity_years)
    return strikes, iv


def test_svi_calibration_recovers_parameters():
    params_true = SVIParameters(a=0.04, b=0.2, rho=-0.4, m=-0.05, sigma=0.3)
    maturity = 0.5
    forward = 100.0
    strikes, iv = generate_synthetic_smile(params_true, maturity, forward)

    vol_curve = fit_surface(
        "svi",
        strikes=strikes,
        iv=iv,
        forward=forward,
        maturity_years=maturity,
    )

    recovered_iv = vol_curve(strikes)
    np.testing.assert_allclose(recovered_iv, iv, atol=5e-3)

    params_dict = vol_curve.params
    assert params_dict["method"] == "svi"
    params_est = SVIParameters(
        a=params_dict["a"],
        b=params_dict["b"],
        rho=params_dict["rho"],
        m=params_dict["m"],
        sigma=params_dict["sigma"],
    )
    recovered_total_var = svi_total_variance(log_moneyness(strikes, forward), params_est)
    target_total_var = to_total_variance(iv, maturity)
    np.testing.assert_allclose(recovered_total_var, target_total_var, atol=5e-3)


def test_svi_calibration_with_noise():
    params_true = SVIParameters(a=0.03, b=0.15, rho=0.2, m=0.0, sigma=0.25)
    maturity = 0.25
    forward = 50.0
    strikes, iv = generate_synthetic_smile(params_true, maturity, forward)
    rng = np.random.default_rng(42)
    noisy_iv = iv + rng.normal(scale=1e-3, size=iv.shape)

    vol_curve = fit_surface(
        "svi",
        strikes=strikes,
        iv=noisy_iv,
        forward=forward,
        maturity_years=maturity,
    )

    recovered = vol_curve(strikes)
    np.testing.assert_allclose(recovered, iv, atol=5e-3)


def test_svi_requires_forward_and_maturity():
    params_true = SVIParameters(a=0.03, b=0.2, rho=0.0, m=0.0, sigma=0.3)
    maturity = 0.5
    forward = 80.0
    strikes, iv = generate_synthetic_smile(params_true, maturity, forward)
    with pytest.raises(ValueError):
        fit_surface("svi", strikes=strikes, iv=iv)


def test_svi_insufficient_points():
    forward = 100.0
    maturity = 0.5
    strikes = np.linspace(90, 110, 4)
    iv = np.linspace(0.2, 0.25, 4)
    with pytest.raises(ValueError):
        fit_surface(
            "svi",
            strikes=strikes,
            iv=iv,
            forward=forward,
            maturity_years=maturity,
        )
