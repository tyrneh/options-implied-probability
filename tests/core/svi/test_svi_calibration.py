from __future__ import annotations

import numpy as np
import pytest

from dataclasses import astuple

from oipd.core.vol_surface_fitting import fit_surface
from oipd.core.vol_surface_fitting.shared.svi import (
    SVICalibrationDiagnostics,
    SVIParameters,
    calibrate_svi_parameters,
    from_total_variance,
    svi_options,
    svi_total_variance,
)


def generate_synthetic_smile(
    params: SVIParameters, maturity_years: float, forward: float
):
    k = np.linspace(-0.4, 0.4, 11)
    strikes = forward * np.exp(k)
    total_var = svi_total_variance(k, params)
    iv = from_total_variance(total_var, maturity_years)
    return strikes, iv


def _synthetic_svi_slice(point_count: int = 11):
    params_true = SVIParameters(a=0.035, b=0.21, rho=-0.15, m=0.01, sigma=0.27)
    maturity = 0.6
    k = np.linspace(-0.5, 0.5, point_count)
    total_var = svi_total_variance(k, params_true)
    base_iv = from_total_variance(total_var, maturity)
    return k, total_var, maturity, base_iv


def _valid_bid_ask(base_iv: np.ndarray, half_spread: float = 0.012):
    return base_iv - half_spread, base_iv + half_spread


def _bid_ask_with_valid_spreads(
    base_iv: np.ndarray,
    valid_count: int,
    half_spread: float = 0.012,
):
    bid_iv = base_iv - 0.01
    ask_iv = base_iv - 0.02
    bid_iv[:valid_count] = base_iv[:valid_count] - half_spread
    ask_iv[:valid_count] = base_iv[:valid_count] + half_spread
    return bid_iv, ask_iv


def _fast_weighting_options(**overrides):
    options = {"global_solver": "none", "n_starts": 0, "random_seed": 7}
    options.update(overrides)
    return options


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
        options=svi_options(huber_delta=1e-6, random_seed=123),
    )

    recovered_iv = vol_curve(strikes)
    max_abs_diff = float(np.max(np.abs(recovered_iv - iv)))
    assert max_abs_diff < 0.12

    diagnostics = vol_curve.diagnostics
    assert isinstance(diagnostics, SVICalibrationDiagnostics)
    assert diagnostics.rmse_unweighted < 0.03
    assert diagnostics.min_g > -1e-6
    assert diagnostics.random_seed == 123


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
        options=svi_options(huber_delta=1e-6, random_seed=123),
    )

    recovered = vol_curve(strikes)
    assert float(np.max(np.abs(recovered - iv))) < 0.12

    diagnostics = vol_curve.diagnostics
    assert diagnostics.rmse_unweighted < 0.05
    assert diagnostics.random_seed == 123


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


def test_calibration_multistart_deterministic():
    params_true = SVIParameters(a=0.04, b=0.25, rho=-0.3, m=-0.05, sigma=0.2)
    maturity = 0.5
    k = np.linspace(-0.4, 0.4, 9)
    total_var = svi_total_variance(k, params_true)

    params_one, diag_one = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        {"n_starts": 5, "random_seed": 7},
    )
    params_two, diag_two = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        {"n_starts": 5, "random_seed": 7},
    )

    np.testing.assert_allclose(
        np.array(astuple(params_one)), np.array(astuple(params_two))
    )
    assert diag_one.objective == pytest.approx(diag_two.objective, rel=1e-12)
    assert diag_one.chosen_start_index == diag_two.chosen_start_index


def test_calibration_global_solver_runs():
    params_true = SVIParameters(a=0.03, b=0.2, rho=0.1, m=0.02, sigma=0.25)
    maturity = 0.25
    k = np.linspace(-0.5, 0.5, 11)
    total_var = svi_total_variance(k, params_true)

    params, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        {
            "global_solver": "de",
            "global_max_iter": 1,
            "random_seed": 123,
            "max_iter": 50,
        },
    )

    assert isinstance(params, SVIParameters)
    assert diagnostics.global_solver == "de"
    assert diagnostics.global_status == "success"
    assert (diagnostics.global_iterations or 0) >= 0
    assert diagnostics.random_seed == 123


def test_calibration_reports_weighting_stats():
    params_true = SVIParameters(a=0.025, b=0.18, rho=-0.25, m=0.01, sigma=0.22)
    maturity = 0.75
    k = np.linspace(-0.6, 0.6, 15)
    total_var = svi_total_variance(k, params_true)

    params, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        None,
    )

    assert isinstance(params, SVIParameters)
    assert diagnostics.weighting_mode == "vega+spread"
    assert diagnostics.weights_max > diagnostics.weights_min
    assert diagnostics.weights_min > 0
    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_measurement_used is False
    assert diagnostics.callspread_step > 0
    assert diagnostics.rmse_unweighted >= 0
    assert diagnostics.rmse_weighted >= 0
    assert diagnostics.random_seed == 1


def test_calibration_huber_option_recorded():
    params_true = SVIParameters(a=0.04, b=0.3, rho=0.2, m=-0.02, sigma=0.28)
    maturity = 0.4
    k = np.linspace(-0.5, 0.5, 13)
    total_var = svi_total_variance(k, params_true)

    params, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        {"huber_delta": 5e-4},
    )

    assert isinstance(params, SVIParameters)
    assert diagnostics.huber_delta == pytest.approx(5e-4)
    assert diagnostics.random_seed == 1


def test_envelope_penalty_within_spread():
    params_true = SVIParameters(a=0.035, b=0.22, rho=-0.15, m=0.02, sigma=0.3)
    maturity = 0.6
    k = np.linspace(-0.5, 0.5, 17)
    total_var = svi_total_variance(k, params_true)
    base_iv = from_total_variance(total_var, maturity)
    bid_iv = base_iv - 0.02
    ask_iv = base_iv + 0.02

    params, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        None,
        bid_iv=bid_iv,
        ask_iv=ask_iv,
    )

    assert isinstance(params, SVIParameters)
    assert diagnostics.envelope_weight == pytest.approx(1e3)
    assert diagnostics.weights_measurement_used is True
    assert diagnostics.envelope_violations_pct == pytest.approx(0.0)
    assert diagnostics.random_seed == 1


def test_envelope_penalty_shape_mismatch_raises():
    params_true = SVIParameters(a=0.03, b=0.18, rho=0.1, m=0.01, sigma=0.25)
    maturity = 0.5
    k = np.linspace(-0.4, 0.4, 9)
    total_var = svi_total_variance(k, params_true)
    bad_bid = np.array([0.2, 0.21])

    with pytest.raises(ValueError):
        calibrate_svi_parameters(
            k,
            total_var,
            maturity,
            {},
            bid_iv=bad_bid,
        )


def test_ask_envelope_penalty_shape_mismatch_raises():
    params_true = SVIParameters(a=0.03, b=0.18, rho=0.1, m=0.01, sigma=0.25)
    maturity = 0.5
    k = np.linspace(-0.4, 0.4, 9)
    total_var = svi_total_variance(k, params_true)
    bad_ask = np.array([0.2, 0.21])

    with pytest.raises(ValueError):
        calibrate_svi_parameters(
            k,
            total_var,
            maturity,
            {},
            ask_iv=bad_ask,
        )


def test_volume_weighting_flagged():
    params_true = SVIParameters(a=0.05, b=0.25, rho=-0.2, m=0.0, sigma=0.28)
    maturity = 0.7
    k = np.linspace(-0.6, 0.6, 15)
    total_var = svi_total_variance(k, params_true)
    volumes = np.linspace(1.0, 3.0, k.size)

    params, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        None,
        volumes=volumes,
    )

    assert isinstance(params, SVIParameters)
    assert diagnostics.weights_volume_used is True
    assert diagnostics.weights_max >= diagnostics.weights_min
    assert (
        diagnostics.rmse_weighted >= diagnostics.rmse_unweighted
        or diagnostics.rmse_weighted >= 0
    )
    assert diagnostics.random_seed == 1


def test_default_weighting_prefers_reliable_bid_ask_over_volume():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _valid_bid_ask(base_iv)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weighting_mode == "vega+spread"
    assert diagnostics.weights_measurement_used is True
    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_auxiliary_source == "measurement"
    assert diagnostics.weights_fallback_reason == ""
    assert diagnostics.weights_bid_ask_valid_count == k.size
    assert diagnostics.weights_bid_ask_coverage == pytest.approx(1.0)
    assert diagnostics.weights_volume_valid_count == k.size


def test_spread_vega_alias_prefers_reliable_bid_ask_over_volume():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _valid_bid_ask(base_iv)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(weighting_mode="spread_vega"),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weights_measurement_used is True
    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_auxiliary_source == "measurement"
    assert diagnostics.weights_bid_ask_valid_count == k.size
    assert diagnostics.weights_bid_ask_coverage == pytest.approx(1.0)


def test_unknown_weighting_mode_prefers_reliable_bid_ask_over_volume():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _valid_bid_ask(base_iv)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(weighting_mode="typo_mode"),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weights_measurement_used is True
    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_auxiliary_source == "measurement"
    assert diagnostics.weights_fallback_reason == "unknown_weighting_mode"


def test_default_weighting_falls_back_to_volume_when_one_spread_is_valid():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _bid_ask_with_valid_spreads(base_iv, valid_count=1)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weights_volume_used is True
    assert diagnostics.weights_measurement_used is False
    assert diagnostics.weights_auxiliary_source == "volume"
    assert diagnostics.weights_fallback_reason == "measurement_insufficient"
    assert diagnostics.weights_bid_ask_valid_count == 1
    assert diagnostics.weights_bid_ask_coverage == pytest.approx(1 / k.size)
    assert diagnostics.weights_volume_valid_count == k.size


def test_default_weighting_falls_back_to_volume_when_bid_ask_coverage_is_low():
    k, total_var, maturity, base_iv = _synthetic_svi_slice(point_count=11)
    bid_iv, ask_iv = _bid_ask_with_valid_spreads(base_iv, valid_count=5)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weights_volume_used is True
    assert diagnostics.weights_measurement_used is False
    assert diagnostics.weights_auxiliary_source == "volume"
    assert diagnostics.weights_fallback_reason == "measurement_insufficient"
    assert diagnostics.weights_bid_ask_valid_count == 5
    assert diagnostics.weights_bid_ask_coverage == pytest.approx(5 / 11)
    assert diagnostics.weights_volume_valid_count == k.size


def test_measurement_mode_with_weak_bid_ask_does_not_fall_back_to_volume():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _bid_ask_with_valid_spreads(base_iv, valid_count=1)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(weighting_mode="measurement"),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_measurement_used is False
    assert diagnostics.weights_auxiliary_source == "none"
    assert diagnostics.weights_fallback_reason == "measurement_insufficient"
    assert diagnostics.weights_bid_ask_valid_count == 1
    assert diagnostics.weights_bid_ask_coverage == pytest.approx(1 / k.size)
    assert diagnostics.weights_volume_valid_count == k.size
    assert diagnostics.weights_min == pytest.approx(1.0)
    assert diagnostics.weights_max == pytest.approx(1.0)


def test_none_weighting_mode_ignores_bid_ask_and_volume():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _valid_bid_ask(base_iv)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(weighting_mode="none"),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_measurement_used is False
    assert diagnostics.weights_auxiliary_source == "none"
    assert diagnostics.weights_fallback_reason == "weighting_mode_none"
    assert diagnostics.weights_bid_ask_valid_count == k.size
    assert diagnostics.weights_volume_valid_count == k.size
    assert diagnostics.weights_min == pytest.approx(1.0)
    assert diagnostics.weights_max == pytest.approx(1.0)


def test_vega_weighting_mode_ignores_bid_ask_and_volume():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _valid_bid_ask(base_iv)
    volumes = np.linspace(1.0, 5.0, k.size)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(weighting_mode="vega"),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        volumes=volumes,
    )

    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_measurement_used is False
    assert diagnostics.weights_auxiliary_source == "none"
    assert diagnostics.weights_fallback_reason == "weighting_mode_vega"
    assert diagnostics.weights_bid_ask_valid_count == k.size
    assert diagnostics.weights_volume_valid_count == k.size


def test_near_zero_positive_spread_uses_floor_and_bounded_weights():
    k, total_var, maturity, base_iv = _synthetic_svi_slice()
    bid_iv, ask_iv = _valid_bid_ask(base_iv, half_spread=5e-9)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        _fast_weighting_options(weight_cap=7.5),
        bid_iv=bid_iv,
        ask_iv=ask_iv,
    )

    assert diagnostics.weights_measurement_used is True
    assert diagnostics.weights_volume_used is False
    assert diagnostics.weights_auxiliary_source == "measurement"
    assert diagnostics.weights_bid_ask_valid_count == k.size
    assert diagnostics.weights_bid_ask_coverage == pytest.approx(1.0)
    assert diagnostics.weights_min >= 1e-6
    assert diagnostics.weights_max <= 7.5
    assert np.isfinite(diagnostics.weights_min)
    assert np.isfinite(diagnostics.weights_max)


def test_volume_weighting_shape_mismatch():
    params_true = SVIParameters(a=0.03, b=0.18, rho=0.1, m=0.01, sigma=0.25)
    maturity = 0.5
    k = np.linspace(-0.4, 0.4, 9)
    total_var = svi_total_variance(k, params_true)
    with pytest.raises(ValueError):
        calibrate_svi_parameters(
            k,
            total_var,
            maturity,
            None,
            volumes=np.array([1.0, 2.0]),
        )


def test_huber_delta_scales_with_slice():
    params_true = SVIParameters(a=0.02, b=0.18, rho=-0.2, m=0.0, sigma=0.3)
    maturity = 1.0
    k = np.linspace(-0.5, 0.5, 13)
    total_var = svi_total_variance(k, params_true)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        None,
    )

    expected_scale = np.median(total_var)
    expected_delta = max(1e-4, 0.01 * expected_scale)
    assert diagnostics.huber_delta == pytest.approx(expected_delta, rel=1e-6)
    assert diagnostics.random_seed == 1


def test_huber_delta_respects_bid_ask_spread():
    params_true = SVIParameters(a=0.03, b=0.2, rho=-0.1, m=0.0, sigma=0.25)
    maturity = 0.75
    k = np.linspace(-0.4, 0.4, 11)
    total_var = svi_total_variance(k, params_true)
    base_iv = from_total_variance(total_var, maturity)
    bid_iv = base_iv - 0.015
    ask_iv = base_iv + 0.015

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        None,
        bid_iv=bid_iv,
        ask_iv=ask_iv,
    )

    tv_bid = np.square(bid_iv) * maturity
    tv_ask = np.square(ask_iv) * maturity
    measurement_scale = 0.5 * np.median(np.abs(tv_ask - tv_bid))

    assert diagnostics.huber_delta + 1e-12 >= measurement_scale
    assert diagnostics.weights_measurement_used is True


def test_callspread_step_adapts_to_spacing():
    params_true = SVIParameters(a=0.03, b=0.22, rho=0.1, m=0.05, sigma=0.4)
    maturity = 1.0
    k = np.linspace(-0.6, 0.6, 7)  # spacing 0.2
    total_var = svi_total_variance(k, params_true)

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        None,
    )

    # baseline spacing = 0.2 -> 0.5 * spacing * scale(=2) = 0.2
    assert diagnostics.callspread_step == pytest.approx(0.2, rel=1e-6)
    assert diagnostics.random_seed == 1


def test_qe_seed_origin_recorded():
    params_true = SVIParameters(a=0.025, b=0.21, rho=-0.15, m=-0.03, sigma=0.28)
    maturity = 0.8
    k = np.linspace(-0.5, 0.5, 11)
    total_var = svi_total_variance(k, params_true)
    # perturb the smile slightly to avoid duplicate seeds
    rng = np.random.default_rng(123)
    total_var = total_var * (1 + rng.normal(scale=1e-3, size=total_var.shape))

    _, diagnostics = calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        {"n_starts": 0, "random_seed": 42},
    )

    assert diagnostics.qe_seed_count >= 1
    assert any(record.start_origin == "qe" for record in diagnostics.trial_records)
    assert diagnostics.random_seed == 42


def test_calibration_emits_logging(caplog):
    params_true = SVIParameters(a=0.035, b=0.18, rho=-0.2, m=0.02, sigma=0.3)
    maturity = 0.6
    k = np.linspace(-0.4, 0.4, 9)
    total_var = svi_total_variance(k, params_true)

    caplog.set_level("INFO", logger="oipd.svi")

    calibrate_svi_parameters(
        k,
        total_var,
        maturity,
        {"random_seed": 314},
    )

    messages = [
        record.message for record in caplog.records if record.name == "oipd.svi"
    ]
    assert any("Starting SVI calibration" in message for message in messages)
    assert any("SVI calibration complete" in message for message in messages)
