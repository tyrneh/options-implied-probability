"""Utilities to rebuild SVI smiles and RND slices from stored parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from oipd.core.probability_density_conversion import (
    calculate_cdf_from_pdf,
    pdf_from_price_curve,
    price_curve_from_iv,
)
from oipd.core.vol_surface_fitting.shared.svi import (
    SVIParameters,
    from_total_variance,
    log_moneyness,
    svi_total_variance,
)
from oipd.core.vol_surface_fitting.shared.ssvi import ssvi_total_variance


@dataclass(frozen=True)
class RebuiltSlice:
    """Container holding reconstructed smile and density information."""

    vol_curve: Callable[[Iterable[float] | np.ndarray], np.ndarray]
    data: pd.DataFrame


@dataclass(frozen=True)
class RebuiltSurface:
    """Bundle of reconstructed slices keyed by maturity (years)."""

    slices: dict[float, RebuiltSlice]

    def slice(self, maturity: float) -> RebuiltSlice:
        """Return the reconstructed slice for a maturity.

        Args:
            maturity: float
                Target maturity expressed in year fractions.

        Returns:
            RebuiltSlice: Reconstructed smile and density for ``maturity``.

        Raises:
            KeyError: If the requested maturity is not available.
        """
        try:
            return self.slices[maturity]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"No reconstructed slice for maturity {maturity!r}") from exc


def rebuild_slice_from_svi(
    svi_params: Mapping[str, float],
    *,
    forward_price: float,
    days_to_expiry: int,
    risk_free_rate: float,
    strike_grid: Sequence[float] | None = None,
) -> RebuiltSlice:
    """Rebuild a volatility smile and RND slice from stored SVI parameters.

    Args:
        svi_params: Mapping[str, float]
            Mapping of raw SVI coefficients ``a``, ``b``, ``rho``, ``m``, and
            ``sigma`` as returned by :meth:`oipd.pipelines.rnd_slice.RND.svi_params`.
        forward_price: float
            Forward level used during calibration.
        days_to_expiry: int
            Number of days to expiry for the slice.
        risk_free_rate: float
            Continuous risk-free interest rate used for pricing.
        strike_grid: Sequence[float], optional
            Strike grid on which to evaluate the smile. When omitted a symmetric
            grid around ``forward_price`` is generated.

    Returns:
        RebuiltSlice: Object containing the analytic volatility curve callable
        and a DataFrame with columns ``strike``, ``fitted_vol``, ``call_price``,
        ``pdf``, and ``cdf``.

    Raises:
        ValueError: If the provided parameters are incomplete or invalid.
    """

    required = {"a", "b", "rho", "m", "sigma"}
    missing = required.difference(svi_params)
    if missing:
        raise ValueError(f"SVI parameters missing required keys: {sorted(missing)}")

    maturity_years = float(days_to_expiry) / 365.0
    if maturity_years <= 0.0:
        raise ValueError("days_to_expiry must be positive")
    if forward_price <= 0.0:
        raise ValueError("forward_price must be positive")

    params = SVIParameters(
        float(svi_params["a"]),
        float(svi_params["b"]),
        float(svi_params["rho"]),
        float(svi_params["m"]),
        float(svi_params["sigma"]),
    )

    if strike_grid is None:
        strike_min = forward_price * 0.5
        strike_max = forward_price * 1.5
        strike_grid = np.linspace(strike_min, strike_max, 201)

    strike_array = np.asarray(list(strike_grid), dtype=float)
    if strike_array.ndim != 1 or strike_array.size == 0:
        raise ValueError("strike_grid must be a one-dimensional sequence of strikes")

    def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
        """Evaluate the implied-volatility smile on the requested strikes.

        Args:
            eval_strikes: Iterable[float] | ndarray
                Strike levels at which to compute implied volatility.

        Returns:
            ndarray: Implied volatility values corresponding to ``eval_strikes``.
        """

        eval_array = np.asarray(eval_strikes, dtype=float)
        log_mny = log_moneyness(eval_array, forward_price)
        total_variance = svi_total_variance(log_mny, params)
        return from_total_variance(total_variance, maturity_years)

    implied_vol = vol_curve(strike_array)
    vol_curve.grid = (strike_array.copy(), implied_vol.copy())  # type: ignore[attr-defined]

    strike_pricing, call_prices = price_curve_from_iv(
        vol_curve,
        forward_price,
        strike_grid=strike_array,
        days_to_expiry=int(days_to_expiry),
        risk_free_rate=float(risk_free_rate),
        pricing_engine="black76",
    )

    min_strike = float(np.min(strike_array))
    max_strike = float(np.max(strike_array))

    prices, pdf = pdf_from_price_curve(
        strike_pricing,
        call_prices,
        risk_free_rate=float(risk_free_rate),
        days_to_expiry=int(days_to_expiry),
        min_strike=min_strike,
        max_strike=max_strike,
    )
    _, cdf = calculate_cdf_from_pdf(prices, pdf)

    data = pd.DataFrame(
        {
            "strike": prices,
            "fitted_vol": vol_curve(prices),
            "call_price": np.interp(prices, strike_pricing, call_prices),
            "pdf": pdf,
            "cdf": cdf,
        }
    )

    return RebuiltSlice(vol_curve=vol_curve, data=data)


def rebuild_surface_from_ssvi(
    ssvi_params: pd.DataFrame | Mapping[str, Sequence[float]],
    *,
    forwards: Mapping[float, float],
    risk_free_rate: float,
    strike_grids: Mapping[float, Sequence[float]] | None = None,
) -> RebuiltSurface:
    """Rebuild per-maturity slices from stored SSVI parameters.

    Args:
        ssvi_params: pandas.DataFrame | Mapping[str, Sequence[float]]
            Table produced by :meth:`RNDSurface.ssvi_params` or an equivalent
            mapping containing the columns ``maturity``, ``theta``,
            ``days_to_expiry``, ``rho``, ``eta``, ``gamma``, and ``alpha``.
        forwards: Mapping[float, float]
            Dictionary mapping maturity (years) to the forward level used for
            pricing each slice during calibration.
        risk_free_rate: float
            Continuous risk-free rate assumed across maturities.
        strike_grids: Mapping[float, Sequence[float]] | None
            Optional per-maturity strike grids. When omitted, each slice uses a
            symmetric grid spanning 50%–150% of its forward.

    Returns:
        RebuiltSurface: Bundle containing reconstructed slices keyed by maturity.
    """

    if isinstance(ssvi_params, pd.DataFrame):
        df = ssvi_params.copy()
    else:
        df = pd.DataFrame(ssvi_params)

    required_columns = {"maturity", "theta", "days_to_expiry", "rho", "eta", "gamma", "alpha"}
    missing_cols = required_columns.difference(df.columns)
    if missing_cols:
        raise ValueError(f"SSVI params missing required columns: {sorted(missing_cols)}")

    slices: dict[float, RebuiltSlice] = {}
    rho = float(df.iloc[0]["rho"])
    eta = float(df.iloc[0]["eta"])
    gamma = float(df.iloc[0]["gamma"])
    alpha = float(df.iloc[0]["alpha"])

    forward_lookup = {float(k): float(v) for k, v in forwards.items()}
    strike_lookup = {float(k): tuple(map(float, v)) for k, v in (strike_grids or {}).items()}

    for _, row in df.iterrows():
        maturity = float(row["maturity"])
        theta_val = float(row["theta"])
        days = int(row["days_to_expiry"])

        forward = forward_lookup.get(maturity)
        if forward is None:
            forward = _match_with_tolerance(maturity, forward_lookup)
        forward_val = float(forward)

        strikes = strike_lookup.get(maturity)
        if strikes is None:
            strike_min = forward_val * 0.5
            strike_max = forward_val * 1.5
            strikes = tuple(np.linspace(strike_min, strike_max, 201))

        def _vol_curve(
            eval_strikes: Iterable[float] | np.ndarray,
            *,
            _theta: float = theta_val,
            _forward: float = forward_val,
            _maturity: float = maturity,
            _rho: float = rho,
            _eta: float = eta,
            _gamma: float = gamma,
            _alpha: float = alpha,
        ) -> np.ndarray:
            eval_array = np.asarray(eval_strikes, dtype=float)
            log_mny = log_moneyness(eval_array, _forward)
            total_var = ssvi_total_variance(log_mny, _theta, _rho, _eta, _gamma)
            if _alpha:
                total_var = total_var + float(_alpha) * _maturity
            return np.sqrt(np.maximum(total_var / max(_maturity, 1e-8), 1e-12))

        implied_vol = _vol_curve(strikes)
        _vol_curve.grid = (np.asarray(strikes, dtype=float).copy(), implied_vol.copy())  # type: ignore[attr-defined]

        strike_arr = np.asarray(strikes, dtype=float)
        strike_pricing, call_prices = price_curve_from_iv(
            _vol_curve,
            forward_val,
            strike_grid=strike_arr,
            days_to_expiry=days,
            risk_free_rate=float(risk_free_rate),
            pricing_engine="black76",
        )

        prices, pdf = pdf_from_price_curve(
            strike_pricing,
            call_prices,
            risk_free_rate=float(risk_free_rate),
            days_to_expiry=days,
        )
        _, cdf = calculate_cdf_from_pdf(prices, pdf)

        data = pd.DataFrame(
            {
                "strike": prices,
                "fitted_vol": _vol_curve(prices),
                "call_price": np.interp(prices, strike_pricing, call_prices),
                "pdf": pdf,
                "cdf": cdf,
                "maturity": maturity,
            }
        )

        slices[maturity] = RebuiltSlice(vol_curve=_vol_curve, data=data)

    return RebuiltSurface(slices)


def _match_with_tolerance(target: float, table: Mapping[float, float], tol: float = 1e-6) -> float:
    """Return table value whose key matches ``target`` within tolerance.

    Args:
        target: float
            Target key to locate.
        table: Mapping[float, float]
            Mapping of keys to values.
        tol: float, default 1e-6
            Absolute tolerance applied when matching keys.

    Returns:
        float: Value corresponding to the best matching key.

    Raises:
        KeyError: If no key matches ``target`` within ``tol``.
    """
    for key, value in table.items():
        if abs(key - target) <= tol:
            return value
    raise KeyError(f"No forward level available for maturity {target!r}")


__all__ = [
    "RebuiltSlice",
    "RebuiltSurface",
    "rebuild_slice_from_svi",
    "rebuild_surface_from_ssvi",
]
