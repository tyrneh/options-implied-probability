"""Utilities to rebuild SVI smiles and RND slices from stored parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import interpolate

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
from oipd.core.vol_surface_fitting.shared.ssvi import SSVISurfaceParams, ssvi_total_variance


@dataclass(frozen=True)
class RebuiltSlice:
    """Container holding reconstructed smile and density information."""

    vol_curve: Callable[[Iterable[float] | np.ndarray], np.ndarray]
    data: pd.DataFrame


@dataclass
class RebuiltSurface:
    """Reconstructed SSVI surface capable of interpolating maturities."""

    theta_interpolator: interpolate.PchipInterpolator
    rho: float
    eta: float
    gamma: float
    alpha: float
    risk_free_rate: float
    forward_map: dict[float, float]
    strike_grids: dict[float, np.ndarray]
    _cache: dict[float, RebuiltSlice]

    def available_maturities(self) -> tuple[float, ...]:
        """Return maturities for which parameters were provided."""

        return tuple(sorted(self.forward_map))

    def slice(
        self,
        maturity: float,
        *,
        strike_grid: Sequence[float] | None = None,
    ) -> RebuiltSlice:
        """Return a reconstructed slice at an arbitrary maturity.

        Args:
            maturity: float
                Target year-fraction maturity.
            strike_grid: optional sequence of strikes. When omitted the helper
                reuses the stored grid for matching maturities or falls back to
                a symmetric 50–150% moneyness grid.

        Returns:
            RebuiltSlice: Reconstructed smile and density for ``maturity``.
        """

        t = float(maturity)
        if strike_grid is None and t in self._cache:
            return self._cache[t]

        slice_obj = self._build_slice(t, strike_grid=strike_grid)
        if strike_grid is None:
            self._cache[t] = slice_obj
        return slice_obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_slice(
        self,
        maturity: float,
        *,
        strike_grid: Sequence[float] | None,
    ) -> RebuiltSlice:
        theta_t = float(self.theta_interpolator(maturity))
        theta_t = max(theta_t, 1e-6)
        forward = self._infer_forward(maturity)

        strikes = self._resolve_strike_grid(maturity, forward, strike_grid)

        def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
            eval_array = np.asarray(eval_strikes, dtype=float)
            log_mny = log_moneyness(eval_array, forward)
            total_var = ssvi_total_variance(log_mny, theta_t, self.rho, self.eta, self.gamma)
            if self.alpha:
                total_var = total_var + float(self.alpha) * maturity
            return np.sqrt(np.maximum(total_var / max(maturity, 1e-8), 1e-12))

        implied_vol = vol_curve(strikes)
        vol_curve.grid = (strikes.copy(), implied_vol.copy())  # type: ignore[attr-defined]

        strike_pricing, call_prices = price_curve_from_iv(
            vol_curve,
            forward,
            strike_grid=strikes,
            days_to_expiry=self._days_from_maturity(maturity),
            risk_free_rate=self.risk_free_rate,
            pricing_engine="black76",
        )

        prices, pdf = pdf_from_price_curve(
            strike_pricing,
            call_prices,
            risk_free_rate=self.risk_free_rate,
            days_to_expiry=self._days_from_maturity(maturity),
        )
        _, cdf = calculate_cdf_from_pdf(prices, pdf)

        data = pd.DataFrame(
            {
                "strike": prices,
                "fitted_vol": vol_curve(prices),
                "call_price": np.interp(prices, strike_pricing, call_prices),
                "pdf": pdf,
                "cdf": cdf,
                "maturity": maturity,
            }
        )

        return RebuiltSlice(vol_curve=vol_curve, data=data)

    def _infer_forward(self, maturity: float) -> float:
        if not self.forward_map:
            raise ValueError("Forward map is empty; cannot infer forward level")

        items = sorted(self.forward_map.items())
        for t, fwd in items:
            if abs(t - maturity) <= 1e-6:
                return float(fwd)
        if maturity <= items[0][0]:
            return float(items[0][1])
        if maturity >= items[-1][0]:
            return float(items[-1][1])
        for (t0, f0), (t1, f1) in zip(items[:-1], items[1:]):
            if t0 <= maturity <= t1:
                weight = (maturity - t0) / (t1 - t0)
                return float((1.0 - weight) * f0 + weight * f1)
        return float(items[-1][1])

    def _resolve_strike_grid(
        self,
        maturity: float,
        forward: float,
        strike_grid: Sequence[float] | None,
    ) -> np.ndarray:
        if strike_grid is not None:
            return np.asarray(strike_grid, dtype=float)

        candidates = [
            grid
            for key, grid in self.strike_grids.items()
            if abs(key - maturity) <= 1e-6
        ]
        if candidates:
            return candidates[0].copy()

        strike_min = forward * 0.5
        strike_max = forward * 1.5
        return np.linspace(strike_min, strike_max, 201)

    @staticmethod
    def _days_from_maturity(maturity: float) -> int:
        days = int(round(float(maturity) * 365.0))
        return max(days, 1)


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

    rho = float(df.iloc[0]["rho"])
    eta = float(df.iloc[0]["eta"])
    gamma = float(df.iloc[0]["gamma"])
    alpha = float(df.iloc[0]["alpha"])

    maturities = df["maturity"].astype(float).to_numpy()
    theta_values = df["theta"].astype(float).to_numpy()

    params = SSVISurfaceParams(
        maturities=np.asarray(maturities, dtype=float),
        theta=np.asarray(theta_values, dtype=float),
        rho=float(rho),
        eta=float(eta),
        gamma=float(gamma),
        alpha=float(alpha),
    )
    theta_interp = params.interpolator()

    forward_lookup = {float(k): float(v) for k, v in forwards.items()}
    strike_lookup = {
        float(k): np.asarray(v, dtype=float)
        for k, v in (strike_grids or {}).items()
    }

    surface = RebuiltSurface(
        theta_interpolator=theta_interp,
        rho=float(rho),
        eta=float(eta),
        gamma=float(gamma),
        alpha=float(alpha),
        risk_free_rate=float(risk_free_rate),
        forward_map=forward_lookup,
        strike_grids=strike_lookup,
        _cache={},
    )

    for maturity in maturities:
        default_grid = strike_lookup.get(float(maturity))
        surface._cache[float(maturity)] = surface._build_slice(
            float(maturity),
            strike_grid=default_grid,
        )

    return surface


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
