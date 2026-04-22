"""Utilities to rebuild SVI smiles and RND slices from stored parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import interpolate
from oipd.core.maturity import convert_days_to_years

from oipd.core.probability_density_conversion import (
    cdf_from_price_curve,
    pdf_from_price_curve,
    price_curve_from_iv,
)
from oipd.core.vol_surface_fitting.shared.svi import (
    SVIParameters,
    from_total_variance,
    log_moneyness,
    svi_total_variance,
)
from oipd.core.vol_surface_fitting.shared.ssvi import (
    SSVISurfaceParams,
    ssvi_total_variance,
)

REMOVED_DAY_COUNT_COLUMN = "days" "_to_expiry"
CdfViolationPolicy = Literal["warn", "raise"]


@dataclass(frozen=True)
class RebuiltSlice:
    """Container holding reconstructed smile and density information."""

    vol_curve: Callable[[Iterable[float] | np.ndarray], np.ndarray]
    data: pd.DataFrame


def _cdf_diagnostic_columns(
    diagnostics: Mapping[str, float | int | bool | str],
) -> dict[str, float | int | bool | str]:
    """Return CDF repair diagnostics as scalar DataFrame columns.

    Args:
        diagnostics: Direct-CDF diagnostics emitted by the core converter.

    Returns:
        dict[str, float | int | bool | str]: Scalar diagnostics that pandas can
        broadcast across a rebuilt slice's rows.
    """
    return {
        "cdf_violation_policy": diagnostics["cdf_violation_policy"],
        "cdf_monotonicity_repair_applied": diagnostics[
            "cdf_monotonicity_repair_applied"
        ],
        "cdf_monotonicity_repair_tolerance": diagnostics[
            "cdf_monotonicity_repair_tolerance"
        ],
        "cdf_total_negative_variation_tolerance": diagnostics[
            "cdf_total_negative_variation_tolerance"
        ],
        "cdf_monotonicity_severity": diagnostics["cdf_monotonicity_severity"],
        "raw_cdf_is_monotone": diagnostics["raw_cdf_is_monotone"],
        "raw_cdf_negative_step_count": diagnostics["raw_cdf_negative_step_count"],
        "raw_cdf_max_negative_step": diagnostics["raw_cdf_max_negative_step"],
        "raw_cdf_total_negative_variation": diagnostics[
            "raw_cdf_total_negative_variation"
        ],
        "raw_cdf_worst_step_strike": diagnostics["raw_cdf_worst_step_strike"],
    }


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
    days_map: dict[float, int]
    time_days_map: dict[float, float]
    cdf_violation_policy: CdfViolationPolicy
    _cache: dict[float, RebuiltSlice]

    def available_time_to_expiry_days(self) -> tuple[float, ...]:
        """Return continuous day-equivalent maturities available in the snapshot."""
        return tuple(sorted(set(self.time_days_map.values())))

    def slice(
        self,
        *,
        time_to_expiry_days: float | None = None,
        time_to_expiry_years: float | None = None,
        strike_grid: Sequence[float] | None = None,
    ) -> RebuiltSlice:
        """Return a reconstructed slice at an arbitrary maturity.

        Args:
            time_to_expiry_days: Continuous day-equivalent maturity.
            time_to_expiry_years: Continuous year-fraction maturity.
            strike_grid: optional sequence of strikes. When omitted the helper
                reuses the stored grid for matching maturities or falls back to
                a symmetric 50–150% moneyness grid.

        Returns:
            RebuiltSlice: Reconstructed smile and density for the requested
            maturity.
        """
        maturity, time_days, calendar_days = _resolve_rebuild_maturity(
            time_to_expiry_days=time_to_expiry_days,
            time_to_expiry_years=time_to_expiry_years,
        )
        days_override = calendar_days

        if strike_grid is None and maturity in self._cache:
            return self._cache[maturity]

        slice_obj = self._build_slice(
            maturity,
            strike_grid=strike_grid,
            days_override=days_override,
        )
        if strike_grid is None:
            self._cache[maturity] = slice_obj
        return slice_obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_slice(
        self,
        maturity: float,
        *,
        strike_grid: Sequence[float] | None,
        days_override: int | None,
    ) -> RebuiltSlice:
        theta_t = float(self.theta_interpolator(maturity))
        theta_t = max(theta_t, 1e-6)
        forward = self._infer_forward(maturity)

        strikes = self._resolve_strike_grid(maturity, forward, strike_grid)

        def vol_curve(eval_strikes: Iterable[float] | np.ndarray) -> np.ndarray:
            eval_array = np.asarray(eval_strikes, dtype=float)
            log_mny = log_moneyness(eval_array, forward)
            total_var = ssvi_total_variance(
                log_mny, theta_t, self.rho, self.eta, self.gamma
            )
            if self.alpha:
                total_var = total_var + float(self.alpha) * maturity
            return np.sqrt(np.maximum(total_var / max(maturity, 1e-8), 1e-12))

        implied_vol = vol_curve(strikes)
        vol_curve.grid = (strikes.copy(), implied_vol.copy())  # type: ignore[attr-defined]

        calendar_days_to_expiry = (
            days_override
            if days_override is not None
            else self._days_from_maturity(maturity)
        )
        time_to_expiry_days = float(maturity * 365.0)

        strike_pricing, call_prices = price_curve_from_iv(
            vol_curve,
            forward,
            strike_grid=strikes,
            time_to_expiry_years=maturity,
            risk_free_rate=self.risk_free_rate,
            pricing_engine="black76",
        )

        prices, pdf = pdf_from_price_curve(
            strike_pricing,
            call_prices,
            risk_free_rate=self.risk_free_rate,
            time_to_expiry_years=maturity,
        )
        cdf_result = cdf_from_price_curve(
            strike_pricing,
            call_prices,
            risk_free_rate=self.risk_free_rate,
            time_to_expiry_years=maturity,
            min_strike=float(prices[0]),
            max_strike=float(prices[-1]),
            reference_price=forward,
            cdf_violation_policy=self.cdf_violation_policy,
        )
        cdf = cdf_result.cdf_values

        data = pd.DataFrame(
            {
                "strike": prices,
                "fitted_vol": vol_curve(prices),
                "call_price": np.interp(prices, strike_pricing, call_prices),
                "pdf": pdf,
                "cdf": cdf,
                "time_to_expiry_years": maturity,
                "time_to_expiry_days": time_to_expiry_days,
                "calendar_days_to_expiry": calendar_days_to_expiry,
                **_cdf_diagnostic_columns(cdf_result.diagnostics),
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
        return _calendar_days_from_time_days(float(maturity) * 365.0)


def _calendar_days_from_time_days(time_days: float) -> int:
    """Convert continuous day-equivalent maturity into deprecated day buckets.

    Args:
        time_days: Continuous ACT/365 day-equivalent maturity.

    Returns:
        int: Non-negative deprecated calendar-day-style bucket.
    """
    return max(int(np.floor(float(time_days) + 1e-12)), 0)


def _resolve_snapshot_maturity_years(df: pd.DataFrame) -> pd.Series:
    """Resolve the year-fraction maturity column from a reconstruction snapshot.

    Args:
        df: Snapshot DataFrame for SSVI reconstruction.

    Returns:
        pd.Series: Year-fraction maturities.

    Raises:
        ValueError: If the canonical year-fraction column is missing or if the
            removed legacy alias is still present.
    """
    if "time_to_expiry_years" not in df.columns:
        if "maturity" in df.columns:
            raise ValueError(
                "SSVI params no longer accept 'maturity'; rename it to "
                "'time_to_expiry_years'."
            )
        raise ValueError("SSVI params must include canonical 'time_to_expiry_years'.")
    if "maturity" in df.columns:
        raise ValueError(
            "SSVI params no longer accept 'maturity'; rename it to "
            "'time_to_expiry_years'."
        )
    return df["time_to_expiry_years"].astype(float)


def _resolve_rebuild_maturity(
    *,
    time_to_expiry_days: float | None = None,
    time_to_expiry_years: float | None = None,
) -> tuple[float, float, int]:
    """Resolve one public rebuild maturity input into canonical quantities.

    Args:
        time_to_expiry_days: Continuous day-equivalent maturity.
        time_to_expiry_years: Continuous year-fraction maturity.

    Returns:
        tuple[float, float, int]: Year fraction, continuous day-equivalent
        maturity, and integer calendar-day bucket.

    Raises:
        ValueError: If zero or multiple maturity inputs are provided, or if the
            resolved year fraction is non-positive.
    """
    canonical_inputs = [
        value is not None for value in (time_to_expiry_days, time_to_expiry_years)
    ]
    if sum(canonical_inputs) > 1:
        raise ValueError(
            "Provide at most one of time_to_expiry_days or time_to_expiry_years."
        )

    if time_to_expiry_years is not None:
        maturity_years = float(time_to_expiry_years)
        time_days = maturity_years * 365.0
    elif time_to_expiry_days is not None:
        time_days = float(time_to_expiry_days)
        maturity_years = convert_days_to_years(time_days)
    else:
        raise ValueError(
            "Provide exactly one of time_to_expiry_years or time_to_expiry_days."
        )

    if maturity_years <= 0.0:
        raise ValueError("time_to_expiry_years must be positive")

    calendar_days = _calendar_days_from_time_days(time_days)
    return float(maturity_years), float(time_days), calendar_days


def rebuild_slice_from_svi(
    svi_params: Mapping[str, float],
    *,
    forward_price: float,
    time_to_expiry_days: float | None = None,
    time_to_expiry_years: float | None = None,
    risk_free_rate: float,
    strike_grid: Sequence[float] | None = None,
    cdf_violation_policy: CdfViolationPolicy = "warn",
) -> RebuiltSlice:
    """Rebuild a volatility smile and RND slice from stored SVI parameters.

    Args:
        svi_params: Mapping[str, float]
            Mapping of raw SVI coefficients ``a``, ``b``, ``rho``, ``m``, and
            ``sigma`` as returned by :meth:`oipd.pipelines.rnd_slice.RND.svi_params`.
        forward_price: float
            Forward level used during calibration.
        time_to_expiry_days: Continuous day-equivalent maturity.
        time_to_expiry_years: Continuous year-fraction maturity.
        risk_free_rate: float
            Continuous risk-free interest rate used for pricing.
        strike_grid: Sequence[float], optional
            Strike grid on which to evaluate the smile. When omitted a symmetric
            grid around ``forward_price`` is generated.
        cdf_violation_policy: Direct-CDF monotonicity violation policy.
            ``"warn"`` repairs and warns; ``"raise"`` fails on material
            violations.

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

    maturity_years, time_days, calendar_days = _resolve_rebuild_maturity(
        time_to_expiry_days=time_to_expiry_days,
        time_to_expiry_years=time_to_expiry_years,
    )
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
        time_to_expiry_years=maturity_years,
        risk_free_rate=float(risk_free_rate),
        pricing_engine="black76",
    )

    min_strike = float(np.min(strike_array))
    max_strike = float(np.max(strike_array))

    prices, pdf = pdf_from_price_curve(
        strike_pricing,
        call_prices,
        risk_free_rate=float(risk_free_rate),
        time_to_expiry_years=maturity_years,
        min_strike=min_strike,
        max_strike=max_strike,
    )
    cdf_result = cdf_from_price_curve(
        strike_pricing,
        call_prices,
        risk_free_rate=float(risk_free_rate),
        time_to_expiry_years=maturity_years,
        min_strike=float(prices[0]),
        max_strike=float(prices[-1]),
        reference_price=forward_price,
        cdf_violation_policy=cdf_violation_policy,
    )
    cdf = cdf_result.cdf_values

    data = pd.DataFrame(
        {
            "strike": prices,
            "fitted_vol": vol_curve(prices),
            "call_price": np.interp(prices, strike_pricing, call_prices),
            "pdf": pdf,
            "cdf": cdf,
            "time_to_expiry_years": maturity_years,
            "time_to_expiry_days": time_days,
            "calendar_days_to_expiry": calendar_days,
            **_cdf_diagnostic_columns(cdf_result.diagnostics),
        }
    )

    return RebuiltSlice(vol_curve=vol_curve, data=data)


def rebuild_surface_from_ssvi(
    ssvi_params: pd.DataFrame | Mapping[str, Sequence[float]],
    *,
    forward_prices: Mapping[float, float],
    risk_free_rate: float,
    strike_grids: Mapping[float, Sequence[float]] | None = None,
    cdf_violation_policy: CdfViolationPolicy = "warn",
) -> RebuiltSurface:
    """Rebuild per-maturity slices from stored SSVI parameters.

    Args:
        ssvi_params: pandas.DataFrame | Mapping[str, Sequence[float]]
            Table produced by :meth:`RNDSurface.ssvi_params` or an equivalent
            mapping containing canonical ``time_to_expiry_years``, plus
            ``theta``, optional ``time_to_expiry_days``, and the SSVI
            parameters ``rho``, ``eta``, ``gamma``, and ``alpha``.
        forward_prices: Mapping[float, float]
            Dictionary mapping maturity (years) to the forward level used for
            pricing each slice during calibration.
        risk_free_rate: float
            Continuous risk-free rate assumed across maturities.
        strike_grids: Mapping[float, Sequence[float]] | None
            Optional per-maturity strike grids. When omitted, each slice uses a
            symmetric grid spanning 50%–150% of its forward.
        cdf_violation_policy: Direct-CDF monotonicity violation policy used for
            each rebuilt slice. ``"warn"`` repairs and warns; ``"raise"``
            fails on material violations.

    Returns:
        RebuiltSurface: Bundle containing reconstructed slices keyed by maturity.
    """

    if isinstance(ssvi_params, pd.DataFrame):
        df = ssvi_params.copy()
    else:
        df = pd.DataFrame(ssvi_params)

    required_columns = {"theta", "rho", "eta", "gamma", "alpha"}
    missing_cols = required_columns.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"SSVI params missing required columns: {sorted(missing_cols)}"
        )

    maturity_years_series = _resolve_snapshot_maturity_years(df)

    rho = float(df.iloc[0]["rho"])
    eta = float(df.iloc[0]["eta"])
    gamma = float(df.iloc[0]["gamma"])
    alpha = float(df.iloc[0]["alpha"])

    if REMOVED_DAY_COUNT_COLUMN in df.columns:
        raise ValueError(
            f"SSVI params no longer accept '{REMOVED_DAY_COUNT_COLUMN}'; rename it to "
            "'time_to_expiry_days'."
        )

    maturities = maturity_years_series.to_numpy(dtype=float)
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

    forward_lookup = {float(k): float(v) for k, v in forward_prices.items()}
    strike_lookup = {
        float(k): np.asarray(v, dtype=float) for k, v in (strike_grids or {}).items()
    }
    if "time_to_expiry_days" in df.columns:
        time_days_lookup = {
            float(row_maturity): float(row_days)
            for row_maturity, row_days in zip(
                maturity_years_series, df["time_to_expiry_days"].astype(float)
            )
        }
        days_lookup = {
            maturity: _calendar_days_from_time_days(time_days)
            for maturity, time_days in time_days_lookup.items()
        }
    else:
        time_days_lookup = {
            float(row_maturity): float(row_maturity) * 365.0
            for row_maturity in maturity_years_series
        }
        days_lookup = {
            maturity: _calendar_days_from_time_days(time_days)
            for maturity, time_days in time_days_lookup.items()
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
        days_map=days_lookup,
        time_days_map=time_days_lookup,
        cdf_violation_policy=cdf_violation_policy,
        _cache={},
    )

    for maturity in maturities:
        default_grid = strike_lookup.get(float(maturity))
        maturity_float = float(maturity)
        override_days = days_lookup.get(maturity_float)
        if override_days is None:
            override_days = _calendar_days_from_time_days(maturity_float * 365.0)
        surface._cache[maturity_float] = surface._build_slice(
            maturity_float,
            strike_grid=default_grid,
            days_override=override_days,
        )

    return surface


def _match_with_tolerance(
    target: float, table: Mapping[float, float], tol: float = 1e-6
) -> float:
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
