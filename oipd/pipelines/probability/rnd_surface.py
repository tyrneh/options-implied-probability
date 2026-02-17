"""Stateless helpers for VolSurface-derived probability estimation."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from oipd.core.probability_density_conversion import (
    normalized_cdf_from_call_curve,
    pdf_and_cdf_from_normalized_cdf,
)
from oipd.core.utils import calculate_time_to_expiry, resolve_risk_free_rate
from oipd.market_inputs import Provenance, ResolvedMarket
from oipd.pricing import black76_call_price


def build_global_log_moneyness_grid(vol_surface: Any, *, points: int) -> np.ndarray:
    """Build a unified log-moneyness grid shared across all fitted expiries.

    Args:
        vol_surface: Fitted volatility surface interface object.
        points: Number of desired grid points.

    Returns:
        np.ndarray: Uniformly spaced log-moneyness grid.
    """
    k_min = np.inf
    k_max = -np.inf

    for expiry_timestamp in vol_surface.expiries:
        slice_data = vol_surface._model.get_slice(expiry_timestamp)
        metadata = slice_data.get("metadata", {})
        chain = slice_data.get("chain")
        forward_price = metadata.get("forward_price")

        if (
            chain is None
            or "strike" not in chain.columns
            or forward_price is None
            or float(forward_price) <= 0.0
        ):
            continue

        strikes = chain["strike"].to_numpy(dtype=float)
        valid_mask = np.isfinite(strikes) & (strikes > 0.0)
        if not np.any(valid_mask):
            continue

        log_moneyness = np.log(strikes[valid_mask] / float(forward_price))
        k_min = min(k_min, float(np.nanmin(log_moneyness)))
        k_max = max(k_max, float(np.nanmax(log_moneyness)))

    if not np.isfinite(k_min) or not np.isfinite(k_max):
        k_min, k_max = -1.25, 1.25
    elif np.isclose(k_min, k_max):
        k_min -= 0.25
        k_max += 0.25

    pad = 0.05 * (k_max - k_min)
    return np.linspace(k_min - pad, k_max + pad, points)


def resolve_surface_query_time(
    vol_surface: Any,
    t: float | str | date | pd.Timestamp,
) -> tuple[pd.Timestamp, float]:
    """Normalize maturity input and enforce strict supported-domain checks.

    Args:
        vol_surface: Fitted volatility surface interface object.
        t: Maturity as year-fraction float or date-like object.

    Returns:
        tuple[pd.Timestamp, float]: ``(expiry_timestamp, t_years)``.

    Raises:
        ValueError: If maturity is non-positive or beyond the calibrated horizon.
    """
    market = vol_surface._market
    valuation_timestamp = pd.to_datetime(market.valuation_date)
    max_expiry = max(vol_surface.expiries)
    max_supported_t = calculate_time_to_expiry(max_expiry, market.valuation_date)

    if isinstance(t, (str, date, pd.Timestamp)):
        expiry_timestamp = pd.to_datetime(t).tz_localize(None)
        t_years = float(
            calculate_time_to_expiry(expiry_timestamp, market.valuation_date)
        )
        if t_years <= 0:
            raise ValueError("Requested maturity must be strictly after valuation date.")
        if expiry_timestamp > max_expiry:
            raise ValueError(
                f"Expiry {expiry_timestamp.date()} is beyond the last fitted pillar "
                f"({max_expiry.date()}). Long-end extrapolation is not supported."
            )
        return expiry_timestamp, t_years

    t_years = float(t)
    if t_years <= 0.0:
        raise ValueError("Requested maturity t must be strictly positive.")
    if t_years > max_supported_t:
        raise ValueError(
            "Requested maturity exceeds the last fitted pillar. "
            "Long-end extrapolation is not supported."
        )

    days_to_expiry = int(round(t_years * 365.0))
    expiry_timestamp = valuation_timestamp + pd.Timedelta(days=days_to_expiry)
    return expiry_timestamp, t_years


def derive_surface_distribution_at_t(
    vol_surface: Any,
    t_years: float,
    *,
    log_moneyness_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive strike-grid PDF/CDF arrays for a given maturity.

    Args:
        vol_surface: Fitted volatility surface interface object.
        t_years: Time to maturity in years.
        log_moneyness_grid: Shared log-moneyness evaluation grid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(strikes, pdf, cdf)``.
    """
    market = vol_surface._market
    forward_price = float(vol_surface.forward_price(t_years))
    strike_grid = forward_price * np.exp(log_moneyness_grid)

    effective_rate = resolve_risk_free_rate(
        market.risk_free_rate, market.risk_free_rate_mode, t_years
    )
    interpolator = vol_surface._interpolator
    implied_vols = np.asarray(interpolator.implied_vol(strike_grid, t_years), dtype=float)
    call_prices = np.asarray(
        black76_call_price(
            forward_price,
            strike_grid,
            implied_vols,
            t_years,
            effective_rate,
        ),
        dtype=float,
    )

    cdf_values = normalized_cdf_from_call_curve(
        call_prices,
        strike_grid,
        log_moneyness_grid,
        effective_rate=effective_rate,
        time_to_expiry_years=t_years,
    )
    pdf_values, rebuilt_cdf_values = pdf_and_cdf_from_normalized_cdf(
        cdf_values,
        strike_grid,
        log_moneyness_grid,
    )
    return np.asarray(strike_grid, dtype=float), pdf_values, rebuilt_cdf_values


def build_interpolated_resolved_market(
    vol_surface: Any,
    t_years: float,
) -> ResolvedMarket:
    """Build a synthetic resolved market snapshot for maturity ``t_years``.

    Args:
        vol_surface: Fitted volatility surface interface object.
        t_years: Time to maturity in years.

    Returns:
        ResolvedMarket: Synthetic market snapshot aligned with ``t_years``.
    """
    market = vol_surface._market
    valuation_timestamp = pd.to_datetime(market.valuation_date)
    expiry_timestamp = valuation_timestamp + pd.Timedelta(days=int(round(t_years * 365.0)))

    return ResolvedMarket(
        risk_free_rate=market.risk_free_rate,
        underlying_price=float(vol_surface.forward_price(t_years)),
        valuation_date=market.valuation_date,
        dividend_yield=market.dividend_yield,
        dividend_schedule=None,
        provenance=Provenance(price="user", dividends="none"),
        source_meta={
            "interpolated": True,
            "expiry": expiry_timestamp,
            "risk_free_rate_mode": market.risk_free_rate_mode,
        },
    )


def build_probcurve_metadata(
    vol_surface: Any,
    expiry_timestamp: pd.Timestamp,
    t_years: float,
) -> dict[str, Any]:
    """Build metadata payload for a ``ProbSurface.slice(...)`` result.

    Args:
        vol_surface: Fitted volatility surface interface object.
        expiry_timestamp: Requested expiry timestamp.
        t_years: Time to expiry in years.

    Returns:
        dict[str, Any]: Metadata dictionary for ``ProbCurve.from_arrays``.
    """
    is_pillar = expiry_timestamp in vol_surface.expiries
    return {
        "interpolated": not is_pillar,
        "method": "unified_surface_probability",
        "expiry": expiry_timestamp,
        "expiry_date": expiry_timestamp.date(),
        "time_to_expiry_years": t_years,
        "days_to_expiry": int(round(t_years * 365.0)),
        "forward_price": float(vol_surface.forward_price(t_years)),
        "at_money_vol": float(vol_surface.atm_vol(t_years)),
    }


def build_daily_fan_density_frame(
    vol_surface: Any,
    *,
    log_moneyness_grid: np.ndarray,
) -> pd.DataFrame:
    """Build daily density-frame payload used by ``ProbSurface.plot_fan``.

    Args:
        vol_surface: Fitted volatility surface interface object.
        log_moneyness_grid: Shared log-moneyness evaluation grid.

    Returns:
        pd.DataFrame: Long dataframe with columns ``expiry_date``, ``strike``, ``cdf``.

    Raises:
        ValueError: If no slices are generated.
    """
    first_expiry = min(vol_surface.expiries)
    last_expiry = max(vol_surface.expiries)
    sample_expiries = pd.date_range(first_expiry, last_expiry, freq="D")

    frames: list[pd.DataFrame] = []
    for expiry in sample_expiries:
        expiry_timestamp = pd.to_datetime(expiry)
        _, t_years = resolve_surface_query_time(vol_surface, expiry_timestamp)
        strikes, _, cdf_values = derive_surface_distribution_at_t(
            vol_surface,
            t_years,
            log_moneyness_grid=log_moneyness_grid,
        )
        frame = pd.DataFrame(
            {
                "expiry_date": np.full(strikes.shape, expiry_timestamp),
                "strike": strikes,
                "cdf": cdf_values,
            }
        )
        frames.append(frame)

    if not frames:
        raise ValueError("No probability slices available for plotting")

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "build_daily_fan_density_frame",
    "build_global_log_moneyness_grid",
    "build_interpolated_resolved_market",
    "build_probcurve_metadata",
    "derive_surface_distribution_at_t",
    "resolve_surface_query_time",
]
