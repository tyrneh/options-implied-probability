"""High-level interface for multi-maturity Implied Volatility surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

from oipd.calibration.ssvi_surface import (
    SSVISliceObservations,
    SSVISurfaceFit,
    calibrate_ssvi_surface,
)
from oipd.calibration.raw_svi_surface import (
    RawSVISurfaceFit,
    calibrate_raw_svi_surface,
)
from oipd.core.errors import CalculationError
from oipd.core.prep import apply_put_call_parity, compute_iv, filter_stale_options, select_price_column
from oipd.core.ssvi import phi_eta_gamma, ssvi_total_variance
from oipd.core.svi import log_moneyness, svi_total_variance, _vega_based_weights
from oipd.core.svi_types import SVICalibrationOptions
from oipd.core.vol_model import SURFACE_METHODS, VolModel
from oipd.estimator import ModelParams
from oipd.market_inputs import MarketInputs, ResolvedMarket, VendorSnapshot, resolve_market
from oipd.pricing.black76 import black76_call_price
from oipd.pricing.utils import prepare_dividends
from oipd.vendor import get_reader

SSVI_WEIGHT_CAP = 50.0
ATM_WEIGHT_MULTIPLIER = 5.0


@dataclass(frozen=True)
class SurfaceSliceData:
    """Container holding per-expiry option data and resolved market inputs."""

    expiry: date
    options: pd.DataFrame
    market: ResolvedMarket
    vendor_snapshot: Optional[VendorSnapshot]
    source: str


def _resolve_surface_vol_model(vol: Optional[VolModel]) -> VolModel:
    resolved = vol or VolModel()
    method = resolved.method or "ssvi"
    if method not in SURFACE_METHODS:
        raise ValueError(
            "VolModel.method must be one of {'ssvi', 'raw_svi'} for term surfaces"
        )
    return replace(resolved, method=method)


def _clone_market_inputs(base: MarketInputs, expiry: date, *, days_to_expiry: Optional[int]) -> MarketInputs:
    return MarketInputs(
        risk_free_rate=base.risk_free_rate,
        valuation_date=base.valuation_date,
        risk_free_rate_mode=base.risk_free_rate_mode,
        underlying_price=base.underlying_price,
        dividend_yield=base.dividend_yield,
        dividend_schedule=base.dividend_schedule,
        expiry_date=expiry,
    )


def _ensure_datetime_series(values: pd.Series, column: str) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"Column '{column}' must contain valid date/datetime values")
    return converted


def _resolve_horizon_end(valuation: date, horizon: str | int | float | timedelta) -> date:
    if isinstance(horizon, timedelta):
        days = horizon.days
    elif isinstance(horizon, int):
        days = horizon
    elif isinstance(horizon, float):
        days = int(round(horizon * 365))
    elif isinstance(horizon, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)([DWMY])\s*", horizon.upper())
        if not match:
            raise ValueError(
                "Horizon must be an int (days), float (years), timedelta, or string like '3M'/'1Y'"
            )
        magnitude = float(match.group(1))
        unit = match.group(2)
        if unit == "D":
            days = int(round(magnitude))
        elif unit == "W":
            days = int(round(magnitude * 7))
        elif unit == "M":
            days = int(round(magnitude * 30))
        else:  # unit == "Y"
            days = int(round(magnitude * 365))
    else:
        raise TypeError(
            "Horizon must be provided as str (e.g. '12M'), int days, float years, or timedelta"
        )

    if days <= 0:
        raise ValueError("Horizon must be positive")
    return valuation + timedelta(days=days)


class RNDSurface:
    """Term-structure volatility surface built from multiple expiries."""

    def __init__(
        self,
        *,
        base_market: MarketInputs,
        slices: Sequence[SurfaceSliceData],
        vol_model: VolModel,
        model: Optional[ModelParams] = None,
        source: str,
        horizon: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> None:
        self.base_market = base_market
        self._raw_slices = list(slices)
        self.vol_model = vol_model
        self.model = model or ModelParams()
        self.source = source
        self.horizon = horizon
        self.ticker = ticker

        self._prepared_observations: Tuple[SSVISliceObservations, ...]
        self._forwards: Dict[float, float]
        self._markets_by_maturity: Dict[float, ResolvedMarket]
        self._ssvi_fit: Optional[SSVISurfaceFit] = None
        self._raw_fit: Optional[RawSVISurfaceFit] = None
        self._fit_kind: str = self.vol_model.method or "ssvi"

        self._prepared_observations, self._forwards = self._prepare_observations()

        if self.vol_model.method == "ssvi":
            self._ssvi_fit = calibrate_ssvi_surface(self._prepared_observations, self.vol_model)
        elif self.vol_model.method == "raw_svi":
            options = None
            if isinstance(self.model.surface_options, SVICalibrationOptions):  # type: ignore[name-defined]
                options = self.model.surface_options
            self._raw_fit = calibrate_raw_svi_surface(
                self._prepared_observations,
                self.vol_model,
                options=options,
            )
        else:  # pragma: no cover - guarded by VolModel validation
            raise NotImplementedError(f"Unsupported surface method {self.vol_model.method}")

    def _prepare_single_slice(
        self, slice_data: SurfaceSliceData
    ) -> tuple[SSVISliceObservations, float, ResolvedMarket]:
        """Preprocess raw option quotes into total variance observations."""

        resolved = slice_data.market
        valuation_date = resolved.valuation_date
        options = slice_data.options.copy()

        if self.model.pricing_engine == "bs":
            effective_spot, effective_dividend = prepare_dividends(
                underlying=resolved.underlying_price,
                dividend_schedule=resolved.dividend_schedule,
                dividend_yield=resolved.dividend_yield,
                r=resolved.risk_free_rate,
                valuation_date=valuation_date,
            )
        else:
            effective_spot = resolved.underlying_price
            effective_dividend = None

        filtered_quotes = filter_stale_options(
            options,
            valuation_date,
            self.model.max_staleness_days,
            emit_warning=False,
        )

        if filtered_quotes.empty:
            raise CalculationError("No valid option prices after staleness filtering")

        try:
            parity_adjusted, parity_forward = apply_put_call_parity(
                filtered_quotes,
                effective_spot,
                resolved,
            )
        except ValueError as exc:
            raise CalculationError("Failed to infer forward after put-call parity preprocessing") from exc

        if self.model.pricing_engine == "black76":
            if parity_forward is None:
                raise CalculationError(
                    "Black-76 engine requires both calls and puts to infer the forward"
                )
            underlying_for_iv = parity_forward
        else:
            underlying_for_iv = effective_spot

        priced = select_price_column(parity_adjusted, self.model.price_method)
        if priced.empty:
            raise CalculationError("No valid option prices after preprocessing")

        iv_df = compute_iv(
            priced,
            underlying_for_iv,
            resolved,
            self.model.solver,
            self.model.pricing_engine,
            effective_dividend,
        )

        bid_iv = None
        ask_iv = None
        strikes_sorted = iv_df["strike"].to_numpy(dtype=float)

        if "bid" in priced.columns:
            bid_frame = priced.loc[(priced["bid"].notna()) & (priced["bid"] > 0.0)].copy()
            if not bid_frame.empty:
                bid_frame = bid_frame.assign(price=bid_frame["bid"])
                bid_iv_df = compute_iv(
                    bid_frame,
                    underlying_for_iv,
                    resolved,
                    self.model.solver,
                    self.model.pricing_engine,
                    effective_dividend,
                )
                bid_iv_series = bid_iv_df.set_index("strike")["iv"]
                bid_iv = bid_iv_series.reindex(strikes_sorted).to_numpy(dtype=float)

        if "ask" in priced.columns:
            ask_frame = priced.loc[(priced["ask"].notna()) & (priced["ask"] > 0.0)].copy()
            if not ask_frame.empty:
                ask_frame = ask_frame.assign(price=ask_frame["ask"])
                ask_iv_df = compute_iv(
                    ask_frame,
                    underlying_for_iv,
                    resolved,
                    self.model.solver,
                    self.model.pricing_engine,
                    effective_dividend,
                )
                ask_iv_series = ask_iv_df.set_index("strike")["iv"]
                ask_iv = ask_iv_series.reindex(strikes_sorted).to_numpy(dtype=float)

        strikes = strikes_sorted
        iv_values = iv_df["iv"].to_numpy(dtype=float)
        maturity_years = resolved.days_to_expiry / 365.0
        if maturity_years <= 0:
            raise CalculationError("Time to expiry must be positive for surface calibration")

        valid_mask = np.isfinite(strikes) & np.isfinite(iv_values)
        strikes = strikes[valid_mask]
        iv_values = iv_values[valid_mask]

        log_mny = log_moneyness(strikes, underlying_for_iv)
        total_variance = np.square(iv_values) * maturity_years

        finite_mask = np.isfinite(log_mny) & np.isfinite(total_variance)
        log_mny = log_mny[finite_mask]
        total_variance = total_variance[finite_mask]
        if log_mny.size < 3:
            raise CalculationError("Insufficient viable quotes for surface calibration")

        if bid_iv is not None:
            bid_iv = bid_iv[finite_mask]
        if ask_iv is not None:
            ask_iv = ask_iv[finite_mask]

        weights_array, _ = _vega_based_weights(
            log_mny,
            total_variance,
            maturity_years,
            "vega",
            SSVI_WEIGHT_CAP,
        )
        weights = np.asarray(weights_array, dtype=float)
        if weights.shape != total_variance.shape:
            raise CalculationError("Failed to construct calibration weights")

        if weights.size:
            atm_index = int(np.argmin(np.abs(log_mny)))
            mean_weight = float(np.mean(weights))
            boosted = max(weights[atm_index], ATM_WEIGHT_MULTIPLIER * max(mean_weight, 1e-6))
            weights[atm_index] = float(np.clip(boosted, 1e-6, SSVI_WEIGHT_CAP))

        order = np.argsort(log_mny)
        log_mny = log_mny[order]
        total_variance = total_variance[order]
        weights = weights[order]
        if bid_iv is not None:
            bid_iv = np.asarray(bid_iv, dtype=float)[order]
        if ask_iv is not None:
            ask_iv = np.asarray(ask_iv, dtype=float)[order]

        observation = SSVISliceObservations(
            maturity=maturity_years,
            log_moneyness=log_mny,
            total_variance=total_variance,
            weights=weights,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
        )

        return observation, float(underlying_for_iv), resolved

    def _prepare_observations(self) -> tuple[Tuple[SSVISliceObservations, ...], Dict[float, float]]:
        """Prepare all expiries for SSVI calibration."""

        observations: List[SSVISliceObservations] = []
        forwards: Dict[float, float] = {}
        markets: Dict[float, ResolvedMarket] = {}

        sorted_slices = sorted(
            self._raw_slices,
            key=lambda item: item.market.days_to_expiry,
        )

        for slice_data in sorted_slices:
            obs, forward, resolved = self._prepare_single_slice(slice_data)
            observations.append(obs)
            forwards[obs.maturity] = forward
            markets[obs.maturity] = resolved

        if not observations:
            raise CalculationError("No valid expiries available for surface calibration")

        self._markets_by_maturity = markets
        return tuple(observations), forwards

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        market: MarketInputs,
        *,
        expiry_col: str = "expiry",
        strike_col: str = "strike",
        price_cols: Sequence[str] | None = ("bid", "ask"),
        model: Optional[ModelParams] = None,
        vol: Optional[VolModel] = None,
    ) -> "RNDSurface":
        if expiry_col not in df.columns:
            raise ValueError(f"DataFrame must contain an '{expiry_col}' column")

        expiry_series = _ensure_datetime_series(df[expiry_col], expiry_col)
        valuation = market.valuation_date
        slices: List[SurfaceSliceData] = []

        vol_model = _resolve_surface_vol_model(vol)

        for expiry_value, group in df.groupby(expiry_series.dt.date):
            if not isinstance(expiry_value, date):
                expiry_value = expiry_value.to_pydatetime().date()  # type: ignore[assignment]

            days_to_expiry = (expiry_value - valuation).days
            if days_to_expiry <= 0:
                raise ValueError("All expiries must be after the valuation date")

            slice_inputs = _clone_market_inputs(
                market, expiry_value, days_to_expiry=days_to_expiry
            )

            resolved = resolve_market(slice_inputs, vendor=None, mode="strict")
            group_copy = group.copy()
            slices.append(
                SurfaceSliceData(
                    expiry=expiry_value,
                    options=group_copy,
                    market=resolved,
                    vendor_snapshot=None,
                    source="dataframe",
                )
            )

        if not slices:
            raise ValueError("No valid expiries were found in the provided DataFrame")

        return cls(
            base_market=market,
            slices=slices,
            vol_model=vol_model,
            model=model,
            source="dataframe",
            horizon=None,
        )

    @classmethod
    def from_ticker(
        cls,
        ticker: str,
        market: MarketInputs,
        *,
        horizon: str | int | float | timedelta = "12M",
        model: Optional[ModelParams] = None,
        vol: Optional[VolModel] = None,
        vendor: str = "yfinance",
        fill: str = "missing",
        cache_enabled: bool = True,
        cache_ttl_minutes: int = 15,
    ) -> "RNDSurface":
        valuation = market.valuation_date
        horizon_end = _resolve_horizon_end(valuation, horizon)
        vol_model = _resolve_surface_vol_model(vol)

        reader_cls = get_reader(vendor)
        try:
            expiry_strings = reader_cls.list_expiry_dates(ticker)
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise NotImplementedError(
                f"Vendor '{vendor}' does not support listing expiries"
            ) from exc

        selected_expiries: List[date] = []
        for expiry_str in expiry_strings:
            expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            if expiry_dt <= horizon_end:
                selected_expiries.append(expiry_dt)

        if not selected_expiries:
            raise ValueError(
                "No expiries within the requested horizon were found for the ticker"
            )

        slices: List[SurfaceSliceData] = []

        for expiry_dt in selected_expiries:
            reader_kwargs = {
                "cache_enabled": cache_enabled,
                "cache_ttl_minutes": cache_ttl_minutes,
            }
            try:
                reader = reader_cls(**reader_kwargs)
            except TypeError:
                reader = reader_cls()

            ticker_expiry = f"{ticker}:{expiry_dt.strftime('%Y-%m-%d')}"
            options_df = reader.read(ticker_expiry, column_mapping=None)

            underlying = options_df.attrs.get("underlying_price")
            dividend_yield = options_df.attrs.get("dividend_yield")
            dividend_schedule = options_df.attrs.get("dividend_schedule")

            snapshot = VendorSnapshot(
                asof=datetime.now(),
                vendor=vendor,
                underlying_price=underlying,
                dividend_yield=dividend_yield,
                dividend_schedule=dividend_schedule,
            )

            days_to_expiry = (expiry_dt - valuation).days
            if days_to_expiry <= 0:
                continue

            slice_inputs = _clone_market_inputs(
                market, expiry_dt, days_to_expiry=days_to_expiry
            )
            resolved = resolve_market(slice_inputs, snapshot, mode=fill)  # type: ignore[arg-type]

            slices.append(
                SurfaceSliceData(
                    expiry=expiry_dt,
                    options=options_df,
                    market=resolved,
                    vendor_snapshot=snapshot,
                    source="ticker",
                )
            )

        if not slices:
            raise ValueError("Failed to assemble any slices within the requested horizon")

        return cls(
            base_market=market,
            slices=slices,
            vol_model=vol_model,
            model=model,
            source="ticker",
            horizon=str(horizon),
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # Accessors / placeholders for future implementation
    # ------------------------------------------------------------------

    @property
    def slices(self) -> Tuple[SurfaceSliceData, ...]:
        return tuple(self._raw_slices)

    @property
    def expiries(self) -> List[date]:
        return [slice_data.expiry for slice_data in self._raw_slices]

    def _ensure_ssvi_fit(self) -> SSVISurfaceFit:
        if self._ssvi_fit is None:
            raise CalculationError("SSVI calibration has not been executed")
        return self._ssvi_fit

    def _ensure_raw_fit(self) -> RawSVISurfaceFit:
        if self._raw_fit is None:
            raise CalculationError("Raw SVI calibration has not been executed")
        return self._raw_fit

    def _infer_forward(self, t: float, override: Optional[float]) -> float:
        if override is not None:
            return float(override)
        if not self._forwards:
            raise ValueError("No forward levels available for surface evaluation")
        tolerance = 5e-3
        items = sorted(self._forwards.items())
        for maturity, forward in items:
            if abs(maturity - t) < tolerance:
                return forward
        if t <= items[0][0]:
            return items[0][1]
        if t >= items[-1][0]:
            return items[-1][1]
        for (t0, f0), (t1, f1) in zip(items[:-1], items[1:]):
            if t0 <= t <= t1:
                weight = (t - t0) / (t1 - t0)
                return (1.0 - weight) * f0 + weight * f1
        return items[-1][1]

    def _total_variance_raw(self, k: np.ndarray, t: float) -> np.ndarray:
        raw_fit = self._ensure_raw_fit()
        slices = sorted(raw_fit.slices, key=lambda s: s.maturity)
        if not slices:
            raise CalculationError("Surface contains no raw SVI slices")
        alpha = float(getattr(raw_fit, "alpha", 0.0))
        if len(slices) == 1:
            return svi_total_variance(k, slices[0].params) + alpha * float(t)

        if t <= slices[0].maturity:
            return svi_total_variance(k, slices[0].params) + alpha * float(t)
        if t >= slices[-1].maturity:
            return svi_total_variance(k, slices[-1].params) + alpha * float(t)

        for left, right in zip(slices[:-1], slices[1:]):
            if left.maturity <= t <= right.maturity:
                weight = (t - left.maturity) / (right.maturity - left.maturity)
                w_left = svi_total_variance(k, left.params)
                w_right = svi_total_variance(k, right.params)
                blended = (1.0 - weight) * w_left + weight * w_right
                return blended + alpha * float(t)

        return svi_total_variance(k, slices[-1].params) + alpha * float(t)

    def plot_iv(
        self,
        *,
        maturities: Optional[Sequence[float]] = None,
        num_points: int = 200,
        x_axis: Literal["log_moneyness", "strike"] = "log_moneyness",
        figsize: tuple[float, float] = (10.0, 5.0),
        title: Optional[str] = None,
        style: Literal["publication", "default"] = "publication",
        source: Optional[str] = None,
        view: Literal["overlay", "grid"] = "overlay",
    ):
        """Plot implied-volatility slices across maturities.

        Args:
            maturities: Optional sequence of maturities (year fractions) to plot.
            num_points: Number of points in the evaluation grid per slice.
            x_axis: ``"log_moneyness"`` (default) or ``"strike"`` axis.
            figsize: Matplotlib figure size.
            title: Optional figure title.
            style: Plot styling configuration.
            source: Optional footer text (publication style only).
            view: ``"overlay"`` plots all slices on a single chart; ``"grid"`` renders
                a matrix of subplots with bid/ask envelopes per slice.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from exc

        style_axes = None
        if style == "publication":
            from oipd.graphics.publication import (
                _apply_publication_style,
                _style_publication_axes,
            )

            _apply_publication_style(plt)
            style_axes = _style_publication_axes

        observations = self._prepared_observations
        if not observations:
            raise CalculationError("Surface calibration has no observations to plot")

        available_maturities = [obs.maturity for obs in observations]
        if maturities is None:
            plot_maturities = available_maturities
        else:
            plot_maturities = []
            for t in maturities:
                tolerance = 5e-3
                matched = None
                for available in available_maturities:
                    if abs(available - t) < tolerance:
                        matched = available
                        break
                if matched is None:
                    matched = float(t)
                plot_maturities.append(matched)

        if num_points < 5:
            raise ValueError("num_points must be at least 5 for plotting")

        all_k = np.concatenate([obs.log_moneyness for obs in observations])
        k_min = float(np.min(all_k))
        k_max = float(np.max(all_k))
        if not np.isfinite(k_min) or not np.isfinite(k_max):
            raise CalculationError("Invalid log-moneyness range for plotting")
        padding = 0.05 * (k_max - k_min if k_max > k_min else 1.0)
        k_grid = np.linspace(k_min - padding, k_max + padding, num_points)

        axis_mode = x_axis.lower()
        if axis_mode not in {"log_moneyness", "strike"}:
            raise ValueError("x_axis must be 'log_moneyness' or 'strike'")

        def _axis_values(k: np.ndarray, forward: float) -> np.ndarray:
            if axis_mode == "log_moneyness":
                return k
            return forward * np.exp(k)

        if view == "overlay":
            fig, ax = plt.subplots(figsize=figsize)

            for maturity in plot_maturities:
                forward = self._infer_forward(maturity, None)
                total_var = self.total_variance(k_grid, maturity)
                iv_curve = np.sqrt(np.maximum(total_var / max(maturity, 1e-8), 1e-12))
                label_days = int(round(maturity * 365))
                label = f"{label_days}d"
                ax.plot(_axis_values(k_grid, forward), iv_curve, label=label)

            ax.set_xlabel("Log Moneyness" if axis_mode == "log_moneyness" else "Strike")
            ax.set_ylabel("Implied Volatility")
            display_title = title or "Implied Volatility Surface"
            ax.set_title(display_title)
            legend = ax.legend(loc="best")
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color("black")

            if style_axes is not None:
                style_axes(ax)
            if source and style == "publication":
                fig.text(0.99, 0.01, source, ha="right", va="bottom", fontsize=8, alpha=0.6)

            return fig

        if view != "grid":
            raise ValueError("view must be either 'overlay' or 'grid'")

        num_slices = len(plot_maturities)
        if num_slices == 0:
            raise CalculationError("Surface calibration has no maturities to plot")

        cols = min(3, num_slices)
        rows = math.ceil(num_slices / cols)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

        flat_axes = axes.flatten()
        for ax in flat_axes[num_slices:]:
            ax.axis("off")

        for ax, maturity in zip(flat_axes, plot_maturities):
            forward = self._infer_forward(maturity, None)
            total_var = self.total_variance(k_grid, maturity)
            iv_curve = np.sqrt(np.maximum(total_var / max(maturity, 1e-8), 1e-12))
            ax.plot(_axis_values(k_grid, forward), iv_curve, color="C0", label="model")

            obs = None
            for candidate in observations:
                if abs(candidate.maturity - maturity) < 5e-3:
                    obs = candidate
                    break
            if obs is not None:
                x_obs = _axis_values(obs.log_moneyness, forward)
                bid_iv = getattr(obs, "bid_iv", None)
                ask_iv = getattr(obs, "ask_iv", None)
                if bid_iv is not None and ask_iv is not None:
                    mask = np.isfinite(bid_iv) & np.isfinite(ask_iv)
                    if mask.any():
                        ax.fill_between(
                            x_obs[mask],
                            bid_iv[mask],
                            ask_iv[mask],
                            color="C2",
                            alpha=0.2,
                            label="bid/ask",
                        )
                elif bid_iv is not None:
                    mask = np.isfinite(bid_iv)
                    if mask.any():
                        ax.scatter(x_obs[mask], bid_iv[mask], color="C2", alpha=0.6, label="bid")
                elif ask_iv is not None:
                    mask = np.isfinite(ask_iv)
                    if mask.any():
                        ax.scatter(x_obs[mask], ask_iv[mask], color="C3", alpha=0.6, label="ask")

            ax.set_xlabel("Log Moneyness" if axis_mode == "log_moneyness" else "Strike")
            ax.set_ylabel("Implied Volatility")
            label_days = int(round(maturity * 365))
            ax.set_title(f"{label_days}d")
            legend = ax.legend(loc="best")
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color("black")
            if style_axes is not None:
                style_axes(ax)

        if title:
            fig.suptitle(title)
        if source and style == "publication":
            fig.text(0.99, 0.01, source, ha="right", va="bottom", fontsize=8, alpha=0.6)

        return fig

    def total_variance(self, k: np.ndarray | Iterable[float], t: float) -> np.ndarray:
        if self._fit_kind == "ssvi":
            fit = self._ensure_ssvi_fit()
            theta_t = max(float(fit.theta_interpolator(t)), 1e-6)
            w = ssvi_total_variance(
                np.asarray(k, dtype=float),
                theta_t,
                fit.params.rho,
                fit.params.eta,
                fit.params.gamma,
            )
            # Apply Gatheral–Jacquier Theorem 4.3 alpha-tilt if provided
            alpha = getattr(fit.params, "alpha", 0.0)
            if alpha:
                w = w + float(alpha) * float(t)
            return w
        if self._fit_kind == "raw_svi":
            return self._total_variance_raw(np.asarray(k, dtype=float), float(t))
        raise CalculationError(f"Unknown surface fit kind {self._fit_kind}")

    def iv(
        self,
        K: np.ndarray | Iterable[float],
        t: float,
        F_t: Optional[float] = None,
    ) -> np.ndarray:
        """Return implied volatility for strikes ``K`` at maturity ``t``."""

        forward = self._infer_forward(float(t), F_t)
        k = np.log(np.asarray(K, dtype=float) / float(forward))
        total_var = self.total_variance(k, t)
        return np.sqrt(np.maximum(total_var / max(t, 1e-8), 1e-12))

    def price(
        self,
        K: np.ndarray | Iterable[float],
        t: float,
        F_t: Optional[float] = None,
        *,
        call: bool = True,
    ) -> np.ndarray:
        """Compute option prices using the calibrated surface."""

        if self.model.pricing_engine != "black76":
            raise NotImplementedError("Only Black-76 pricing is currently supported for surfaces")

        strikes = np.asarray(K, dtype=float)
        forward = self._infer_forward(float(t), F_t)
        sigma = self.iv(strikes, t, forward)
        prices = black76_call_price(forward, strikes, sigma, float(t), self.base_market.risk_free_rate)
        if call:
            return prices
        df = np.exp(-self.base_market.risk_free_rate * float(t))
        return prices - df * (forward - strikes)

    def slice(self, t: float | date) -> "RND":  # type: ignore[name-defined]
        raise NotImplementedError("Slice extraction is not yet implemented")

    def check_no_arbitrage(self) -> Dict[str, float]:
        """Return diagnostic margins for the calibrated surface."""
        if self._fit_kind == "ssvi":
            fit = self._ensure_ssvi_fit()
            summary = {
                "objective": fit.objective,
                "min_calendar_margin": fit.calendar_margin,
                "calendar_margins": fit.calendar_margins,
            }
            summary.update(fit.inequality_margins)
            return summary

        raw_fit = self._ensure_raw_fit()
        return {
            "objective": raw_fit.objective,
            "min_calendar_margin": raw_fit.min_calendar_margin,
            "min_butterfly": raw_fit.min_butterfly,
            "alpha": raw_fit.alpha,
            "raw_calendar_margins": raw_fit.raw_calendar_margins or [],
            "raw_calendar_deltas": raw_fit.raw_calendar_deltas or [],
        }
