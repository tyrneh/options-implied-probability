"""Internal models for lazy probability materialization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from oipd.market_inputs import ResolvedMarket


def snapshot_resolved_market(resolved_market: ResolvedMarket) -> ResolvedMarket:
    """Return a defensive copy of resolved market inputs.

    Args:
        resolved_market: Resolved market state to freeze for later probability
            materialization.

    Returns:
        ResolvedMarket: Copied market state with mutable nested data detached.
    """
    dividend_schedule = (
        resolved_market.dividend_schedule.copy(deep=True)
        if resolved_market.dividend_schedule is not None
        else None
    )
    return ResolvedMarket(
        risk_free_rate=resolved_market.risk_free_rate,
        underlying_price=resolved_market.underlying_price,
        valuation_date=resolved_market.valuation_timestamp,
        dividend_yield=resolved_market.dividend_yield,
        dividend_schedule=dividend_schedule,
        provenance=resolved_market.provenance,
        source_meta=copy.deepcopy(resolved_market.source_meta),
    )


@dataclass(frozen=True)
class MaterializationSpec:
    """Numerical policy for turning a probability definition into arrays.

    Attributes:
        domain: Optional explicit native PDF domain. ``None`` keeps the adaptive
            full-domain behavior.
        points: Requested native grid resolution.
    """

    domain: tuple[float, float] | None = None
    points: int = 200


@dataclass(frozen=True)
class DistributionSnapshot:
    """Materialized probability arrays and metadata.

    Attributes:
        prices: Native price grid.
        pdf_values: PDF values aligned with ``prices``.
        cdf_values: CDF values aligned with ``prices``.
        metadata: Full materialization metadata, including domain diagnostics.
    """

    prices: np.ndarray
    pdf_values: np.ndarray
    cdf_values: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Defensively copy arrays and metadata into the frozen snapshot."""
        object.__setattr__(
            self, "prices", np.array(self.prices, dtype=float, copy=True)
        )
        object.__setattr__(
            self,
            "pdf_values",
            np.array(self.pdf_values, dtype=float, copy=True),
        )
        object.__setattr__(
            self,
            "cdf_values",
            np.array(self.cdf_values, dtype=float, copy=True),
        )
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))


@dataclass(frozen=True)
class CurveProbabilityDefinition:
    """Frozen recipe for a single-expiry probability distribution.

    Attributes:
        vol_curve: Fitted volatility callable.
        resolved_market: Resolved market snapshot.
        pricing_engine: Pricing convention used for the fitted volatility curve.
        vol_metadata: Metadata from volatility fitting before probability
            materialization adds domain diagnostics.
        native_spec: Default native materialization policy.
    """

    vol_curve: Any
    resolved_market: ResolvedMarket
    pricing_engine: str
    vol_metadata: dict[str, Any]
    native_spec: MaterializationSpec

    @classmethod
    def from_vol_curve(
        cls,
        vol_curve: Any,
        *,
        native_spec: MaterializationSpec,
        metadata: dict[str, Any] | None = None,
    ) -> "CurveProbabilityDefinition":
        """Build a frozen probability definition from a fitted VolCurve object.

        Args:
            vol_curve: Fitted public ``VolCurve`` interface object.
            native_spec: Default native materialization policy.
            metadata: Optional metadata override. When omitted, the fitted
                ``VolCurve`` metadata is copied.

        Returns:
            CurveProbabilityDefinition: Frozen single-expiry probability recipe.

        Raises:
            ValueError: If the fitted curve or market snapshot is missing.
        """
        fitted_callable = getattr(vol_curve, "_vol_curve", None)
        resolved_market = getattr(vol_curve, "_resolved_market", None)
        vol_metadata = (
            metadata if metadata is not None else getattr(vol_curve, "_metadata", None)
        )
        if fitted_callable is None or resolved_market is None or vol_metadata is None:
            raise ValueError(
                "Cannot build probability definition from an unfitted VolCurve."
            )

        return cls(
            vol_curve=copy.deepcopy(fitted_callable),
            resolved_market=snapshot_resolved_market(resolved_market),
            pricing_engine=str(getattr(vol_curve, "pricing_engine")),
            vol_metadata=copy.deepcopy(vol_metadata),
            native_spec=native_spec,
        )


@dataclass(frozen=True)
class SurfaceProbabilityDefinition:
    """Frozen recipe for a probability surface.

    Attributes:
        vol_surface: Deep-copied fitted volatility surface interface object.
        expiries: Fitted pillar expiries available on the surface.
        native_spec: Default native materialization policy for each slice.
    """

    vol_surface: Any
    expiries: tuple[pd.Timestamp, ...]
    native_spec: MaterializationSpec

    @classmethod
    def from_vol_surface(
        cls,
        vol_surface: Any,
        *,
        native_spec: MaterializationSpec,
    ) -> "SurfaceProbabilityDefinition":
        """Build a frozen probability definition from a fitted VolSurface object.

        Args:
            vol_surface: Fitted public ``VolSurface`` interface object.
            native_spec: Default native materialization policy for surface
                slices.

        Returns:
            SurfaceProbabilityDefinition: Frozen surface probability recipe.

        Raises:
            ValueError: If the surface is not fitted.
        """
        if (
            getattr(vol_surface, "_model", None) is None
            or getattr(vol_surface, "_interpolator", None) is None
            or getattr(vol_surface, "_market", None) is None
            or len(vol_surface.expiries) == 0
        ):
            raise ValueError(
                "Cannot build probability definition from an unfitted VolSurface."
            )

        return cls(
            vol_surface=copy.deepcopy(vol_surface),
            expiries=tuple(pd.Timestamp(expiry) for expiry in vol_surface.expiries),
            native_spec=native_spec,
        )
