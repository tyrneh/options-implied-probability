"""Surface model definitions and protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import pandas as pd
from oipd.pipelines.market_inputs import ResolvedMarket


@runtime_checkable
class FittedSurface(Protocol):
    """Protocol for fitted volatility surface models."""

    def get_slice(self, expiry: pd.Timestamp) -> dict[str, Any]:
        """Retrieve the fitted curve and metadata for a specific expiry.

        Returns:
            dict containing:
                - "curve": The callable volatility curve
                - "metadata": Fitting metadata/diagnostics
                - "resolved_market": The ResolvedMarket used for fitting
                - "chain": The option chain DataFrame used for fitting

        Raises:
            ValueError: If the expiry is not found or supported.
        """
        ...

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        """Return the available expiries."""
        ...


class DiscreteSurface:
    """A collection of independently fitted volatility slices."""

    def __init__(
        self,
        slices: dict[pd.Timestamp, dict[str, Any]],
        resolved_markets: dict[pd.Timestamp, ResolvedMarket],
        slice_chains: dict[pd.Timestamp, pd.DataFrame],
    ) -> None:
        self._slices = slices
        self._resolved_markets = resolved_markets
        self._slice_chains = slice_chains

    def get_slice(self, expiry: pd.Timestamp) -> dict[str, Any]:
        if expiry not in self._slices:
            raise ValueError(f"Expiry {expiry} not found in fitted surface")

        return {
            "curve": self._slices[expiry]["curve"],
            "metadata": self._slices[expiry]["metadata"],
            "resolved_market": self._resolved_markets[expiry],
            "chain": self._slice_chains[expiry],
        }

    @property
    def expiries(self) -> tuple[pd.Timestamp, ...]:
        return tuple(sorted(self._slices.keys()))
