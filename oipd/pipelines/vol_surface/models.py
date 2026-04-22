"""Surface model definitions and protocols."""

from __future__ import annotations

import copy
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from oipd.market_inputs import ResolvedMarket


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
        fit_warning_reports: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        """Initialize a collection of independently fitted volatility slices.

        Args:
            slices: Fitted slice payloads keyed by expiry.
            resolved_markets: Resolved market inputs keyed by expiry.
            slice_chains: Input option chains keyed by expiry.
            fit_warning_reports: Structured warning-worthy fit reports collected
                during surface fitting. Reports are facts for the interface layer
                to translate into public diagnostics and summarized warnings.
        """
        self._slices = slices
        self._resolved_markets = resolved_markets
        self._slice_chains = slice_chains
        self._fit_warning_reports = copy.deepcopy(fit_warning_reports or {})

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

    @property
    def fit_warning_reports(self) -> dict[str, list[dict[str, Any]]]:
        """Return structured warning-worthy reports from surface fitting.

        Returns:
            dict[str, list[dict[str, Any]]]: Defensive copy of reports grouped
                by warning event type.
        """
        return copy.deepcopy(self._fit_warning_reports)
