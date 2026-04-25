from __future__ import annotations

"""Low-level registry for option-pricing kernels.

The high-level ``VolCurve`` and ``VolSurface`` APIs do not expose pricing-engine
selection. They use the forward-based Black-76 path internally. The engine
strings in this module are retained for internal pipelines, legacy compatibility,
and focused pricing tests.
"""

from typing import Callable, Dict

# ---------------------------------------------------------------------------
# Import concrete pricers (keep the import cost minimal – each module should be
# lightweight).
# ---------------------------------------------------------------------------
from .black_scholes import black_scholes_call_price
from .black76 import black76_call_price

# Map internal engine key -> pricing function.
_PRICERS: Dict[str, Callable] = {
    "bs": black_scholes_call_price,  # Black-Scholes European call
    "black76": black76_call_price,  # Black-76 forward call
}


def get_pricer(engine: str = "bs") -> Callable:
    """Return a low-level pricing function by internal engine key.

    The historical ``"bs"`` default is kept for compatibility with direct
    low-level calls. High-level public workflows pass their internal engine
    explicitly and should not ask users to choose one.

    Parameters
    ----------
    engine: str, default "bs"
        Identifier registered in the internal ``_PRICERS`` mapping.

    Returns
    -------
    Callable
        A function with signature `(S, K, sigma, t, r, q=0.0) -> ndarray` that
        prices a European call option (or whatever the specific model implies).
    """
    try:
        return _PRICERS[engine]
    except KeyError as exc:  # pragma: no cover – defensive
        raise ValueError(
            f"Unknown pricing engine '{engine}'. Available: {list(_PRICERS.keys())}"
        ) from exc
