from __future__ import annotations

"""Registry + convenience helpers for option pricing engines.

Add any new pricing model here and expose it through *get_pricer()* so the
rest of the codebase can stay unaware of the implementation.
"""

from typing import Callable, Dict

# ---------------------------------------------------------------------------
# Import concrete pricers (keep the import cost minimal – each module should be
# lightweight).
# ---------------------------------------------------------------------------
from .european import european_call_price

# Map *engine_name* -> pricing function.  Naming convention keeps it short but
# descriptive.  Users will refer to these strings via ModelParams.pricing_engine.
_PRICERS: Dict[str, Callable] = {
    "bs": european_call_price,  # Black-Scholes European call
}


def get_pricer(engine: str = "bs") -> Callable:
    """Return a pricing function by *engine* key.

    Parameters
    ----------
    engine: str, default "bs"
        Identifier registered in the internal `_PRICERS` mapping.

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
