"""Data preprocessing scaffolding package.

This namespace will eventually host the refactored preprocessing helpers.
While the migration is in flight, it simply re-exports the legacy functions
so early adopters can rely on the new module paths without breaking.
"""

from . import dividends, iv, moneyness, parity, selection, validation  # noqa: F401
from .iv import (
    compute_iv,
    smooth_iv,
    bs_iv_brent_method,
    bs_iv_newton_method,
    black76_iv_brent_method,
)  # noqa: F401
from .parity import (  # noqa: F401
    apply_put_call_parity,
    apply_put_call_parity_to_quotes,
    detect_parity_opportunity,
    infer_forward_from_atm,
    preprocess_with_parity,
)
from .selection import filter_stale_options, select_price_column  # noqa: F401

__all__ = [
    "apply_put_call_parity",
    "detect_parity_opportunity",
    "apply_put_call_parity_to_quotes",
    "infer_forward_from_atm",
    "preprocess_with_parity",
    "filter_stale_options",
    "select_price_column",
    "compute_iv",
    "smooth_iv",
    "bs_iv_brent_method",
    "bs_iv_newton_method",
    "black76_iv_brent_method",
    "parity",
    "selection",
    "iv",
    "moneyness",
    "dividends",
    "validation",
]
