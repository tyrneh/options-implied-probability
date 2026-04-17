"""Convenience namespace for data-processing helpers.

This package primarily re-exports the preprocessing functions implemented in
its submodules so callers can import them from one stable package path.
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
    infer_forward_from_put_call_parity,
    preprocess_with_parity,
)
from .selection import filter_stale_options, select_price_column  # noqa: F401

__all__ = [
    "apply_put_call_parity",
    "detect_parity_opportunity",
    "apply_put_call_parity_to_quotes",
    "infer_forward_from_atm",
    "infer_forward_from_put_call_parity",
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
