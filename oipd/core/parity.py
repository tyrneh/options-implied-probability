"""Compatibility shim for parity helpers."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable

from oipd.core.data_processing import parity as _parity


def _deprecated(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a parity helper so usage emits a deprecation warning.

    Args:
        func: Target callable from the new parity module.

    Returns:
        Wrapped callable that proxies to ``func`` while emitting a ``DeprecationWarning``.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        warnings.warn(
            "oipd.core.parity is deprecated; migrate to oipd.core.data_processing.parity",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


apply_put_call_parity_to_quotes = _deprecated(
    _parity.apply_put_call_parity_to_quotes
)
apply_put_call_parity = _deprecated(_parity.apply_put_call_parity_to_quotes)
infer_forward_from_atm = _deprecated(_parity.infer_forward_from_atm)
detect_parity_opportunity = _deprecated(_parity.detect_parity_opportunity)
preprocess_with_parity = _deprecated(_parity.preprocess_with_parity)

__all__ = [
    "apply_put_call_parity",
    "apply_put_call_parity_to_quotes",
    "detect_parity_opportunity",
    "infer_forward_from_atm",
    "preprocess_with_parity",
]
