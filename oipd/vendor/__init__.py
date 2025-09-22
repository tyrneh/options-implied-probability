from __future__ import annotations

"""Registry of data-vendor reader classes.

Each vendor package should call *register()* at import time.  Users then pick a
vendor by string, eg. `get_reader("yfinance")`.
"""

from importlib import import_module
from typing import Type, Dict

_REGISTRY: Dict[str, str] = {}


def register(name: str, dotted_path: str) -> None:
    """Register *name* → *dotted_path* pointing to module containing a Reader class.

    The module **must** expose a class named ``Reader`` implementing *AbstractReader*.
    """
    _REGISTRY[name] = dotted_path


def get_reader(name: str):
    """Return the Reader class for a given vendor name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown vendor '{name}'. Available: {list(_REGISTRY)}"
        )
    module = import_module(_REGISTRY[name])
    # convention: module defines `Reader` class to instantiate
    return getattr(module, "Reader")


# ------------------------------------------------------------------
# Pre-register built-in vendors
# ------------------------------------------------------------------
register("yfinance", "oipd.vendor.yfinance.reader")
