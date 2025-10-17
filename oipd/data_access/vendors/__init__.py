"""Vendor registry for data access layer."""

from __future__ import annotations

from importlib import import_module
from typing import Dict

_REGISTRY: Dict[str, str] = {}


def register(name: str, dotted_path: str) -> None:
    """Register *name* → *dotted_path* pointing to a vendor reader module."""

    _REGISTRY[name] = dotted_path


def get_reader(name: str):
    """Return the Reader class for a given vendor."""

    if name not in _REGISTRY:
        raise ValueError(f"Unknown vendor '{name}'. Available: {list(_REGISTRY)}")
    module = import_module(_REGISTRY[name])
    return getattr(module, "Reader")


# Pre-register built-in vendors
register("yfinance", "oipd.data_access.vendors.yfinance.reader")


__all__ = ["register", "get_reader"]
