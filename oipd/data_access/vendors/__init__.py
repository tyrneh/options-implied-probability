"""Vendor registry for data access layer."""

from __future__ import annotations

from importlib import import_module
from typing import Dict

from oipd.data_access.vendors.base import VendorAdapter

_READER_REGISTRY: Dict[str, str] = {}
_ADAPTER_REGISTRY: Dict[str, str] = {}


def register(name: str, dotted_path: str) -> None:
    """Register *name* → *dotted_path* pointing to a vendor reader module."""
    _READER_REGISTRY[name] = dotted_path


def register_adapter(name: str, dotted_path: str) -> None:
    """Register *name* → *dotted_path* pointing to a vendor adapter module."""
    _ADAPTER_REGISTRY[name] = dotted_path


def get_reader(name: str):
    """Return the Reader class for a given vendor."""
    if name not in _READER_REGISTRY:
        raise ValueError(
            f"Unknown vendor '{name}'. Available: {list(_READER_REGISTRY)}"
        )
    module = import_module(_READER_REGISTRY[name])
    return getattr(module, "Reader")


def get_adapter(name: str) -> VendorAdapter:
    """Return an adapter instance for a given vendor."""
    if name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown vendor '{name}'. Available: {list(_ADAPTER_REGISTRY)}"
        )
    module = import_module(_ADAPTER_REGISTRY[name])
    adapter_cls = getattr(
        module, "YFinanceAdapter"
    )  # TODO: generalize class name lookup
    return adapter_cls()


# Pre-register built-in vendors
register("yfinance", "oipd.data_access.vendors.yfinance.reader")
register_adapter("yfinance", "oipd.data_access.vendors.yfinance.adapter")


__all__ = ["register", "register_adapter", "get_reader", "get_adapter"]
